#!/usr/bin/env python3
from __future__ import annotations

import csv
import functools
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.optimize import BFGS
from deepmd.calculator import DP

try:
    from hiphive.calculators.zbl import ZBLCalculator
except ImportError:
    ZBLCalculator = None  # type: ignore[assignment]

# from ase.calculators.lj import LennardJones
from sansde.operators import (
    crossover,
    local_optimize,
    local_transformation,
    mutate,
    random_cluster,
    select,
    similarity_function,
    sphere_cut_splice,
)
from sansde.pop import Population


# 初始化参数: 默认值内联, 可由运行目录下的 params.json 覆盖, 缺失的键保持默认值.
def load_config(params_path: Path) -> None:
    """Load run configuration from a JSON file and build the potential.

    Parameters
    ----------
    params_path : pathlib.Path
        Parameter JSON file. A missing file keeps the built-in defaults.

    Returns
    -------
    None
        The configuration is stored in module-level names used by ``main``.
    """
    global params, element, a, atom_min, atom_max
    global init_dir, init_pop_dir, best_dir, keep_snapshots, pop_dir
    global pop_size, generations, stagnation_limit, ediff
    global similarity_reset, similarity_energy_reset, candidate_size
    global DP_MODEL, use_zbl, ZBL_CUTOFF, ZBL_SKIN
    global fmax, steps, potential, relax

    try:
        with params_path.open('r', encoding='utf-8') as f:
            params = json.load(f)
        if not isinstance(params, dict):
            raise TypeError(f"{params_path} must contain a JSON object")
        print(f"load parameters from {params_path}")
    except FileNotFoundError:
        params = {}

    # 体系配置
    element = str(params.get('element', 'Au'))
    a = float(params.get('a', 2.88))
    atom_min = int(params.get('atom_min', 3))
    atom_max = int(params.get('atom_max', 5))

    # IO
    init_dir = Path(params.get('init_dir', 'init'))
    init_pop_dir = Path(params.get('init_pop_dir', '')) # 非空时忽略 init_dir
    best_dir = Path(params.get('best_dir', 'best'))
    keep_snapshots = int(params.get('keep_snapshots', 1))
    pop_dir = Path(params.get('pop_dir', 'Populations'))

    # 全局优化设置
    pop_size = int(params.get('pop_size', 12))
    generations = int(params.get('generations', 500))
    stagnation_limit = int(params.get('stagnation_limit', 10))
    ediff = float(params.get('ediff', 1e-5))
    similarity_reset = float(params.get('similarity_reset', 0.9))
    similarity_energy_reset = float(params.get('similarity_energy_reset', 0.95))
    candidate_size = int(params.get('candidate_size', 2))

    # 势能面
    # DP (+ 可选 ZBL) 势: DP 负责长程/成键描述, hiphive 的 ZBL 提供短程排斥, 避免
    # 原子重叠. use_zbl=True 时两者能量/力由 SumCalculator 直接相加:
    # E_total = E_DP + E_ZBL; use_zbl=False 时仅使用 DP.
    # 注意: hiphive 的 ZBLCalculator 内部用 ASE NeighborList, 其有效作用距离约为
    #   r_eff = cutoff + 2 * skin
    DP_MODEL = Path(params.get('DP_MODEL', "/home/linyj/02-linyj/potentials/dp/Au3c.pth"))
    use_zbl = bool(params.get('use_zbl', True))
    ZBL_CUTOFF = float(params.get('ZBL_CUTOFF', 1.5))
    ZBL_SKIN = float(params.get('ZBL_SKIN', 0))
    if use_zbl and ZBLCalculator is None:
        print("Warning: hiphive is not installed; ZBL is disabled.")
        use_zbl = False

    # 优化器参数: 收敛判据与最大优化步数
    fmax = float(params.get('fmax', 0.05))
    steps = int(params.get('steps', 300))

    # 输出全部初始化参数, 便于确认本次运行配置.
    print("initialized parameters:")
    for name, value in (
        ("element", element),
        ("a", a),
        ("atom_min", atom_min),
        ("atom_max", atom_max),
        ("init_dir", init_dir),
        ("init_pop_dir", init_pop_dir),
        ("best_dir", best_dir),
        ("keep_snapshots", keep_snapshots),
        ("pop_dir", pop_dir),
        ("pop_size", pop_size),
        ("generations", generations),
        ("stagnation_limit", stagnation_limit),
        ("ediff", ediff),
        ("similarity_reset", similarity_reset),
        ("similarity_energy_reset", similarity_energy_reset),
        ("candidate_size", candidate_size),
        ("DP_MODEL", DP_MODEL),
        ("use_zbl", use_zbl),
        ("ZBL_CUTOFF", ZBL_CUTOFF),
        ("ZBL_SKIN", ZBL_SKIN),
        ("fmax", fmax),
        ("steps", steps),
    ):
        print(f"  {name} = {value}")

    pes = DP(model=DP_MODEL)
    if use_zbl:
        zbl = ZBLCalculator(cutoff=ZBL_CUTOFF, skin=ZBL_SKIN)
        potential = SumCalculator([pes, zbl])
    else:
        potential = pes

    # LJ 势: 用于 N=3-5 小团簇的优化测试, 需要时取消下面注释并注释上面的混合势.
    # potential = LennardJones(epsilon=0.4443, sigma=2.6367)

    relax = functools.partial(
        local_optimize,
        cell=None,
        # 当前使用上面的 DP(+ZBL) 势; 如需 LJ, 改为 potential=LennardJones(...).
        potential = potential, # type: ignore
        optimizer=BFGS,
        fmax=fmax,
        steps=steps,
    )


stats_path = Path("opt_stat.csv")
run_start: float | None = None


def append_stats_row(
    symbol: str,
    atom: int,
    gen: int,
    pop: Population,
    best_energy: float,
    best_pg: str | None,
    best_drop: float | None,
    similarity: float | None,
    stagnation_count: int,
    candidate_size: int,
    elapsed_time: float,
) -> None:
    """Append one generation's statistics to the run CSV.

    The CSV is created on first write and appended afterwards. At most three
    clusters are reported (``e_best`` plus two sub-optimal candidates), so the
    number of sub-optimal columns is ``min(candidate_size - 1, 2)``. When more
    candidates are available, the first ones in the candidate list are written.

    Parameters
    ----------
    symbol : str
        Cluster chemical formula, e.g. ``"Au13"``.
    atom : int
        Number of atoms in the cluster.
    gen : int
        Generation index.
    pop : Population
        Selected population; its ``candidate`` list is written.
    best_energy : float
        Lowest energy of the generation.
    best_pg : str | None
        Point group of the best cluster.
    best_drop : float | None
        Best-energy decrease relative to the previous generation.
        ``None`` on the first generation.
    similarity : float | None
        Similarity between the current and previous best cluster.
        ``None`` on the first generation.
    stagnation_count : int
        Consecutive generations without structural change.
    candidate_size : int
        Requested number of candidate entries (best + sub-optimal). The output
        is capped at three entries in total.
    elapsed_time : float
        Total program run time in seconds.
    """
    candidates = list(pop.candidate)
    n_sub = min(max(candidate_size - 1, 0), 2)
    # candidate[0] 是当前候选最优, 按列表顺序直接取前两个次优候选.
    sub_candidates = candidates[1:1 + n_sub]
    sub_energies = [float(cluster.info['energy']) for cluster in sub_candidates]
    sub_pgs = [
        "" if cluster.info.get('PG') is None else str(cluster.info['PG'])
        for cluster in sub_candidates
    ]

    header = ["symbol", "atoms", "gen", "e_best"]
    header.extend(f"e_cand{i}" for i in range(1, n_sub + 1))
    header.extend(["delta_e", "PG"])
    header.extend(f"PG_cand{i}" for i in range(1, n_sub + 1))
    header.extend(["S", "fp", "p", "CRm", "best_streak", "time"])

    row = [symbol, atom, gen, f"{best_energy:.8f}"]
    row.extend(
        "" if energy is None else f"{energy:.8f}"
        for energy in sub_energies
    )
    row.extend([""] * (n_sub - len(sub_energies)))
    row.append("" if best_drop is None else f"{best_drop:.8f}")
    row.append("" if best_pg is None else str(best_pg))
    row.extend(sub_pgs)
    row.extend([""] * (n_sub - len(sub_pgs)))
    row.append("" if similarity is None else f"{similarity:.2f}")
    row.extend([
        f"{pop.fp:.6f}",
        f"{pop.p:.6f}",
        f"{pop.crm:.6f}",
        stagnation_count,
        f"{elapsed_time:.0f}",
    ])

    write_header = not stats_path.exists() or stats_path.stat().st_size == 0
    with stats_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def main(atom: int) -> None:
    system_start = time.perf_counter()
    # 1) 初始化种群: init_pop_dir 非空时按尺寸加载给定种群;
    #    否则从 init/ 目录加载单个最优结构, 再随机补足至 pop_size.
    expected_symbol = f"{element}{atom}"
    pop = Population(symbol=expected_symbol)
    init_loaded = False
    init_from_pop = False

    # Path("") 会被规范化为 '.', 以此判断 init_pop_dir 是否为空.
    if init_pop_dir != Path('.'):
        # 文件名支持 {element}{atom}.json、{element}{atom}_gen3.json 等;
        # 逐个尝试, 用化学式校验, 不匹配则继续下一个.
        for init_pop_path in sorted(init_pop_dir.glob(f"{expected_symbol}*.json")):
            pop = Population()
            pop.import_population(init_pop_path)
            if pop.symbol != expected_symbol or len(pop) == 0:
                print(f"skip initial population {init_pop_path}: symbol mismatch")
                continue
            init_from_pop = True
            init_loaded = True
            print(f"load initial population {element}{atom} from {init_pop_path}...")
            break
        if init_from_pop:
            # 初始种群可能来自旧势, 用当前势重算团簇和候选团簇的能量与力.
            for cluster in [*pop.clusters, *pop.candidate]:
                cluster.calc = potential
                try:
                    energy_value = float(cluster.get_potential_energy())
                    force_value = np.asarray(cluster.get_forces(), dtype=float)
                finally:
                    cluster.calc = None
                if np.isfinite(energy_value):
                    cluster.info['energy'] = energy_value
                    cluster.arrays['forces'] = force_value
                else:
                    cluster.info['energy'] = float('inf')
                    cluster.arrays['forces'] = np.zeros((len(cluster), 3))
            # 重算能量后候选列表可能不再按能量升序, 重新排序以保证
            # candidate[0] 是当前最优候选.
            pop.candidate.sort(key=lambda cluster: float(cluster.info['energy']))
            # 候选数超过 candidate_size 时, 保留能量最低的 candidate_size 个.
            del pop.candidate[candidate_size:]
        else:
            pop = Population(symbol=expected_symbol)
    else:
        init_path = init_dir / f"{element}{atom}.xyz"
        if init_path.is_file():
            print(f"load structure {element}{atom}...")
            # best.xyz 是候选列表的多帧文件, 第一帧仍作为初始种群结构.
            # 其余帧全部恢复到候选列表.
            init_frames = read(init_path, index=':')
            if isinstance(init_frames, Atoms):
                init_frames = [init_frames]
            for frame in init_frames:
                # 用当前势(DP 或 DP+ZBL)重算能量和力, 不信任文件里可能来自旧势的数据.
                frame.calc = potential
                try:
                    energy_value = float(frame.get_potential_energy())
                    force_value = np.asarray(frame.get_forces(), dtype=float)
                finally:
                    frame.calc = None  # 摘掉计算器, 避免进入后续种群流程
                if np.isfinite(energy_value):
                    frame.info['energy'] = energy_value
                    frame.arrays['forces'] = force_value
                else:
                    frame.info['energy'] = float('inf')
                    frame.arrays['forces'] = np.zeros((len(frame), 3))
            pop.add_cluster(init_frames[0])
            # 先把所有帧加入候选, 再按能量升序排序; 第一帧加入种群不受排序影响.
            # 暂时不截断, 避免排序前丢失低能量结构.
            for frame in init_frames:
                pop.add_candidate(frame, candidate_size=len(init_frames))
            pop.candidate.sort(key=lambda cluster: float(cluster.info['energy']))
            # 排序后再按 candidate_size 截断.
            del pop.candidate[candidate_size:]
            init_loaded = True

    if not init_from_pop:
        remaining = pop_size - 1 if init_loaded else pop_size
        if atom <= 20:
            for i in range(remaining):
                if i < pop_size / 2:
                    random_cluster(pop, a=a, delta=0.5, relax=relax, is_2d=True)
                else:
                    random_cluster(pop, a=a, delta=0.5, relax=relax)
        else:
            for i in range(remaining):
                if i < pop_size / 4:
                    random_cluster(pop, a=a, delta=0.5, relax=relax, is_2d=True)
                else:
                    random_cluster(pop, a=a, delta=0.5, relax=relax)

    prev_best: float | None = None
    prev_pg: str | None = None
    prev_best_cluster: Atoms | None = None
    stagnation_count = 0

    for gen in range(1, generations + 1):
        tick = time.perf_counter()

        # 2) 变异 + 交叉 + 球切交叉 + 局部变换.
        pop_mut = mutate(pop, a=20 * a, debug=True)
        pop_cross = crossover(pop, pop_mut)
        pop_splice = sphere_cut_splice(pop, pop_cross, r=0.8, relax=relax, debug=True)
        pop_trans = local_transformation(pop_splice, r=0.25, relax=relax)

        # 3) 选择下一代并输出每代指标.
        pop_next = select(pop, pop_trans, relax=relax)

        energy_values = [float(e) for e in pop_next.energies if e is not None]
        if not energy_values:
            raise ValueError("Population has no available energy")
        best_cluster = pop_next.best_cluster
        # candidate 为空时直接记录本代最优结构.
        if not pop_next.candidate:
            pop_next.add_candidate(best_cluster, candidate_size=candidate_size)
        current_best = float(best_cluster.info['energy'])
        current_pg = best_cluster.info.get('PG')
        similarity_current: float | None = None
        if prev_best is None or prev_best_cluster is None:
            stagnation_count = 0
            delta = "NA"
            similarity_value = "NA"
        else:
            diff = current_best - prev_best
            delta = f"{diff:.6f}"
            similarity = similarity_function(best_cluster, prev_best_cluster)
            similarity_current = similarity
            similarity_value = f"{similarity:.6f}"
            # 点群改变或相似性 S < 0.9: 结构明显变化, 停滞计数清零.
            # 0.9 <= S < 0.95 时, 仅当最优能量变化超过 ediff 才清零.
            if (
                current_pg != prev_pg
                or similarity < similarity_reset
                or (similarity < similarity_energy_reset and abs(diff) > ediff)
            ):
                stagnation_count = 0
                # 结构变化判据满足: 本代最优结构加入候选列表.
                pop_next.add_candidate(best_cluster, candidate_size=candidate_size)
                # 输出结构变化判据: 点群变化/相似性/能量变化.
                print(
                    f"atom={atom} gen={gen:03d}"
                    f" pg={prev_pg}->{current_pg}"
                    f" similarity={similarity:.6f}"
                    f" energy_diff={diff:.6f}"
                )
            else:
                stagnation_count += 1
                # 结构变化判据不满足: 用本代最优替换候选首位,
                # 不新增候选, 避免插入后截断导致候选列表整体移位.
                if pop_next.candidate:
                    pop_next.candidate[0] = best_cluster.copy()

        # 每代写一行统计到统计 CSV.
        append_stats_row(
            symbol=f"{element}{atom}",
            atom=atom,
            gen=gen,
            pop=pop_next,
            best_energy=current_best,
            best_pg=current_pg,
            best_drop=None if prev_best is None else prev_best - current_best,
            similarity=similarity_current,
            stagnation_count=stagnation_count,
            candidate_size=candidate_size,
            elapsed_time=time.perf_counter() - (
                run_start if run_start is not None else system_start
            ),
        )
        elapsed = time.perf_counter() - tick

        print(
            f"atom={atom} gen={gen:03d} best_energy={current_best:.8f}"
            f" delta_prev={delta} similarity={similarity_value}"
            f" pg={current_pg} iter_time_s={elapsed:.4f}"
            f" best_streak={stagnation_count}"
        )

        pop = pop_next
        prev_best = current_best
        prev_pg = current_pg
        prev_best_cluster = best_cluster.copy()
        # pop.dump(f"{element}{atom}_best")
        pop.dump_xyz(f"{element}{atom}_best")
        snapshot_path = pop.export(name=f"{element}{atom}_gen{gen}")

        # 清理超出保留数量的旧快照
        if gen > keep_snapshots:
            old = gen - keep_snapshots
            for f in pop_dir.glob(f"{element}{atom}_gen{old}*.json"):
                f.unlink()

        if stagnation_count >= stagnation_limit:
            sys_elapsed = time.perf_counter() - system_start
            print(
                f"atom={atom} gen={gen:03d} converged=True"
                f" reason=stagnant_{stagnation_limit}"
                f" sys_time_s={sys_elapsed:.4f}"
            )
            break

    # 5) 导出最终结果: 候选列表写入 best/ 的 extxyz.
    pop.dump_xyz(f"{element}{atom}_best", candidate=True, dir=best_dir)
    print(f"atom={atom} final_snapshot={snapshot_path}")


def run(params_path: Path) -> None:
    """Run the optimization for every configured cluster size.

    Parameters
    ----------
    params_path : pathlib.Path
        Parameter JSON file passed to ``load_config``.
    """
    global run_start, stats_path

    program_start = time.perf_counter()
    run_start = program_start
    load_config(params_path)
    # CSV 与参数文件同名(31.json -> 31.csv); 没有可用 JSON 时用 opt_stat.csv.
    if params_path.is_file():
        stats_path = params_path.with_suffix(".csv")
    else:
        stats_path = Path("opt_stat.csv")
    # 每次运行重新生成统计文件, 避免上次运行的内容混在一起.
    if stats_path.exists():
        stats_path.unlink()
    for atom in range(atom_min, atom_max + 1):
        main(atom)
    total_elapsed = time.perf_counter() - program_start
    print(f"program total_time_s={total_elapsed:.4f}")


if __name__ == "__main__":
    run(Path("params.json"))
