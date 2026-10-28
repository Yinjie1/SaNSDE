#!/usr/bin/env python3
from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.io.xyz import write_xyz
from ase.optimize import BFGS
from deepmd.calculator import DP

try:
    from hiphive.calculators.zbl import ZBLCalculator
except ImportError:
    ZBLCalculator = None  # type: ignore[assignment]

# from ase.calculators.lj import LennardJones
from operators import (
    crossover,
    local_optimize,
    local_transformation,
    mutate,
    random_cluster,
    select,
    sphere_cut_splice,
)
from pop import Population

# 初始化参数: 默认值内联, 可由运行目录下的 params.json 覆盖, 缺失的键保持默认值.
param_file = Path('params.json')
try:
    with param_file.open('r', encoding='utf-8') as f:
        params: dict[str, Any] = json.load(f)
    if not isinstance(params, dict):
        raise TypeError(f"{param_file} must contain a JSON object")
    print(f"load parameters from {param_file}")
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


def similarity_function(atoms_a: Atoms, atoms_b: Atoms) -> float:
    """计算两个团簇的相似性函数 S = 1 / (1 + q).

    自动从 ``Atoms`` 对象读取坐标并构造升序原子间距列表, 不依赖预先存储的
    距离列表. 公式见 docs/similarity_function.md.

    Parameters
    ----------
    atoms_a : Atoms
        第一个团簇.
    atoms_b : Atoms
        第二个团簇.

    Returns
    -------
    float
        相似性函数 S = 1 / (1 + q), 取值 (0, 1].
        原子数不一致, 原子数少于 2, 坐标含非有限值, 或最短间距为 0 时
        返回 0.0.
    """
    if len(atoms_a) != len(atoms_b) or len(atoms_a) < 2:
        return 0.0

    positions_a = np.asarray(atoms_a.get_positions(), dtype=float)
    positions_b = np.asarray(atoms_b.get_positions(), dtype=float)
    if positions_a.shape != positions_b.shape:
        return 0.0
    if not np.isfinite(positions_a).all() or not np.isfinite(positions_b).all():
        return 0.0

    pair_vectors_a = positions_a[:, None, :] - positions_a[None, :, :]
    pair_vectors_b = positions_b[:, None, :] - positions_b[None, :, :]
    upper_indices = np.triu_indices(len(positions_a), k=1)
    distances_a = np.sort(np.linalg.norm(pair_vectors_a, axis=-1)[upper_indices])
    distances_b = np.sort(np.linalg.norm(pair_vectors_b, axis=-1)[upper_indices])

    if distances_a.size == 0:
        return 0.0
    if distances_a[0] <= 0.0 or distances_b[0] <= 0.0:
        return 0.0

    normalized_a = distances_a / distances_a[0]
    normalized_b = distances_b / distances_b[0]
    q_value = float(np.sqrt(np.mean((normalized_a - normalized_b) ** 2)))
    return 1.0 / (1.0 + q_value)


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
        else:
            pop = Population(symbol=expected_symbol)
    else:
        init_path = init_dir / f"{element}{atom}.xyz"
        if init_path.is_file():
            print(f"load structure {element}{atom}...")
            cluster = read(init_path)
            # 用当前势(DP 或 DP+ZBL)重算能量和力, 不信任文件里可能来自旧势的数据.
            cluster.calc = potential
            try:
                energy_value = float(cluster.get_potential_energy())
                force_value = np.asarray(cluster.get_forces(), dtype=float)
            finally:
                cluster.calc = None  # 摘掉计算器: add_cluster 会 deepcopy, DP 不可深拷贝
            if np.isfinite(energy_value):
                cluster.info['energy'] = energy_value
                cluster.arrays['forces'] = force_value
            else:
                cluster.info['energy'] = float('inf')
                cluster.arrays['forces'] = np.zeros((len(cluster), 3))
            pop.add_cluster(cluster)
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
        if prev_best is None or prev_best_cluster is None:
            stagnation_count = 0
            delta = "NA"
            similarity_value = "NA"
        else:
            diff = current_best - prev_best
            delta = f"{diff:.6f}"
            similarity = similarity_function(best_cluster, prev_best_cluster)
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
        pop.dump(f"{element}{atom}_best")
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

    # 5) 导出最终结果.
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / f"{element}{atom}_best.xyz"
    best_energy = float(pop.best_cluster.info['energy'])
    with best_path.open('w', encoding='utf-8') as f:
        write_xyz(f, [pop.best_cluster], comment=str(best_energy))
    print(f"atom={atom} final_snapshot={snapshot_path}")


if __name__ == "__main__":
    program_start = time.perf_counter()
    for atom in range(atom_min, atom_max + 1):
        main(atom)
    total_elapsed = time.perf_counter() - program_start
    print(f"program total_time_s={total_elapsed:.4f}")
