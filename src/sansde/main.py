from __future__ import annotations

import functools
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.calculators.mixing import SumCalculator
from ase.optimize import BFGS

from sansde.calc import ZBLCalculator
from sansde.io import append_stats_row, read_run_params
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
from sansde.utils import print_banner


def run(
    atom: int,
    round_index: int = 1,
    *,
    params: dict[str, Any],
    potential: BaseCalculator,
    relax: Callable[..., Any],
    stats_path: Path,
    run_start: float,
) -> None:
    """Optimize one cluster size.

    Parameters
    ----------
    atom : int
        Number of ``ELEMENT`` atoms in the cluster. When ``ELEMENT2`` is set,
        the cluster also holds ``TOTAL_ATOM - atom`` atoms of ``ELEMENT2``.
    round_index : int, optional
        Current optimization round, used to derive the round directories.
        Default 1.
    params : dict[str, Any]
        Parsed run parameters.
    potential : BaseCalculator
        ASE calculator used to re-evaluate imported structures.
    relax : Callable
        Local optimization callable.
    stats_path : Path
        Path of the main statistics CSV.
    run_start : float
        Program start time from ``time.perf_counter``.
    """
    system_start = time.perf_counter()

    if int(params['ROUND']) > 1:
        init_dir = (
            Path('init') if round_index == 1 else Path(f'best{round_index - 1}')
        )
        best_dir = Path(f'best{round_index}')
        init_pop_dir = Path('')
    else:
        init_dir = Path(params['INIT_DIR'])
        best_dir = Path(params['BEST_DIR'])
        init_pop_dir = Path(params['INIT_POP_DIR'])

    # 1) 初始化种群: init_pop_dir 非空时按尺寸加载给定种群;
    #    否则从 init/ 目录加载单个最优结构, 再随机补足至 pop_size.
    element = str(params['ELEMENT'])
    element2 = str(params['ELEMENT2'] or '')
    if element2:
        second_count = int(params['TOTAL_ATOM']) - atom
        expected_symbol = Atoms(
            f"{element}{atom}{element2}{second_count}"
        ).get_chemical_formula()
    else:
        expected_symbol = f"{element}{atom}"
    pop = Population(symbol=expected_symbol)
    init_from_pop = False

    # Path("") 会被规范化为 '.', 以此判断 init_pop_dir 是否为空.
    if init_pop_dir != Path('.'):
        # 文件名支持 {element}{atom}.json、{element}{atom}_gen3.json 等;
        # 逐个尝试, 用化学式校验, 不匹配则继续下一个.
        for init_pop_path in sorted(init_pop_dir.glob(f"{expected_symbol}*.json")):
            pop = Population()
            pop.load(init_pop_path, optimal_only=not bool(params['CONTINUE']))
            if pop.symbol != expected_symbol or len(pop) == 0:
                print(f"skip initial population {init_pop_path}: symbol mismatch")
                continue
            init_from_pop = True
            print(f"load initial population {expected_symbol} from {init_pop_path}...")
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
            # Re-sort after recomputing energies: decreasing order (worst
            # first) keeps candidate[-1] as the current best.
            pop.candidate.sort(
                key=lambda cluster: float(cluster.info['energy']),
                reverse=True,
            )
            # Keep the candidate_size lowest-energy entries at the tail.
            del pop.candidate[:max(len(pop.candidate) - int(params['CAND_SIZE']), 0)]
        else:
            pop = Population(symbol=expected_symbol)
    else:
        # An empty INIT_DIR means no initial structure is read.
        if params['INIT_DIR']:
            for init_path in sorted(init_dir.glob(f"{expected_symbol}*.xyz")):
                try:
                    pop.load_xyz(init_path)
                except ValueError as e:
                    print(f"skip structure {init_path}: {e}")
                    continue
                if len(pop) == 0:
                    continue
                print(f"load structure {expected_symbol} from {init_path}...")
                # Recompute energies and forces for the imported population and
                # candidates with the current potential.
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
                pop.candidate.sort(
                    key=lambda cluster: float(cluster.info['energy']),
                    reverse=True,
                )
                # Keep the candidate_size lowest-energy entries at the tail.
                del pop.candidate[:max(len(pop.candidate) - int(params['CAND_SIZE']), 0)]
                break

    if len(pop) < int(params['POP_SIZE']):
        remaining = int(params['POP_SIZE']) - len(pop)
        if atom <= 20:
            for i in range(remaining):
                if i < int(params['POP_SIZE']) / 2:
                    random_cluster(pop, a=float(params['LATTICE_CONST']), delta=0.5, relax=relax, is_2d=True)
                else:
                    random_cluster(pop, a=float(params['LATTICE_CONST']), delta=0.5, relax=relax)
        else:
            for i in range(remaining):
                if i < int(params['POP_SIZE']) / 4:
                    random_cluster(pop, a=float(params['LATTICE_CONST']), delta=0.5, relax=relax, is_2d=True)
                else:
                    random_cluster(pop, a=float(params['LATTICE_CONST']), delta=0.5, relax=relax)

    prev_best: float | None = None
    prev_pg: str | None = None
    prev_best_cluster: Atoms | None = None
    stagnation_count = 0

    for gen in range(1, int(params['MAX_GEN']) + 1):
        tick = time.perf_counter()

        # Resample the per-individual CR every CR_PERIOD generations.
        if int(params['CR_PERIOD']) > 0 and (gen - 1) % int(params['CR_PERIOD']) == 0:
            for cluster in pop:
                cluster.info['cr'] = float(
                    np.clip(np.random.normal(pop.crm, 0.1), 0.0, 1.0)
                )

        # 2) 变异 + 交叉 + 球切交叉 + 局部变换.
        pop_mut = mutate(pop, a=20 * float(params['LATTICE_CONST']), debug=True)
        pop_cross = crossover(pop, pop_mut)
        pop_splice = sphere_cut_splice(pop, pop_cross, relax=relax)
        pop_trans = local_transformation(pop_splice, r=0.25, relax=relax)

        # 3) 选择下一代并输出每代指标.
        pop_next = select(
            pop,
            pop_trans,
            relax=relax,
            update_crm=int(params['CRM_PERIOD']) > 0 and gen % int(params['CRM_PERIOD']) == 0,
            ediff=float(params['EDIFF']),
        )
        # p and fp are updated only when the learning period ends.
        if int(params['LEARNING_PERIOD']) > 0 and gen % int(params['LEARNING_PERIOD']) == 0:
            pop_next.update_adaptive(adapt_fp=bool(params['FP_ADAPTIVE']))

        energy_values = [float(e) for e in pop_next.energies if e is not None]
        if not energy_values:
            raise ValueError("Population has no available energy")
        best_cluster = pop_next.best_cluster
        # candidate 为空时直接记录本代最优结构.
        if not pop_next.candidate:
            pop_next.add_candidate(best_cluster, candidate_size=int(params['CAND_SIZE']))
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
            #  若结构变化判据满足, 停滞计数清零.
            if (
                current_pg != prev_pg
                or similarity < float(params['S_STRICT'])
                or (similarity < float(params['S_LOOSE']) and abs(diff) > 10 * float(params['EDIFF']))
            ):
                stagnation_count = 0
                # 结构变化判据满足: 本代最优结构加入候选列表.
                pop_next.add_candidate(best_cluster, candidate_size=int(params['CAND_SIZE']))
                # The criterion only fires after the best cluster has changed,
                # so imported initial candidates are never recorded here.
                # Frames are appended so the whole optimization history is kept.
                pop_next.dump_xyz(
                    name=f"{expected_symbol}_better",
                    dir=params['BETTER_DIR'],
                )
                # 输出结构变化判据: 点群变化/相似性/能量变化.
                print(
                    f"atom={atom} gen={gen:03d}"
                    f" pg={prev_pg}->{current_pg}"
                    f" similarity={similarity:.6f}"
                    f" energy_diff={diff:.6f}"
                )
            else:
                stagnation_count += 1
                # 结构变化判据不满足: 仅当本代最优能量更低时替换候选末位(最优),
                # 不新增候选, 避免追加后裁减导致候选列表整体移位.
                if pop_next.candidate:
                    candidate_energy = pop_next.candidate[-1].info.get('energy')
                    if candidate_energy is not None and current_best < float(candidate_energy):
                        pop_next.candidate[-1] = best_cluster.copy()

        # 每代写一行统计到统计 CSV.
        append_stats_row(
            stats_path=stats_path,
            symbol=expected_symbol,
            atom=atom,
            gen=gen,
            pop=pop_next,
            best_energy=current_best,
            best_pg=current_pg,
            best_drop=None if prev_best is None else prev_best - current_best,
            similarity=similarity_current,
            stagnation_count=stagnation_count,
            candidate_size=int(params['CAND_SIZE']),
            elapsed_time=time.perf_counter() - run_start,
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
        pop.dump_xyz(f"{expected_symbol}_best")
        snapshot_path = pop.export(
            name=f"{expected_symbol}_gen{gen}",
            dir=params['POP_DIR'],
        )

        # 清理超出保留数量的旧快照
        if gen > int(params['KEEP_SNAPSHOTS']):
            old = gen - int(params['KEEP_SNAPSHOTS'])
            for f in Path(params['POP_DIR']).glob(f"{expected_symbol}_gen{old}*.json"):
                f.unlink()

        # CONV_STEPS=0 means no early stopping: run until MAX_GEN.
        if int(params['CONV_STEPS']) > 0 and stagnation_count >= int(params['CONV_STEPS']):
            sys_elapsed = time.perf_counter() - system_start
            print(
                f"atom={atom} gen={gen:03d} converged=True"
                f" reason=stagnant_{params['CONV_STEPS']}"
                f" sys_time_s={sys_elapsed:.4f}"
            )
            break

    # 5) 导出最终结果: 候选列表写入 best/ 的 extxyz.
    pop.dump_xyz(f"{expected_symbol}_best", candidate=True, dir=best_dir)
    print(f"atom={atom} final_snapshot={snapshot_path}")


def main(params_path: Path) -> None:
    """Run the optimization for every configured cluster size.

    Parameters
    ----------
    params_path : pathlib.Path
        Parameter JSON file.
    """
    print_banner()
    program_start = time.perf_counter()

    params = read_run_params(params_path)

    # Optional environment variables applied to the current process before
    # SCRIPTS, for example {"OMP_NUM_THREADS": "6"}.
    EXPORT = params['EXPORT']
    if not isinstance(EXPORT, dict):
        raise TypeError("EXPORT must be a dictionary of environment variables")
    for name, value in EXPORT.items():
        os.environ[str(name)] = str(value)

    # Optional Python snippets executed in order before any other parameter is
    # used. They can import calculators or modify ``params``, for example:
    # ["from ase.calculators.lj import LennardJones", "from deepmd.calculator import DP"]
    SCRIPTS = params['SCRIPTS']
    if not isinstance(SCRIPTS, list) or not all(
        isinstance(script, str) for script in SCRIPTS
    ):
        raise TypeError("SCRIPTS must be a list of Python code strings")
    for script in SCRIPTS:
        exec(script, globals())  # noqa: S102, user-provided calculator setup

    # Binary alloy runs: ELEMENT2 non-empty activates them. Each size then
    # holds `atom` atoms of ELEMENT and TOTAL_ATOM - atom atoms of ELEMENT2.
    if str(params['ELEMENT2'] or ''):
        if str(params['ELEMENT2']) == str(params['ELEMENT']):
            raise ValueError("ELEMENT2 must differ from ELEMENT")
        if int(params['TOTAL_ATOM']) <= int(params['MAX_ATOM']):
            raise ValueError(
                "TOTAL_ATOM must be larger than MAX_ATOM when ELEMENT2 is set"
            )

    # Potential energy surface.
    # With USE_ZBL the built-in ZBL short-range repulsion is added on top of
    # POTENTIAL: E_total = E_POTENTIAL + E_ZBL.
    potential_spec = params['POTENTIAL']
    if isinstance(potential_spec, str):
        potential_spec = eval(potential_spec, globals())
    if not isinstance(potential_spec, BaseCalculator):
        raise TypeError(
            "POTENTIAL must be an ASE calculator or an expression that "
            f"evaluates to one, got {type(potential_spec).__name__}"
        )
    POTENTIAL: BaseCalculator = potential_spec

    if bool(params['USE_ZBL']):
        zbl = ZBLCalculator(
            cutoff=float(params['ZBL_CUTOFF']),
            smooth=bool(params['ZBL_SMOOTH']),
        )
        potential = SumCalculator([POTENTIAL, zbl])
    else:
        potential = POTENTIAL

    print("initialized parameters:")
    print(json.dumps(params, indent=2, ensure_ascii=False))

    relax = functools.partial(
        local_optimize,
        cell=None,
        potential=potential,  # type: ignore
        optimizer=BFGS,
        fmax=float(params['RELAX_FMAX']),
        steps=int(params['RELAX_STEPS']),
    )

    # CSV 与参数文件同名(31.json -> 31.csv); 没有可用 JSON 时用 opt_stat.csv.
    if params_path.is_file():
        stats_path = params_path.with_suffix(".csv")
    else:
        stats_path = Path("opt_stat.csv")
    # 每次运行重新生成统计文件, 避免上次运行的内容混在一起.
    if stats_path.exists():
        stats_path.unlink()
    pop_stats_path = stats_path.with_stem(f"{stats_path.stem}_pop")
    if pop_stats_path.exists():
        pop_stats_path.unlink()
    rounds = int(params['ROUND'])
    for round_index in range(1, rounds + 1):
        if rounds > 1:
            # 第 1 轮从 init/ 读取, 之后各轮读取上一轮的 best{round-1}/.
            init_dir = (
                Path('init')
                if round_index == 1
                else Path(f'best{round_index - 1}')
            )
            best_dir = Path(f'best{round_index}')
            print(
                f"round {round_index}/{rounds}: "
                f"init_dir={init_dir} best_dir={best_dir}"
            )
        for atom in range(int(params['MIN_ATOM']), int(params['MAX_ATOM']) + 1):
            run(
                atom,
                round_index,
                params=params,
                potential=potential, # type: ignore
                relax=relax,
                stats_path=stats_path,
                run_start=program_start,
            )
    total_elapsed = time.perf_counter() - program_start
    print(f"program total_time_s={total_elapsed:.4f}")

if __name__ == "__main__":
    main(Path("params.json"))
