from __future__ import annotations
import functools
import time
from pathlib import Path
import numpy as np
from ase.calculators.lj import LennardJones
from SaNSDE_new import *
from ase.io import read
from ase.io.xyz import write_xyz
# from ase.calculators.mixing import SumCalculator
# from deepmd.calculator import DP
# from hiphive.calculators.zbl import ZBLCalculator

# 初始化参数
element: str = 'Au'
a: float = 2.88
pop_size: int = 12
generations: int = 500
stagnation_limit: int = 50
ediff: float = 1e-5
atom_min: int = 3
atom_max: int = 5
init_dir: Path = Path('init')
keep_snapshots: int = 1
pop_dir: Path = Path('Populations')
best_dir: Path = Path('best')

# LJ 势: 用于 N=3-5 小团簇的优化测试.
potential: LennardJones = LennardJones(epsilon=0.4443, sigma=2.6367)

# DP + ZBL 混合势: DP 负责长程/成键描述, hiphive 的 ZBL 提供短程排斥, 避免
# 原子重叠. 两者能量/力由 SumCalculator 直接相加: E_total = E_DP + E_ZBL.
# 注意: hiphive 的 ZBLCalculator 内部用 ASE NeighborList, 其有效作用距离约为
#   r_eff = cutoff + 2 * skin
# 需要时取消下面注释即可恢复.
# DP_MODEL: str = "/home/linyj/02-linyj/potentials/dp/Au3c.pth"
# ZBL_CUTOFF: float = 1.5
# ZBL_SKIN: float = 0
# dp: DP = DP(model=DP_MODEL)
# zbl: ZBLCalculator = ZBLCalculator(cutoff=ZBL_CUTOFF, skin=ZBL_SKIN)
# potential: SumCalculator = SumCalculator([dp, zbl])
relax = functools.partial(
    local_optimize,
    cell=None,
    # 这里替换为你需要的势函数的ase计算器对象, 如potential=dp
    # potential=LennardJones(epsilon=0.4443, sigma=2.6367),
    potential = potential,
    optimizer=BFGS,
    fmax=0.05,
    steps=300,
)

def main(atom: int) -> None:
    system_start = time.perf_counter()
    # 1) 初始化种群: 先从 init/ 目录尝试加载已有结构, 再随机补足至 pop_size.
    pop = Population(symbol=f"{element}{atom}")
    init_path = init_dir / f"{element}{atom}_best.xyz"
    init_loaded = False
    if init_path.is_file():
        print(f"load structure {element}{atom}...")
        cluster = read(init_path)
        # 用当前势(DP+ZBL)做一次单点计算得到能量, 不信任文件里可能来自旧势的能量.
        cluster.calc = potential
        energy_value = float(cluster.get_potential_energy())
        cluster.calc = None  # 摘掉计算器: add_cluster 会 deepcopy, DP 不可深拷贝
        if not np.isfinite(energy_value):
            energy_value = float('inf')
        cluster.energy = energy_value
        pop.add_cluster(cluster)
        init_loaded = True

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
    stagnation_count: int = 0

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
        current_best = min(energy_values)
        if prev_best is None:
            stagnation_count = 0
            delta = "NA"
        else:
            diff = current_best - prev_best
            delta = f"{diff:.6f}"
            if abs(diff) <= ediff:
                stagnation_count += 1
            else:
                stagnation_count = 0
        elapsed = time.perf_counter() - tick

        print(
            f"atom={atom} gen={gen:03d} best_energy={current_best:.8f}"
            f" delta_prev={delta} iter_time_s={elapsed:.4f}"
            f" best_streak={stagnation_count}"
        )

        pop = pop_next
        prev_best = current_best
        pop.dump(f"{element}{atom}_best")
        pop.dump_xyz(f"{element}{atom}_best")
        pop.export(name=f"{element}{atom}_gen{gen}")

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

    # 5) 导出最终快照.
    snapshot_path = pop.export(name=f"{element}{atom}_end", xyz=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / f"{element}{atom}_best.xyz"
    best_energy = float(pop.best_cluster.energy)
    with open(best_path, 'w', encoding='utf-8') as f:
        write_xyz(f, [pop.best_cluster], comment=str(best_energy))
    print(f"atom={atom} final_snapshot={snapshot_path} best_xyz={best_path}")


if __name__ == "__main__":
    program_start = time.perf_counter()
    for atom in range(atom_min, atom_max + 1):
        main(atom)
    total_elapsed = time.perf_counter() - program_start
    print(f"program total_time_s={total_elapsed:.4f}")
