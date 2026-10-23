from __future__ import annotations
import functools
import time
from pathlib import Path
import numpy as np
# from ase.calculators.lj import LennardJones
from SaNSDE_new import *
from ase.io import read
from ase.io.xyz import write_xyz
from ase.calculators.mixing import SumCalculator
from deepmd.calculator import DP
from hiphive.calculators.zbl import ZBLCalculator

# 初始化参数
element: str = 'Au'
a: float = 2.88
popsize: int = 12
generations: int = 500
stagnationLimit: int = 50
ediff: float = 1e-5
atomMin: int = 3
atomMax: int = 30
initDir: Path = Path('init')
keepSnapshots: int = 1
popDir: Path = Path('Populations')
bestDir: Path = Path('best')

# DP + ZBL 混合势: DP 负责长程/成键描述, hiphive 的 ZBL 提供短程排斥, 避免
# 原子重叠. 两者能量/力由 SumCalculator 直接相加: E_total = E_DP + E_ZBL.
#
# 注意: hiphive 的 ZBLCalculator 内部用 ASE NeighborList, 其有效作用距离约为
#   r_eff = cutoff + 2 * skin
# (skin 为邻居表缓冲). ZBL 屏蔽函数本身衰减较慢, 取值越大对正常键长区间
# 干扰越大, 可按需要调整下面的 cutoff/skin.
DP_MODEL: str = "/home/linyj/02-linyj/potentials/dp/Au3c.pth"
ZBL_CUTOFF: float = 1.5
ZBL_SKIN: float = 0

dp: DP = DP(model=DP_MODEL)
zbl: ZBLCalculator = ZBLCalculator(cutoff=ZBL_CUTOFF, skin=ZBL_SKIN)
potential: SumCalculator = SumCalculator([dp, zbl])
relax = functools.partial(
    localOptimize,
    cell=None,
    # 这里替换为你需要的势函数的ase计算器对象, 如potential=dp
    # potential=LennardJones(epsilon=0.4443, sigma=2.6367),
    potential = potential,
    optimizer=BFGS,
    fmax=0.05,
    steps=300,
)

def main(atom: int) -> None:
    systemStart = time.perf_counter()
    # 1) 初始化种群: 先从 init/ 目录尝试加载已有结构, 再随机补足至 popsize.
    pop = Population(symbol=f"{element}{atom}")
    initPath = initDir / f"{element}{atom}_best.xyz"
    initLoaded = False
    if initPath.is_file():
        print(f"load structure {element}{atom}...")
        cluster = read(initPath)
        # 用当前势(DP+ZBL)做一次单点计算得到能量, 不信任文件里可能来自旧势的能量.
        cluster.calc = potential
        energy_val = float(cluster.get_potential_energy())
        cluster.calc = None  # 摘掉计算器: addCluster 会 deepcopy, DP 不可深拷贝
        if not np.isfinite(energy_val):
            energy_val = float('inf')
        cluster.energy = energy_val
        pop.addCluster(cluster)
        initLoaded = True

    remaining = popsize - 1 if initLoaded else popsize
    if atom <= 20:
        for i in range(remaining):
            if i < popsize / 2:
                randomCluster(pop, a=a, delta=0.5, relax=relax, is2D=True)
            else:
                randomCluster(pop, a=a, delta=0.5, relax=relax)
    else:
        for i in range(remaining):
            if i < popsize / 4:
                randomCluster(pop, a=a, delta=0.5, relax=relax, is2D=True)
            else:
                randomCluster(pop, a=a, delta=0.5, relax=relax)

    prevBest: float | None = None
    stagnationCount: int = 0

    for gen in range(1, generations + 1):
        tick = time.perf_counter()

        # 2) 变异 + 交叉 + 球切交叉 + 局部变换.
        popMut = mutate(pop, a=20 * a, debug=True)
        popCross = crossover(pop, popMut)
        popSplice = sphereCutSplice(pop, popCross, r=0.8, relax=relax, debug=True)
        popTrans = localTransformation(popSplice, r=0.25, relax=relax)

        # 3) 选择下一代并输出每代指标.
        popNext = select(pop, popTrans, relax=relax)

        energyValues = [float(e) for e in popNext.energies if e is not None]
        if not energyValues:
            raise ValueError("Population has no available energy")
        currentBest = min(energyValues)
        if prevBest is None:
            stagnationCount = 0
            delta = "NA"
        else:
            diff = currentBest - prevBest
            delta = f"{diff:.6f}"
            if abs(diff) <= ediff:
                stagnationCount += 1
            else:
                stagnationCount = 0
        elapsed = time.perf_counter() - tick

        print(
            f"atom={atom} gen={gen:03d} best_energy={currentBest:.8f}"
            f" delta_prev={delta} iter_time_s={elapsed:.4f}"
            f" best_streak={stagnationCount}"
        )

        pop = popNext
        prevBest = currentBest
        pop.dump(f"{element}{atom}_best")
        pop.dumpxyz(f"{element}{atom}_best")
        pop.export(name=f"{element}{atom}_gen{gen}")

        # 清理超出保留数量的旧快照
        if gen > keepSnapshots:
            old = gen - keepSnapshots
            for f in popDir.glob(f"{element}{atom}_gen{old}*.json"):
                f.unlink()

        if stagnationCount >= stagnationLimit:
            sysElapsed = time.perf_counter() - systemStart
            print(
                f"atom={atom} gen={gen:03d} converged=True"
                f" reason=stagnant_{stagnationLimit}"
                f" sys_time_s={sysElapsed:.4f}"
            )
            break

    # 5) 导出最终快照.
    snapshotPath = pop.export(name=f"{element}{atom}_end", xyz=True)
    bestDir.mkdir(parents=True, exist_ok=True)
    bestPath = bestDir / f"{element}{atom}_best.xyz"
    bestEnergy = float(pop.bestCluster.energy)
    with open(bestPath, 'w', encoding='utf-8') as f:
        write_xyz(f, [pop.bestCluster], comment=str(bestEnergy))
    print(f"atom={atom} final_snapshot={snapshotPath} best_xyz={bestPath}")


if __name__ == "__main__":
    programStart = time.perf_counter()
    for atom in range(atomMin, atomMax + 1):
        main(atom)
    totalElapsed = time.perf_counter() - programStart
    print(f"program total_time_s={totalElapsed:.4f}")
