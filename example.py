from __future__ import annotations
import functools
import time
import numpy as np
from ase.calculators.lj import LennardJones
from SaNSDE_new import *
# from deepmd.calculator import DP

# 初始化参数
element: str = 'Au'
a: float = 2.88
popsize: int = 12
generations: int = 10
stagnationLimit: int = 20
atomMin: int = 3
atomMax: int = 50
# dp = DP(model="Au-Ce-C-O.pb")
relax = functools.partial(
    localOptimize,
    cell=[15, 15, 15],
    # 这里替换为你需要的势函数的ase计算器对象, 如potential=dp
    potential=LennardJones(epsilon=0.4443, sigma=2.6367),
    optimizer=BFGS,
    fmax=0.05,
    steps=100,
)

def main(atom: int) -> None:
    # 1) 初始化种群: 先固定化学式, 再随机生成并添加 popsize 个 {element}{atom} 团簇.
    pop = Population(symbol=f"{element}{atom}")
    if atom <= 20:
        for i in range(popsize):
            if i < popsize / 2:
                randomCluster(pop, a=a, delta=0.5, relax=relax, is2D=True)
            else:
                randomCluster(pop, a=a, delta=0.5, relax=relax)
    else:
        for i in range(popsize):
            if i < popsize / 4:
                randomCluster(pop, a=a, delta=0.5, relax=relax, is2D=True)
            else:
                randomCluster(pop, a=a, delta=0.5, relax=relax)

    prevBest: float | None = None
    stagnationCount: int = 0

    for gen in range(1, generations + 1):
        tick = time.perf_counter()

        # 2) 变异 + 交叉 + 球切交叉 + 局部变换.
        popMut = mutate(pop, a=10 * a, debug=True)
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
            delta = f"{diff:.8f}"
            if abs(diff) <= 1e-12:
                stagnationCount += 1
            else:
                stagnationCount = 0
        elapsed = time.perf_counter() - tick

        print(
            f"atom={atom} gen={gen:03d} best_energy={currentBest:.8f}"
            f"delta_prev={delta} iter_time_s={elapsed:.4f}"
            f" best_streak={stagnationCount}"
        )

        pop = popNext
        prevBest = currentBest
        pop.dump(f"{element}{atom}_best")
        pop.export(name=f"{element}{atom}_gen{gen}")

        if stagnationCount >= stagnationLimit:
            print(
                f"atom={atom} gen={gen:03d} converged=True"
                f" reason=stagnant_{stagnationLimit}"
            )
            break

    # 5) 导出最终快照.
    snapshotPath = pop.export(name=f"{element}{atom}_end", xyz=True)
    print(f"atom={atom} final_snapshot={snapshotPath}")


if __name__ == "__main__":
    for atom in range(atomMin, atomMax + 1):
        main(atom)
