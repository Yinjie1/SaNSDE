from __future__ import annotations

import time
import numpy as np
from SaNSDE_new import *


def main() -> None:
    # 1) 初始化种群: 先固定化学式, 再随机生成并添加 12 个 Au13 团簇.
    pop = Population(symbol="Au13")
    for _ in range(12):
        randomCluster(
            pop,
            a=2.88,
            delta=0.5,
            relax=relax,
        )

    prevBest: float | None = None
    generations = 10

    for gen in range(1, generations + 1):
        tick = time.perf_counter()

        # 2) 变异 + 交叉 + 球切交叉 + 局部变换.
        popMut = mutate(pop)
        popCross = crossover(pop, popMut)
        popSplice = SphereCutSplice(pop, popCross, r=0.8, relax=relax)
        popTrans = localTransformation(popSplice, r=0.25, relax=relax)

        # 3) 选择下一代并输出每代指标.
        popNext = select(pop, popTrans, relax=relax)

        energyValues = [float(e) for e in popNext.energies if e is not None]
        if not energyValues:
            raise ValueError("Population has no available energy")
        currentBest = min(energyValues)
        delta = "NA" if prevBest is None else f"{currentBest - prevBest:.8f}"
        elapsed = time.perf_counter() - tick

        print(
            f"gen={gen:03d} best_energy={currentBest:.8f} "
            f"delta_prev={delta} iter_time_s={elapsed:.4f}"
        )

        pop = popNext
        prevBest = currentBest

    # 5) 导出最终快照.
    snapshotPath = pop.export(name="Au13_workflow_example_gen10", xyz=True)
    print(f"final_snapshot={snapshotPath}")


if __name__ == "__main__":
    main()
