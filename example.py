from __future__ import annotations

import time
import numpy as np
from SaNSDE_new import *
# from deepmd.calculator import DP

# 参数设置
atom = 13
a=2.88
popsize = 12
generations = 10
# dp = DP(model="Au-Ce-C-O.pb")
relax = functools.partial(
    localOptimize,
    cell=[15, 15, 15],
    # 这里替换为你需要的势函数的ase计算器对象, 如potential=dp
    potential=LennardJones(),
    optimizer=BFGS,
    fmax=0.05,
    steps=100,
)

# 1) 初始化种群: 先固定化学式, 再随机生成并添加 popsize 个 Au{atom} 团簇.
pop = Population(symbol=f"Au{atom}")
for _ in range(popsize):
    randomCluster(pop, a=a, delta=0.5, relax=relax)

prevBest: float | None = None

for gen in range(1, generations + 1):
    tick = time.perf_counter()

    # 2) 变异 + 交叉 + 球切交叉 + 局部变换.
    popMut = mutate(pop, a=a, debug=True)
    popCross = crossover(pop, popMut)
    popSplice = sphereCutSplice(pop, popCross, r=0.8, relax=relax, debug=True)
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
    pop.dump(f"Au{atom}_best")
    pop.export(name=f"Au{atom}_gen{gen}")

# 5) 导出最终快照.
snapshotPath = pop.export(name=f"Au{atom}_end", xyz=True)
print(f"final_snapshot={snapshotPath}")