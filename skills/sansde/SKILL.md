---
name: sansde
description: ...
license: MIT
---
## SaNSDE

SaNSDE 是一个对纳米团簇进行结构搜索和全局优化的程序, 使用自适应邻域差分进化算法对团簇势能面(PES)进行高效的搜索, 能够快速地找到全局最优结构(Global minimum).

## 如何使用

这是一个 CLI 黑箱工具, 你无需对程序内部实现有任何了解, 只需知道我们要优化什么体系(元素, 原子数), 使用什么势能面(LJ, DP等), 机器信息(CPU/GPU, 可用核数, 环境等). 你只需按部就班地执行以下操作.

可用执行 init 子命令在当前目录按默认参数生成输入 json 文件:
```bash
sansde init
```

以下是一个用 DP 势优化的输入文件示例(这里用jsonc是为了方便说明参数含义, 请严格按照json格式), 为列出的参数如果用户没有要求, 那么你无需关心它的作用
```jsonc
{
    "SCRIPTS": ["from deepmd.calculator import DP"],  // 程序开始运行时按顺序执行其中的python语句, 可用于导入ASE的计算器
    "ELEMENT": "Au",
    "LATTICE_CONST": 3, // 一般设对应bulk的晶格参数
    "MIN_ATOM": 2,  // 从该尺寸开始优化
    "MAX_ATOM": 18, // 优化到该尺寸结束(含)
    "INIT_DIR": "init",
    "INIT_POP_DIR": "",
    "CONTINUE": false,
    "MAX_GEN": 500, // 优化步数上限
    "CONV_STEPS": 250,  // 若结构连续 CONV_STEPS 代不变, 则视为优化收敛
    "EDIFF": 1e-05, // 结构弛豫收敛判据, 精度不要高于势函数的精度
    "CAND_SIZE": 10,  // 每种尺寸候选个体上限, 决定了保留多少个次优解, 设为1则只保留最优解
    "POTENTIAL": "DP(model=Path('/home/linyj/02-linyj/potentials/dp/Au3c.pth'))", // ASE 计算器配置
}
```

配置完输入文件和初始结构/种群后, 运行以下命令进行检查:
```bash
sansde check
```
初始结构/种群缺失警告可以忽视, 因为这不是破坏性的

一切准备就绪后, 运行以下命令正式开始优化(可能需要很长时间):
```bash
sansde run <input>.json
```
