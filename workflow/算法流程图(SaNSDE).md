# 算法流程图 (SaNSDE)

该文档描述了 SaNSDE 算法的执行流程。

## Mermaid 流程图

```mermaid
graph TD
    %% 开始节点
    Start(开始) --> InitNode
    
    %% 初始化阶段
    subgraph Initialization [初始化]
        direction TB
        InitNode[所有原子在一个指定半径的球体内<br>随机生成]
    end
    
    InitNode --> MutNode1

    %% 变异操作阶段
    subgraph Mutation [变异操作]
        direction TB
        MutNode1[随机选择三个不同团簇和当前最优团簇]
        MutNode2[通过变异策略生成暂定团簇]
        MutNode1 --> MutNode2
    end
    
    MutNode2 --> CrossNode1

    %% 交叉操作阶段
    subgraph Crossover [交叉操作]
        direction TB
        CrossNode1[与上述暂定团簇交换原子坐标<br>更新团簇原子坐标]
        CrossNode2[根据球形区域与当前最优团簇<br>交换原子坐标]
        CrossNode1 --> CrossNode2
    end

    CrossNode2 --> LocalNode1

    %% 局部变换阶段
    subgraph LocalTransformation [局部变换]
        direction TB
        LocalNode1[随机移动原子某一维度上的坐标]
        LocalNode2[使用梯度下降法进行局部优化]
        LocalNode1 --> LocalNode2
    end

    LocalNode2 --> UpdateNode

    %% 循环控制与结束
    UpdateNode[更新当前最优团簇和<br>SaNSDE的参数] --> Decision{是否满足迭代<br>终止条件}
    
    Decision -- 否 --> MutNode1
    Decision -- 是 --> OutputNode[输出]
    
    OutputNode --> End(结束)
```

## 文本描述

1.  **开始**
2.  **初始化**:
    *   所有原子在一个指定半径的球体内随机生成。
3.  **进入主循环**:
    *   **变异操作**:
        *   随机选择三个不同团簇和当前最优团簇。
        *   通过变异策略生成暂定团簇。
    *   **交叉操作**:
        *   与上述暂定团簇交换原子坐标，更新团簇原子坐标。
        *   根据球形区域与当前最优团簇交换原子坐标。
    *   **局部变换**:
        *   随机移动原子某一维度上的坐标。
        *   使用梯度下降法进行局部优化。
    *   **更新状态**:
        *   更新当前最优团簇和 SaNSDE 的参数。
    *   **判断终止条件**:
        *   若**未满足**迭代终止条件：返回执行**变异操作**。
        *   若**满足**迭代终止条件：继续下一步。
4.  **输出**: 输出结果。
5.  **结束**
