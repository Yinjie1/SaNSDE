#!/usr/bin/env python
# coding: utf-8

# ## 1 导入第三方库

# In[32]:


# 时间相关库
import datetime

# 科学计算库
import numpy as np
# import pandas as pd
import random
from deepmd.infer import DeepPot
import time
# 美化界面库
# from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 内存管理库
# import gc


# ## 2 设置训练参数

# In[33]:


# 固定随机数
# np.random.seed(2023)
# random.seed(2023)

# 总原子数
atom = 13


# 合金中Co原子个数
Co_rate = atom

# 多少次输出坐标10

# record_times = 1

# 种群中团簇个数
pool_size = 12

# 运行时间
running_time = 3600 * 36

# 总迭代次数
Max_step = 3000

# 局部最优迭代次数
Ite_step = 100


# 模型载入
model_path = '/home/linyj/01-yangwh/01-GA/graph.pb'  # 替换为你的模型路径
# t1 = time()
deep_pot = DeepPot(model_path)


# ## 3 自定义函数
# def check_atom_distances(coordinates, threshold=1.5):
#     """
#     检查两两原子之间的距离是否小于指定的阈值。
#     :param coordinates: 输入的原子坐标，形状为 (n, m, 3)，n 为团簇数，m 为原子数。
#     :param threshold: 距离阈值，默认为 2.0。
#     :return: 布尔值，True 表示所有距离都合法，False 表示存在小于阈值的距离。
#     """
#     for cluster in coordinates:  # 遍历每个团簇
#         num_atoms = cluster.shape[0]
#         for i in range(num_atoms):
#             for j in range(i + 1, num_atoms):
#                 distance = np.linalg.norm(cluster[i] - cluster[j])
#                 if distance < threshold:
#                     print(f"原子距离过近：距离={distance:.3f}, 原子对=({i}, {j})")
#                     return False
#     return True
# ### 3.1 计算：能量&力

# In[34]:
# 生成团簇中原子的初始坐标
def generate(atom, deep_pot):

    # 元素的第一紧邻距离
    correction_coefficient = 2  # Fe - 2.49

    # 随机生成坐标的球区域的半径
    R1 = (atom ** (1.0 / 3.0)) * correction_coefficient

    # 初始化团簇坐标：原子数*3
    temp_genes = np.zeros((atom, 3))
    coords = []

    for order in range(0, pool_size):

        # 团簇坐标：第8个团簇随机产生；第9，10团簇使用生长法n+1；11，12使用生长法n+2
        temp_genes = np.zeros((atom, 3))
        temp_genes[0][0]=0
        temp_genes[0][1]=0
        temp_genes[0][2]=0
        if order < 12:

            for i in range(1, atom):
                while(True):
                    idx_bondatom = np.random.randint(0, i)
                    bondl = np.random.rand()*1.0 + 2.0  # 2.0-3.0
                    bondtheta = np.random.rand()*3.141592654
                    bondphi = np.random.rand()*2.0*3.141592654
                    trial_x = []
                    trial_x.append(temp_genes[idx_bondatom][0] + bondl* np.cos(bondtheta)* np.cos(bondphi))
                    trial_x.append(temp_genes[idx_bondatom][1] + bondl* np.cos(bondtheta)* np.sin(bondphi))
                    trial_x.append(temp_genes[idx_bondatom][2] + bondl* np.sin(bondtheta))
                    trial_x = np.array(trial_x)
                    OK = True
                    for j in range(i):
                        dij = np.sqrt(np.square(trial_x[0]-temp_genes[j][0]) + np.square(trial_x[1]-temp_genes[j][1]) + np.square(trial_x[2]-temp_genes[j][2]))
                        if dij <= 1.75:
                            OK = False
                            break
                    if OK:
                        temp_genes[i]=trial_x
                        break            
            # 随机生成atom行3列
            # temp_genes = (np.random.rand(atom, 3) - 0.5) * R1
            # print("temp_genes:", temp_genes.shape)
        # 在局部优化前检查原子距离
        
        
        # elif 8 <= order < 9:
        #     filename = "/home/linyj/01-yangwh/01-GA/1-CoPt_refer/38Co1Pt37.txt"
        #     data = np.loadtxt(filename)
        #     temp_genes = np.reshape(data, (-1, 3)) 
        #     if not check_atom_distances(coords):
        #         print("生成的初始团簇中存在原子距离小于2的情况，停止优化。")
        #         return None  # 或者根据需求调整坐标后继续

        # elif 9 <= order < 10:
        #     filename = "/home/linyj/01-yangwh/01-GA/1-CoPt_refer/Pt38.txt"
        #     data = np.loadtxt(filename)
        #     temp_genes = np.reshape(data, (-1, 3))    
        
        # elif 10 <= order < 11:
        #     filename = "/home/linyj/01-yangwh/01-GA/1-CoPt_refer/Pt38.txt"
        #     data = np.loadtxt(filename)
        #     temp_genes = np.reshape(data, (-1, 3))
        
        # elif 11 <= order < 12:
        #     filename = "/home/linyj/01-yangwh/01-GA/1-CoPt_refer/38-Marks.txt"
        #     data = np.loadtxt(filename)
        #     temp_genes = np.reshape(data, (-1, 3))

        # else:
        #     filename = "/home/linyj/01-yangwh/01-GA/1-CoPt_refer/Co38.txt"
        #     data = np.loadtxt(filename)
        #     temp_genes = np.reshape(data, (-1, 3))
        
        temp_genes = np.array(temp_genes)
        coords.append(temp_genes)
    coords = np.array(coords)
    # print("coords:", coords.type)
    # print("1111111111111111111111111111")                                                  
    
    
    # 局部优化当前团簇
    individual = local_optimization(atom, coords, deep_pot)

    for i in range(0, pool_size):
        print("初始能量:", individual[0][i][0])

    return individual[0], individual[1] # 能量，坐标


# @jit(nopython=True)
# def check_distances(coordinates, threshold):
    
#     # 将坐标扩展为三维数组以进行矩阵操作
#     expanded_coordinates = np.expand_dims(coordinates, axis=1)

#     # 计算每对坐标之间的差异
#     differences = expanded_coordinates - coordinates

#     # 计算欧氏距离
#     distances = np.linalg.norm(differences, axis=2)

#     # 将对角线元素设置为大值以排除自身与自身的距离
#     np.fill_diagonal(distances, np.inf)

#     # 如果有距离大于阈值，输出“True”；否则，输出“False”
#     if np.any(distances > threshold):
#         return True
#     else:
#         return False


# In[35]:



def EF_cal(deep_pot, coord):
    # print("------coord--------:", coord.shape)
    # 将Atoms的信息转换为eval函数的输入格式
    cell = np.diag(20 * np.ones(3)).reshape([1, -1])
    cell = np.array([cell] * pool_size)
    atype = [0] * 6 + [1] * (atom - 6)
    # atype = [0 for k in range(atom)] # 0代表Co

    # 调用eval函数，计算能量和力
    start_time = time.time()
    e, f, _ = deep_pot.eval(coords=coord, cells=cell, atom_types=atype)
    end_time = time.time()
    eval_time = end_time - start_time
    print(f"Time taken for deep_pot.eval: {eval_time:.4f} seconds")
    energy = np.array(e)
    energy = np.nan_to_num(energy)
    force = np.array(f)
    # print("------energy--------:", energy)

    return energy, force # array: energy -> 12*1, force -> 12*atom*3


# ### 3.2 定局部优化

# In[36]:

def local_optimization(atom, example, deep_pot):

    # 转为numpy数组
    example = np.array(example)
    
    # 梯度下降总步数
    max_step = Ite_step

    # 当前梯度下降步数
    step = 1

    # 步长
    alpha = np.array([0.01] * pool_size)

    # 步长调整系数
    alpha_climbing_index = np.array([1.1] * pool_size)
    alpha_descend_index = np.array([0.6] * pool_size)
    e = 1E-6

    # 计算初始能量和力
    init_energy, Force = EF_cal(deep_pot, example)
    F_max = np.array([max(map(max, matrix)) for matrix in np.abs(Force)])

    # 梯度下降D
    while np.max(alpha) > e and step < max_step:
    # while step < max_step:D

        # 更新团簇坐标：坐标 + 步长 * (学习率/最大学习率)
        temp_coordinate = example + alpha[:, None, None] * Force / F_max[:, None, None]

        # 计算更新后团簇的能量
        temp_energy, Force = EF_cal(deep_pot, temp_coordinate)

        # 比较能量，更新团簇genes、fitness、alpha、F
        min_indices = np.argmin(np.concatenate([temp_energy, init_energy], axis=1), axis=1)
        # 在 min_indices 为 0 的位置将 array3 中的值替换为 array4 中对应位置的值
        example[min_indices == 0] = temp_coordinate[min_indices == 0]
        init_energy[min_indices == 0] = temp_energy[min_indices == 0]
        alpha[min_indices == 0] *= alpha_climbing_index[min_indices == 0]
        alpha[min_indices == 1] *= alpha_descend_index[min_indices == 1]
        F_max = np.array([max(map(max, matrix)) for matrix in np.abs(Force)])
        step += 1
    # print("局部优化次数", step)
    return [init_energy, example]


# ### 3.3 变异&交叉操作

# In[37]:


def map_function(temp_population, count):
    crossed_list = []
    CR_list = []
    for x, pair in enumerate(temp_population):

        # 产生随机数作为参与变异策略团簇的索引值，并确保三个索引值互异：index_1，index_2，index_3

        index_list = random.sample([i for i in range(pool_size) if i != x], 3)
        index_1 = index_list[0]
        index_2 = index_list[1]
        index_3 = index_list[2]

        # 执行交变异操作

        f_p = 0.5

        # 将当前团簇，和随机团簇1，2的中心移动到坐标原点
        center_this = np.mean(pair[1], axis=0)
        this_individual = pair[1] - center_this

        center_1 = np.mean(temp_population[index_1][1], axis=0)
        random_individual1 = temp_population[index_1][1] - center_1

        center_2 = np.mean(temp_population[index_2][1], axis=0)
        random_individual2 = temp_population[index_2][1] - center_2

        #  -----------------------------确定变异因子F_i-----------------------------

        # (1) 生成[0,1]随机数
        index = random.random()

        # (2) 若随机数小于f_p = 0.5，变异因子F为正态分布随机数；否则，F为标准柯西分布随机数
        if index < f_p:
            F_i = random.gauss(0.5, 0.3)
        else:
            # 标准柯西随机数，作为变异因子F
            F_i = np.random.standard_cauchy()

        # -------------------------- 确定变异策略进行变异操作-----------------------------

        # (1) 生成[0,1]随机数
        index = random.random()

        # (2) 若随机数小于策略选择概率p，则使用团簇1，2和3进行变异；否则，使用当前团簇，1，2，和最优进行变异
        if index < pair[8]:
            # Mutation Strategy 1
            center_3 = np.mean(temp_population[index_3][1], axis=0)
            random_individual3 = temp_population[index_3][1] - center_3
            mutated = random_individual1 + F_i * (random_individual2 - random_individual3)
        else:
            # Mutation Strategy 2
            population_first_Best = pair[-1]
            center_best = np.mean(population_first_Best, axis=0)
            population_first_Best = population_first_Best - center_best
            mutated = this_individual + F_i * (population_first_Best - this_individual) + F_i * (random_individual1 - random_individual2)

        # 记录变异策略，用于更新选择概率p
        if index < pair[8]:
            strategy = 1
        else:
            strategy = 2 

        # 执行交叉操作

        CR_i = random.gauss(pair[9], 0.1)

        # 设置jrand：[1,atom)的整数，来确保交叉团簇中至少有一个原子来自变异团簇
        jrand = np.random.randint(0, atom - 1)

        crossed = np.zeros_like(pair[1])

        point = jrand
        step_count = 0

        # ------------------------交叉团簇的第point个原子坐标信息来自变异团簇还当前团簇--------------------
        while step_count < atom:
            r = np.random.random()
            if r <= CR_i or point == jrand:
                crossed[point] = mutated[point]
            else:
                crossed[point] = pair[1][point]

            if point == atom - 1:
                point = 0
            else:
                point += 1

            step_count += 1
        
        # 记录本次迭代交叉后的团簇crossed_individual.genes, 交叉概率crossed_individual.CR
        crossed_list.append(crossed)
        CR_list.append(CR_i)
        
    # 局部优化交叉后的团簇
    crossed_clu = local_optimization(atom, crossed_list, deep_pot)

    # 记录本次迭代的能量差
    energy_ori, _ = EF_cal(deep_pot, np.array([arr.tolist() for arr in temp_population[:, 1]]))

    delta_fitness = energy_ori - crossed_clu[0]

    # 比较能量，更新团簇genes、fitness、alpha、F
    min_indices = np.argmin(np.concatenate([crossed_clu[0], energy_ori], axis=1), axis=1)
    # 比较交叉后的团簇的能量与当前团簇的能量，更新团簇；记录交叉概率，能量差，变异策略
    temp_population[min_indices == 0, 1] = [np.array(matrix) for matrix in crossed_clu[1][min_indices == 0]]
    temp_population[min_indices == 0, 6] = np.array(CR_list)[min_indices == 0]
    temp_population[min_indices == 0, 7] = [np.array(matrix) for matrix in delta_fitness[min_indices == 0]]
    temp_population[min_indices == 0, 10] = [np.array(matrix) for matrix in crossed_clu[0][min_indices == 0]]

    # 比较交叉变异操作后当前种群能量最低的团簇的坐标
    tempDE_best_idx = np.nanargmin(temp_population[:, 10])
    # tempDE_best_idx = np.argmin(temp_population[:, 10])
    # 变异后的最优个体与变异前的最优个体比较能量，更新坐标和能量
    if temp_population[tempDE_best_idx, 10] < temp_population[tempDE_best_idx, -2]:
        temp_population[:, -1] = [temp_population[tempDE_best_idx, 1]] * pool_size # 更新当前种群最优团簇坐标
        temp_population[:, -2] = temp_population[tempDE_best_idx, 10] # 更新当前种群最优团簇能量
        # 更新当前种群最优团簇信息
        count = 0
    else:
        count += 1
        
    return temp_population, count


# ### 3.4 球切操作

# In[38]:


def update_fitness_SC(temp_population, count):

    child1_genes_list = []
    child2_genes_list = []

    for pair in temp_population:


        # 将父本母本团簇的中心移动到坐标原点
        center_father = np.mean(pair[1], axis=0)
        father = pair[1] - center_father

        center_mother = np.mean(pair[-1], axis=0)
        mother = pair[-1] - center_mother

        child1_genes = father.copy()  # 默认使用父本基因
        child2_genes = mother.copy()  # 默认使用母本基因
        # 将每个原子的三维坐标作和
        frd = np.sum(father, axis=1)

        mrd = np.sum(mother, axis=1)

        frdrank = np.copy(frd)

        mrdrank = np.copy(mrd)

        # 寻找交叉点
        existId = []

        for i in range(atom - 1):
            for j in range(i + 1, atom):
                if frd[i] == frd[j]:
                    frd[i] += (2 + np.random.random()) * 1e-16
                    frdrank[i] = frd[i]
                if mrd[i] == mrd[j]:
                    mrd[i] += (2 + np.random.random()) * 1e-16
                    mrdrank[i] = mrd[i]

        frdrank.sort()

        mrdrank.sort()

        for k in range(1, atom):
            if mrdrank[k - 1] < frdrank[k - 1] and frdrank[k] < mrdrank[k]:
                existId.append(k)

        ExistId = np.array(existId)

        if len(ExistId) == 0:
            child1_genes = father.copy()
            child2_genes = mother.copy()
        else:
            mid = int(np.ceil(ExistId.size * np.random.random()))

            if mid == 0:
                mid = 1

            kcross = ExistId[mid - 1]
        
        # 进行球切交叉操作
            child1_genes[(frd < frdrank[kcross] - 1e-16), :] = father[(frd < frdrank[kcross] - 1e-16), :]
            child1_genes[(mrd > mrdrank[kcross] + 1e-16), :] = mother[(mrd > mrdrank[kcross] + 1e-16), :]
            child2_genes[(frd > frdrank[kcross] + 1e-16), :] = father[(frd > frdrank[kcross] + 1e-16), :]
            child2_genes[(mrd < mrdrank[kcross] - 1e-16), :] = mother[(mrd < mrdrank[kcross] - 1e-16), :]
        child1_genes_list.append(child1_genes)
        child2_genes_list.append(child2_genes)

    child1 = local_optimization(atom, np.array(child1_genes_list), deep_pot)
    child2 = local_optimization(atom, np.array(child2_genes_list), deep_pot)

    # 比较能量，更新团簇genes、fitness、alpha、F
    min_indices = np.argmin(np.concatenate([child2[0], child1[0]], axis=1), axis=1)
    # 比较球切操作后的团簇的能量与当前团簇的能量，更新团簇；记录交叉概率，能量差，变异策略
    temp_population[min_indices == 0, 1] = [np.array(matrix) for matrix in child2[1][min_indices == 0]]
    temp_population[min_indices == 0, 10] = [np.array(matrix) for matrix in child2[0][min_indices == 0]]
    temp_population[min_indices == 1, 1] = [np.array(matrix) for matrix in child1[1][min_indices == 1]]
    temp_population[min_indices == 1, 10] = [np.array(matrix) for matrix in child1[0][min_indices == 1]]
        
    # 比较球切交叉操作后当前种群能量最低的团簇的坐标
    tempDE_best_idx = np.nanargmin(temp_population[:, 10])
    # 球切交叉后的最优个体与球切交叉前的最优个体比较能量，更新坐标和能量
    if temp_population[tempDE_best_idx, 10] < temp_population[tempDE_best_idx, -2]:
        temp_population[:, -1] = [temp_population[tempDE_best_idx, 1]] * pool_size
        temp_population[:, -2] = temp_population[tempDE_best_idx, 10]
        # 更新当前种群最优团簇信息
        count = 0
    else:
        count += 1
    # print("-----------------",temp_population)
    return temp_population, count


# ### 3.5 局部平移操作

# In[39]:


def update_fitness_Dist(temp_population, count):

    temp_genes_list = []

    for pair in temp_population:
        adjust_index = 0.5

        temp_genes = np.copy(pair[1])

        # 对所有原子的三维坐标加随机扰动，每个坐标+[-0.25, 0.25]随机数

        temp_genes += (np.random.random((atom, 3)) - 0.5) * adjust_index
        temp_genes_list.append(temp_genes)
    
    temp = local_optimization(atom, np.array(temp_genes_list), deep_pot)
    
    min_indices = np.argmin(np.concatenate([temp[0], [arr.tolist() for arr in temp_population[:, 10]]], axis=1), axis=1)
    # 比较球切操作后的团簇的能量与当前团簇的能量，更新团簇；记录交叉概率，能量差，变异策略
    temp_population[min_indices == 0, 1] = [np.array(matrix) for matrix in temp[1][min_indices == 0]]
    temp_population[min_indices == 0, 10] = [np.array(matrix) for matrix in temp[0][min_indices == 0]]

    # 比较随机扰动操作后当前种群能量最低的团簇的坐标
    tempDE_best_idx = np.nanargmin(temp_population[:, 10])
    # 随机扰动的最优个体与随机扰动前的最优个体比较能量，更新坐标和能量
    if temp_population[tempDE_best_idx, 10] < temp_population[tempDE_best_idx, -2]:
        temp_population[:, -1] = [temp_population[tempDE_best_idx, 1]] * pool_size
        temp_population[:, -2] = temp_population[tempDE_best_idx, 10]
        # 更新当前种群最优团簇信息
        count = 0
    else:
        count += 1
        
    return temp_population, count


# ### 3.6 定义函数calculate_CR：计算交叉策略CR值

# In[40]:


# @jit(nopython=True)
def calculate_CR(CR_rec, F_rec, CR_m):
    cr_rec = np.array(CR_rec)
    f_rec = np.array(F_rec)
    cr_m = 0.0
    if len(CR_rec) == 0:
        cr_m = CR_m
    else:
        sum_f = sum(f_rec)
        for i in range(len(cr_rec)):
            f_rec[i] = f_rec[i] / sum_f
            cr_m += f_rec[i] * cr_rec[i]

    return cr_m


# ### 3.7 定义函数update_pp：更新变异系数p值

# In[41]:


# @jit(nopython=True)
def update_CRm(pair):
    pair[9] = calculate_CR(pair[6], pair[7], pair[9])
    pair[6].clear()
    pair[7].clear()

    return pair


# ### 3.8 初始化矩阵

# In[42]:



# 生成初始矩阵
def build_RDD(pool_size, atom, deep_pot):
        
    # 生成序列[索引值（0~12）, 实例化为Chromosome的团簇:
    # [nS_1, nS_2, nF_1, nF_2, CR, F, p, CRm, 当前团簇能量, 最优团簇能量, 最优团簇坐标]，将数据分为12个分区
    rdd = []
    energy, pos = generate(atom, deep_pot)
    for i in range(0, pool_size):
        rdd.append([i, pos[i], 0, 0, 0, 0, [], [], 0.5, 0.5, energy[i], 0, np.zeros((atom, 3))])

    return np.array(rdd, dtype=object) 


# ## 4 主函数运行

# In[44]:


# 记录开始运行时间
start_time = time.time()

# 运行种群进化

time_start = time.time()

current = build_RDD(pool_size, atom, deep_pot)

# 当前迭代步数
step = 0

count = 0
update = 0

update_p = 50 

# 更新交叉概率所需参数

update_CR = 50  

every_moment = 0.0

# 获取当前时间
now = datetime.datetime.now().strftime("%m-%d %H-%M-%S")

# 创建一个文件名字符串，包含当前时间和总原子数信息
time_file_name = "/home/linyj/01-yangwh/01-GA/result-01/Co{}-recond-{}.txt".format(atom, now)
# 创建一个文件名字符串，包含当前迭代次数、时间和原子数坐标
coord_file_name = "/home/linyj/01-yangwh/01-GA/result-01/Co{}-coord-{}.txt".format(atom, now)


# In[1]:


while every_moment < running_time:
# while step < Max_step:
# while step < 30:

    # -----------------current------>>(变异交叉操作)-------->>DE_RDD-------------------------
    current, update = map_function(current, update)

    # -----------------DE_RDD------>>(球切操作)-------->>SphereCut_RDD-----------------------
    current, update = update_fitness_SC(current, update)

    # -----------------SphereCut_RDD------>>(局部变换操作)-------->>Dis_RDD------------------
    current, update = update_fitness_Dist(current, update)
    
    # 局部变换后的最优个体与变换前的最优个体比较能量，更新坐标和能量
    if update > 2:
        print("未找到更低能量", current[0, -2])
        count += 1
    else:
        count = 0

    # 计算运行时间
    every_time_end = time()
    every_moment = every_time_end - time_start
    step += 1
    print("当前代数:{}\t重复最优代数:{}\t已用时间:{}".format(step, count, every_moment))
    print("the best energy is:", current[0, -2])

    # 打开文件开始写入
    with open(time_file_name, "a") as time_writer:
        # 写入这一代的运行时间和最优能量

        time_writer.write("{}\t {}\t {} \n".format(step, every_moment, current[0, -2]))
    
    with open(coord_file_name, "a") as coord_writer:
        if count == 0:
            coord_writer.write(str(atom) + "\n")
            coord_writer.write("Co" + "," + str(current[0, -2]) + "\n")
            pos_best = current[0, -1]
            for i in range(atom):
                coord_writer.write(
                    "Co" + "\t" + format(pos_best[i, 0], ".6f") + "\t" + format(pos_best[i, 1], ".6f") + "\t" + format(
                        pos_best[i, 2], ".6f") + "\n")


# In[46]:


# 写输出文件: 记录最优团簇的能量和坐标
File_name = f"/home/gzw/workspace/Anaconda/01_Vera/result-01/Co{atom}-best-{now}.txt"
best_ordinary = current[0, -1]

with open(File_name, 'w') as writer_point:
    writer_point.write(str(atom) + "\n")
    writer_point.write("Co" + "," + str(current[0, -2]) + "\n")
    for i in range(atom):
        writer_point.write(
            "Co" + "\t" + format(best_ordinary[i, 0], ".6f") + "\t" + format(best_ordinary[i, 1], ".6f") + "\t" + format(
                best_ordinary[i, 2], ".6f") + "\n")

print("Final Energy:", current[0, -2], "eV")

