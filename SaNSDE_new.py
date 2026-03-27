"""SaNSDE_new.py
python 3.13.12
"""
import os
import json
import copy
import functools
import numpy as np
from typing import Any, Callable
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS, LBFGS, GPMin, FIRE
from ase.optimize.optimize import Optimizer
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from datetime import datetime


class Population:
    """种群对象
    
    包含一组Atoms团簇对象, 管理其能量和力. 
    种群中所有团簇必须具有相同的化学式(symbol).

    Parameters
    ----------
    symbol : str | None = None
        团簇的化学式 (e.g., "Au13"). 如果未指定, 将在添加第一个团簇时自动设定.

    Attributes
    ----------
    clusters : list[Atoms]
        当前种群中的团簇列表.
    symbol : str | None
        种群化学式. 仅允许在未设置时初始化一次.
    num : int
        单个团簇的原子数. 当 symbol 未设置时为 0.
    fp : float
        变异缩放因子采样分布的选择概率参数.
    p : float
        变异策略选择概率参数.
    CRm : float
        交叉率中心参数.
    energies : list[float | None]
        按需从每个团簇对象属性 ``cluster.energy`` 读取的能量列表.
        若某团簇缺失该属性, 对应位置为 None.
    forces : list[list[list[float]] | None]
        按需从每个团簇对象属性 ``cluster.forces`` 读取的力列表.
        若某团簇缺失该属性, 对应位置为 None.

    Methods
    -------
    addCluster(cluster: Atoms, deepcopy: bool = True) -> None
        添加团簇到种群.
    export(name: str | None = None, xyz: bool = False) -> str
        导出种群快照为 JSON, 并可选导出 xyz 结构文件.
    import_population(filename: str, confirm: bool = False) -> None
        从 JSON 文件导入种群.
        
    Examples
    --------
    初始化种群:
    >>> pop = Population(symbol="Au13")

    添加团簇:
    >>> cluster = Atoms('Au13', positions=...)
    >>> pop.addCluster(cluster)

    导出种群:
    >>> json_path = pop.export(name="Gen_1")

    导入种群:
    >>> pop_new = Population()
    >>> pop_new.import_population(filename=json_path)
    """
    def __init__(self, symbol: str | None = None):
        self._clusters: list[Atoms] = []
        self._symbol: str | None = symbol
        self._fp: float = 0.5
        self._p: float = 0.5
        self._CRm: float = 0.5

    @property
    def clusters(self) -> list[Atoms]:
        return self._clusters

    @property
    def fp(self) -> float:
        return self._fp

    @property
    def p(self) -> float:
        return self._p

    @property
    def CRm(self) -> float:
        return self._CRm

    @property
    def symbol(self) -> str | None:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        if self._symbol is not None:
            raise ValueError("symbol已经被设置!")
        self._symbol = value

    def __iter__(self):
        return iter(self._clusters)

    def __getitem__(self, index: int) -> Atoms:
        return self._clusters[index]

    def __len__(self) -> int:
        return len(self._clusters)

    @property
    def num(self) -> int:
        if self._symbol is None:
            return 0
        # 根据化学式symbol推出一个团簇的原子数(int)
        temp = Atoms(self._symbol)
        return len(temp)

    @property
    def energies(self) -> list[float | None]:
        """按需从团簇对象属性提取能量,缺失时返回 None."""
        energies: list[float | None] = []
        for cluster in self:
            energyValue = getattr(cluster, 'energy', None)
            if energyValue is not None:
                energyValue = float(energyValue)
            energies.append(energyValue)
        return energies

    @property
    def forces(self) -> list[list[list[float]] | None]:
        """按需从团簇对象属性提取力,缺失时返回 None."""
        forces: list[list[list[float]] | None] = []
        for cluster in self:
            forceValue = getattr(cluster, 'forces', None)
            forces.append(forceValue)
        return forces

    def addCluster(self, cluster: Atoms, deepcopy: bool = True):
        """导入一个团簇对象(Atoms类型)到_clusters列表末尾.

        必须确保种群中所有团簇化学式相同.
        在添加前会将团簇质心平移到原点.
        若种群 symbol 为空, 则在添加第一个团簇时自动设置.

        Parameters
        ----------
        cluster : Atoms
            ASE Atoms 对象, 代表一个团簇.
            其化学式必须与种群设定的 symbol 一致.
        deepcopy : bool, optional
            是否在添加时对 cluster 做深拷贝. Default True.

        Returns
        -------
        None
            该方法原地修改种群对象, 无返回值.
        """
        if not isinstance(cluster, Atoms):
            raise TypeError("Cluster must be an ase.Atoms object")

        clusterToAdd = copy.deepcopy(cluster) if deepcopy else cluster
        # 平移质心到原点
        clusterToAdd.set_center_of_mass([0.0, 0.0, 0.0])

        current_symbol = clusterToAdd.get_chemical_formula()
        
        if self._symbol is None:
            self.symbol = current_symbol 
        else:
            # 必须确保种群中所有团簇化学式相同
            if current_symbol != self._symbol:
                 raise ValueError(f"Cluster symbol {current_symbol} does not match population symbol {self._symbol}")
            
        self._clusters.append(clusterToAdd)

    def export(self, name: str | None = None, xyz: bool = False) -> str:
        """导出种群数据到文件系统.

        在当前目录'./Populations/'(如不存在则新建)下输出一个JSON文件.
        JSON文件名为'name.json'(缺省时name为种群名称).
        当 xyz=True 时, 会同时新建同名文件夹 name/, 其中输出团簇xyz文件.

        Parameters
        ----------
        name : str, optional
            导出名称. 默认使用种群的 symbol.
        xyz : bool, optional
            是否同时导出团簇xyz文件. Default False.

        Returns
        -------
        str
            导出JSON文件的完整绝对路径.
        """
        if name is None:
            name = self.symbol if self.symbol else "Population"
            
        # Create directory ./Populations/name
        base_dir = './Populations'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        target_dir = os.path.join(base_dir, name)

        # Check for duplicate JSON filename
        summary_path = os.path.join(base_dir, f"{name}.json")
        if os.path.exists(summary_path):
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            name = f"{name}_{timestamp}"
            target_dir = os.path.join(base_dir, name)
            summary_path = os.path.join(base_dir, f"{name}.json")
            print(f'Duplicate file found. New export name: {name}!')

        # Optional xyz snapshot directory
        if xyz:
            target_dir = os.path.join(base_dir, name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Export clusters as .xyz files (extxyz content)
            for i, cluster in enumerate(self._clusters):
                fname = f"{self.symbol}_{i}.xyz"
                fpath = os.path.join(target_dir, fname)

                exportCluster = cluster.copy()
                if self.energies[i] is not None:
                    setattr(exportCluster, 'energy', self.energies[i])

                try:
                    write(fpath, exportCluster, format='extxyz', write_results=False)
                except Exception as e:
                    print(f"Warning: Failed to write {fname}: {e}")
        
        min_e = None
        min_idx = None
        
        valid_energies = [(i, e) for i, e in enumerate(self.energies) if e is not None]
        if valid_energies:
            min_idx, min_e = min(valid_energies, key=lambda x: x[1])
            if isinstance(min_e, np.generic):
                min_e = min_e.item()

        clusters_info = []
        for i, cluster in enumerate(self._clusters):
            # 准备单个团簇的信息
            force_data = self.forces[i]
            if force_data is not None and isinstance(force_data, np.ndarray):
                force_data = force_data.tolist()
            elif force_data is None:
                force_data = None

            energyValue = self.energies[i]
            if isinstance(energyValue, np.generic):
                energyValue = energyValue.item()

            cluster_info = {
                "name": f"{self.symbol}_{i}",
                "symbol": cluster.get_chemical_formula(),
                "energy": energyValue,
                "positions": cluster.get_positions().tolist(),
                "forces": force_data
            }
            clusters_info.append(cluster_info)

        summary_data = {
            "date": datetime.now().strftime('%Y%m%d%H%M%S'),
            "symbol": self.symbol,
            "size": len(self),
            "min_energy": min_e,
            "min_energy_index": min_idx,
            "clusters": clusters_info
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)

        return os.path.abspath(summary_path)

    def import_population(self, filename: str, confirm: bool = False):
        """从JSON文件导入种群数据.

        直接读取指定的JSON文件.
        直接使用JSON文件中的数据重建Atoms对象, 能量和力.
        如果种群非空, 需要设置 confirm=True 才能覆盖现有数据.

        Parameters
        ----------
        filename : str
            JSON文件的完整路径.
        confirm : bool, optional
            是否确认覆盖当前种群数据. Default False.
        """
        if not os.path.exists(filename):
             print(f"JSON file {filename} not found.")
             return

        if len(self) > 0:
            if not confirm:
                print("Population is not empty. Set confirm=True to overwrite.")
                return 
            else:
                self._clusters = []
                self._symbol = None

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            return
            
        # Set population attributes from JSON
        if self._symbol is None:
             self._symbol = data.get('symbol')
        
        if self._symbol != data.get('symbol'):
             print(f"Warning: JSON symbol ({data.get('symbol')}) does not match population symbol ({self._symbol})")
             # Proceeding anyway, but this is a potential issue if user intends strict type checking
        
        clusters_data = data.get('clusters', [])
        
        for c_data in clusters_data:
            try:
                # Reconstruct Atoms object
                symbol = c_data.get('symbol')
                positions = c_data.get('positions')
                
                # Ensure positions is a list or numpy array
                if positions is None:
                    print(f"Skipping cluster {c_data.get('name')} due to missing positions.")
                    continue
                    
                cluster = Atoms(symbol, positions=positions)
                
                # Add to population
                # Note: self.addCluster appends None to energies/forces, so we overwrite them below
                self._clusters.append(cluster)
                
                # Set energy and forces on cluster object
                energy = c_data.get('energy')
                forces = c_data.get('forces')
                
                # Optional: Attach attributes to Atoms object for convenience
                if energy is not None:
                     setattr(cluster, 'energy', float(energy))
                if forces is not None:
                     setattr(cluster, 'forces', forces)
                
            except Exception as e:
                print(f"Error loading cluster from JSON: {e}")



def localOptimize(pop: Population | Atoms, 
                  cell: list[float] | np.ndarray = [15, 15, 15], 
                  potential: Calculator | None = None,
                  optimizer: type[Optimizer] = BFGS,
                  fmax: float = 0.05, 
                  steps: int = 100,
                  deepcopy: bool = True) -> Population | Atoms:
    """进行局域优化(结构弛豫)
    
    对种群中的所有团簇或单个团簇进行结构优化, 返回优化后的新对象.
    不会原地修改输入对象.
    会自动添加周期性元胞(cell)进行计算, 并在优化完成后清除元胞信息.

    Parameters
    ----------
    pop : Population or Atoms
        待优化的对象, 可以是整个种群对象(Population)或单个团簇(Atoms).
    cell : list or np.ndarray, optional
        用于计算的模拟盒子大小, default [15, 15, 15].
    potential : Calculator, optional
        ASE计算器对象(e.g., LJ, EMT, DeepMD等).
        当 potential=None 时:
        - 对 Population: 遍历查找第一个已有 calc 的团簇,并将该 calc 应用于整个种群,打印 warning;
          若未找到则报错.
        - 对 Atoms: 要求 atoms.calc 已存在,否则报错.
    optimizer : type[Optimizer], optional
        ASE 优化器类(Optimizer 子类), 例如 BFGS/LBFGS/FIRE.
        Default BFGS.
    fmax : float, optional
        收敛判据, 最大力阈值 (eV/Ang). Default 0.05.
    steps : int, optional
        最大优化步数. Default 100.
    deepcopy : bool, optional
        是否深拷贝输入对象后再优化. Default True.
        为 False 时直接使用输入对象引用进行后续流程.
        
    Returns
    -------
    Population | Atoms
        返回与输入类型一致的新对象.
        输入为 Population 时返回优化后的新 Population.
        输入为 Atoms 时返回优化后的新 Atoms.

    Examples
    --------
    >>> popNew = localOptimize(pop, potential=calc, optimizer='LBFGS')
    """
    if isinstance(pop, Population):
        newClusters = copy.deepcopy(pop) if deepcopy else pop
        isSingle = False
    elif isinstance(pop, Atoms):
        newClusters = Population()
        newClusters.addCluster(pop, deepcopy=deepcopy)
        isSingle = True
    else:
        raise TypeError("pop must be a Population or an ase.Atoms object")

    inheritedCalc: Calculator | None = None
    if potential is None:
        for atoms in newClusters:
            if atoms.calc is not None:
                inheritedCalc = atoms.calc
                break
        if inheritedCalc is None:
            raise ValueError("potential is None and no calculator found in input clusters")
        if isinstance(pop, Population):
            print("Warning: potential is None. Inherited first found cluster calculator for all clusters.")

    for atoms in newClusters:
        # cell 自动添加到团簇对象上
        atoms.set_cell(cell)
        atoms.set_pbc(False)

        # 计算器
        atoms.calc = inheritedCalc if potential is None else potential

        # Optimizer selection: 统一使用 ASE Optimizer 子类
        opt_class = optimizer
        if not isinstance(opt_class, type) or not issubclass(opt_class, Optimizer):
            raise TypeError("optimizer must be an ASE Optimizer subclass")

        if atoms.calc is None:
             continue

        try:
            dyn = opt_class(atoms, logfile=None) # type: ignore 
            dyn.run(fmax=fmax, steps=steps)
        except Exception as e:
            print(f"Optimization failed: {e}")

        # 只同步到团簇对象属性,Population.energies/forces 按需读取这里.
        energyValue = None
        forceValue = None
        if atoms.calc is not None:
            try:
                energyValue = float(atoms.get_potential_energy())
            except Exception:
                energyValue = None
            try:
                forceValue = atoms.get_forces().tolist()
            except Exception:
                forceValue = None
        setattr(atoms, 'energy', energyValue)
        setattr(atoms, 'forces', forceValue)

        # 计算后清除 cell
        # del atoms.cell # cell attribute deletion is not supported
        atoms.set_cell([0.0, 0.0, 0.0]) # Effectively clear cell dimensions

    if isSingle:
        return newClusters.clusters[0]
    return newClusters


# 局域优化偏函数:固定常用参数,供 select 等流程直接复用
relax = functools.partial(
    localOptimize,
    cell=[15, 15, 15],
    potential=LennardJones(),
    optimizer=BFGS,
    fmax=0.05,
    steps=100,
)


def randomCluster(pop: Population, 
                  a: float, 
                  delta: float, 
                  cell: list[float] | np.ndarray, 
                  potential: Calculator,
                  optimizer: Any, 
                  fmax: float, 
                  steps: int):
    """随机生成一个团簇并添加进种群.
    
    使用生长法随机生成团簇初始坐标, 并进行局域优化后加入种群.
    生成过程: 
    1. 第一个原子置于原点.
    2. 后续原子在已生成原子的附近随机生成 (键长 bondl, 方向随机).
    3. 检查新原子与已有原子的距离, 若过近(<0.75a)则重新生成.

    Parameters
    ----------
    pop : Population
        目标种群对象.
    a : float
        晶格常数或第一近邻距离 (Angstrom). 用于控制生成团簇的致密程度.
    delta : float
        键长扰动因子 (0~1). bondl = (1 + delta * uniform(-0.25, 0.75)) * a.
    cell : list or np.ndarray
        模拟盒子大小.
    potential : Calculator, optional
        ASE计算器.
    optimizer : str or Optimizer, optional
        优化器类型.
    fmax : float
        力收敛判据.
    steps : int
        最大优化步数.
        
    Examples
    --------
    >>> randomCluster(pop, a=2.5, delta=0.5, cell=[20]*3, potential=calc, optimizer='BFGS', fmax=0.02, steps=200)
    """
    N = pop.num
    if N == 0:
        if pop.symbol:
             N = len(Atoms(pop.symbol))
        else:
             raise ValueError("Population symbol not set, cannot determine cluster size.")

    # R0 = 2*a*(1/2+(3*N/(4*np.pi*np.sqrt(2)))**(1/3))
    R0 = 2 * a * (0.5 + (3 * N / (4 * np.pi * np.sqrt(2)))**(1.0/3.0))

    # Initialize positions
    dummy = Atoms(pop.symbol)
    symbols = dummy.get_chemical_symbols()
    
    positions = np.zeros((N, 3))
    positions[0] = [0.0, 0.0, 0.0]

    for i in range(1, N):
        while True:
            idx_bondatom = np.random.randint(0, i)
            
            # bondl = (1+delta*np.random.uniform(-0.25,0.75))*a
            bondl = (1 + delta * np.random.uniform(-0.25, 0.75)) * a
            
            # 随机角方向
            bondtheta = np.random.rand() * np.pi 
            bondphi = np.random.rand() * 2 * np.pi
            
            # 生成一个新原子
            x = positions[idx_bondatom][0] + bondl * np.cos(bondtheta) * np.cos(bondphi)
            y = positions[idx_bondatom][1] + bondl * np.cos(bondtheta) * np.sin(bondphi)
            z = positions[idx_bondatom][2] + bondl * np.sin(bondtheta)
            
            trial_pos = np.array([x, y, z])
            
            # if 该原子和其他原子距离 < 0.75*a: 重新生成
            dists = np.linalg.norm(positions[:i] - trial_pos, axis=1)
            if np.any(dists < 0.75 * a):
                continue
            
            positions[i] = trial_pos
            break

    # Create cluster
    cluster = Atoms(symbols=symbols, positions=positions)
    
    # localOptimize
    optimizedCluster = localOptimize(
        cluster,
        cell=cell,
        potential=potential,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        deepcopy=False,
    )
    if not isinstance(optimizedCluster, Atoms):
        raise TypeError("localOptimize should return Atoms when input is Atoms")
    
    # Add to pop
    pop.addCluster(optimizedCluster, deepcopy=False)
    
    # Update energy and forces
    energyValue = getattr(optimizedCluster, 'energy', None)
    forceValue = getattr(optimizedCluster, 'forces', None)
    setattr(pop.clusters[len(pop) - 1], 'energy', energyValue)
    setattr(pop.clusters[len(pop) - 1], 'forces', forceValue)


def mutate(pop: Population) -> Population:
    """对种群执行突变,返回突变后的暂定种群."""
    if len(pop) < 4:
        raise ValueError("Mutation requires at least 4 clusters in population")

    tempClusters = copy.deepcopy(pop)

    # 选取当前最优个体索引(最低能量),要求所有个体能量都已就绪
    energies = pop.energies
    for i, energyValue in enumerate(energies):
        if energyValue is None:
            raise ValueError(f"All clusters must have energy before mutation. Missing index: {i}")
    indexBest = int(np.argmin(np.array(energies, dtype=float)))

    for i in range(len(pop)):
        # 采样缩放因子 F:按 fp 在高斯/柯西分布之间切换
        if np.random.uniform(0.0, 1.0) < tempClusters.fp:
            scaleFactor = float(np.random.normal(0.5, 0.3))
        else:
            scaleFactor = float(np.random.standard_cauchy())

        # 从除 i 外的个体中采样互异索引
        candidateIndices = [idx for idx in range(len(pop)) if idx != i]
        index1, index2, index3 = np.random.choice(candidateIndices, size=3, replace=False)

        # 按 p 在两种突变策略之间切换
        if np.random.uniform(0.0, 1.0) < tempClusters.p:
            # DE/rand/1
            newPositions = (
                pop[index1].get_positions()
                + scaleFactor * (pop[index2].get_positions() - pop[index3].get_positions())
            )
            mutTag = 0
        else:
            # DE/current-to-best/2
            newPositions = (
                pop[i].get_positions()
                + scaleFactor * (pop[indexBest].get_positions() - pop[i].get_positions())
                + scaleFactor * (pop[index1].get_positions() - pop[index2].get_positions())
            )
            mutTag = 1

        tempClusters[i].set_positions(newPositions)
        setattr(tempClusters[i], 'mut', mutTag)
        setattr(tempClusters[i], 'energy', None)
        setattr(tempClusters[i], 'forces', None)

    return tempClusters


def crossover(pop1: Population, pop2: Population) -> Population:
    """对原种群与突变种群执行交叉,返回交叉后种群."""
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for crossover")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for crossover")

    tempClusters = Population(symbol=pop1.symbol)
    tempClusters._fp = pop1.fp
    tempClusters._p = pop1.p
    tempClusters._CRm = pop1.CRm

    for i in range(len(pop1)):
        # 采样当前个体交叉率,并裁剪到 [0, 1].
        # 裁剪是为了保证交叉概率合法,避免出现负值或大于 1.
        crossRate = np.random.normal(pop1.CRm, 0.1)
        crossRate = np.clip(crossRate, 0.0, 1.0)

        # 至少保留一个原子来自突变种群
        jrand = np.random.randint(0, pop1.num)

        # 使用坐标拷贝进行交叉,保证不原地改动输入种群 pop1/pop2.
        basePositions = pop1[i].get_positions().copy()
        mutPositions = pop2[i].get_positions().copy()
        trialPositions = basePositions.copy()

        # 逐原子交叉:按 CR 或 jrand 选择保留突变位姿
        for j in range(pop1.num):
            if np.random.uniform(0.0, 1.0) <= crossRate or j == jrand:
                trialPositions[j] = mutPositions[j]

        trialCluster = Atoms(
            symbols=pop1[i].get_chemical_symbols(),
            positions=trialPositions,
        )
        setattr(trialCluster, 'CR', crossRate)
        setattr(trialCluster, 'mut', getattr(pop2[i], 'mut', None))

        tempClusters.addCluster(trialCluster)

    return tempClusters


def SphereCutSplice(
    pop1: Population,
    pop2: Population,
    r: float = 0.8,
    relax: Callable[[Population | Atoms], Population | Atoms] | None = None,
) -> Population:
    """球切交叉:按球面划分并交换外层原子坐标,择优进入下一代.

    Parameters
    ----------
    pop1 : Population
        父代种群 1.
    pop2 : Population
        父代种群 2.
    r : float, optional
        半径缩放系数,取值范围 (0, 1].
        球面半径定义为 R = dmaxA * r.
    relax : Callable, optional
        局域优化函数. 可传入偏函数 relax() 或函数对象 localOptimize().
        若使用默认值 None, 则直接报错.

    Returns
    -------
    Population
        经过球切交叉与择优后的新种群.
    """
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for SphereCutSplice")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for SphereCutSplice")
    if len(pop1) < 2:
        raise ValueError("SphereCutSplice requires at least 2 clusters per population")
    if not (0.0 < r <= 1.0):
        raise ValueError("r must be in (0, 1]")

    tempClusters = Population(symbol=pop1.symbol)
    tempClusters._fp = pop1.fp
    tempClusters._p = pop1.p
    tempClusters._CRm = pop1.CRm

    if relax is None:
        raise ValueError("SphereCutSplice requires explicit relax callable")

    def spliceOne(clusterA: Atoms, clusterB: Atoms) -> Atoms | None:
        workA = copy.deepcopy(clusterA)
        workB = copy.deepcopy(clusterB)

        # 这里按需求使用质心(center of mass)移动到原点,不使用几何中心.
        workA.set_center_of_mass([0.0, 0.0, 0.0])
        workB.set_center_of_mass([0.0, 0.0, 0.0])

        positionsA = workA.get_positions().copy()
        positionsB = workB.get_positions().copy()
        symbolsA = workA.get_chemical_symbols()
        symbolsB = workB.get_chemical_symbols()

        distA = np.linalg.norm(positionsA, axis=1)
        distB = np.linalg.norm(positionsB, axis=1)

        dmaxA = float(np.max(distA))
        radius = dmaxA * r

        symbolSet = sorted(set(symbolsA))
        swapPairs: list[tuple[int, int]] = []

        for symbol in symbolSet:
            outerIndicesA = [
                i for i, atomSymbol in enumerate(symbolsA)
                if atomSymbol == symbol and distA[i] > radius
            ]

            countOuter = len(outerIndicesA)
            if countOuter == 0:
                continue

            candidateIndicesB = [
                i for i, atomSymbol in enumerate(symbolsB)
                if atomSymbol == symbol
            ]
            candidateIndicesB.sort(key=lambda i: distB[i], reverse=True)
            selectedIndicesB = candidateIndicesB[:countOuter]

            if len(selectedIndicesB) < countOuter:
                return None

            # 若团簇2被选中交换的原子里出现 <= dmaxA*r/2,则本次交换失败并重选.
            threshold = radius / 2.0
            if any(distB[idx] <= threshold for idx in selectedIndicesB):
                return None

            outerIndicesA.sort(key=lambda i: distA[i], reverse=True)
            selectedIndicesB.sort(key=lambda i: distB[i], reverse=True)
            swapPairs.extend(zip(outerIndicesA, selectedIndicesB))

        trialA = copy.deepcopy(workA)
        trialB = copy.deepcopy(workB)
        trialPosA = trialA.get_positions().copy()
        trialPosB = trialB.get_positions().copy()

        for indexA, indexB in swapPairs:
            cache = trialPosA[indexA].copy()
            trialPosA[indexA] = trialPosB[indexB]
            trialPosB[indexB] = cache

        trialA.set_positions(trialPosA)
        trialB.set_positions(trialPosB)

        relaxedA = relax(trialA)
        relaxedB = relax(trialB)
        if not isinstance(relaxedA, Atoms) or not isinstance(relaxedB, Atoms):
            raise TypeError("relax must return Atoms when input is Atoms")

        energyA = getattr(relaxedA, 'energy', None)
        energyB = getattr(relaxedB, 'energy', None)
        if energyA is None or energyB is None:
            raise ValueError("relax result must contain energy for SphereCutSplice")

        return relaxedA if float(energyA) <= float(energyB) else relaxedB

    targetSize = len(pop1)
    maxRetries = targetSize * 200
    retryCount = 0

    while len(tempClusters) < targetSize:
        if retryCount >= maxRetries:
            raise RuntimeError("SphereCutSplice failed to generate enough offspring after retries")
        retryCount += 1

        selected1 = np.random.choice(len(pop1), size=2, replace=False)
        selected2 = np.random.choice(len(pop2), size=2, replace=False)

        winners: list[Atoms] = []
        roundFailed = False
        for localIndex in range(2):
            winner = spliceOne(pop1[int(selected1[localIndex])], pop2[int(selected2[localIndex])])
            if winner is None:
                roundFailed = True
                break
            winners.append(winner)

        if roundFailed:
            continue

        for winner in winners:
            if len(tempClusters) >= targetSize:
                break
            tempClusters.addCluster(winner, deepcopy=False)

    return tempClusters


def select(
    pop1: Population,
    pop2: Population,
    relax: Callable[[Population], Population | Atoms],
) -> Population:
    """对原种群和候选种群执行选择并更新自适应参数."""
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for selection")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for selection")

    tempClusters = Population(symbol=pop1.symbol)
    tempClusters._fp = pop1.fp
    tempClusters._p = pop1.p
    tempClusters._CRm = pop1.CRm

    # 先对两组候选做局域优化,再比较能量
    tempPop1 = relax(pop1)
    tempPop2 = relax(pop2)
    if not isinstance(tempPop1, Population) or not isinstance(tempPop2, Population):
        raise TypeError("relax must return Population when input is Population")

    crRec: list[float] = []
    deltaRec: list[float] = []
    ns1, ns2, nf1, nf2 = 0, 0, 0, 0

    # 选择环节要求两组种群都具备完整能量信息
    pop1Energies = tempPop1.energies
    pop2Energies = tempPop2.energies
    pop1EnergyValues: list[float] = []
    pop2EnergyValues: list[float] = []

    for i, energyValue in enumerate(pop1Energies):
        if energyValue is None:
            raise ValueError(f"pop1 is missing energy at index {i}")
        pop1EnergyValues.append(float(energyValue))
    for i, energyValue in enumerate(pop2Energies):
        if energyValue is None:
            raise ValueError(f"pop2 is missing energy at index {i}")
        pop2EnergyValues.append(float(energyValue))

    # 选择环节要求 pop2 含 mutTag 信息,否则无法更新 p 和 CRm
    for i in range(len(tempPop2)):
        if getattr(tempPop2[i], 'mut', None) is None:
            raise ValueError(f"pop2 is missing mutTag at index {i}")

    for i in range(len(tempPop1)):
        energy1 = pop1EnergyValues[i]
        energy2 = pop2EnergyValues[i]
        mutTag = getattr(tempPop2[i], 'mut', None)

        choosePop2 = energy2 < energy1

        if choosePop2:
            tempClusters.addCluster(tempPop2[i], deepcopy=True)
            currentCR = getattr(tempPop2[i], 'CR', None)
            if currentCR is not None:
                crRec.append(float(currentCR))
                deltaRec.append(float(energy1 - energy2))
            if mutTag == 0:
                ns1 += 1
            elif mutTag == 1:
                ns2 += 1
        else:
            tempClusters.addCluster(tempPop1[i], deepcopy=True)
            if mutTag == 0:
                nf1 += 1
            elif mutTag == 1:
                nf2 += 1

    # 更新 CRm:使用加权平均(仅统计成功个体)
    if crRec and sum(deltaRec) > 0:
        tempClusters._CRm = float(np.average(crRec, weights=deltaRec))

    # 更新 p:按成功/失败统计更新策略选择概率
    denominator = ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2)
    if denominator > 0:
        tempClusters._p = float((ns1 * (ns2 + nf2)) / denominator)

    return tempClusters
