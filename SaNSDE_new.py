"""SaNSDE_new.py
python 3.13.12
"""
import os
import json
import copy
import functools
import numpy as np
from typing import Any
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS, LBFGS, GPMin, FIRE
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
        """按需从团簇对象属性提取能量，缺失时返回 None。"""
        energies: list[float | None] = []
        for cluster in self:
            energyValue = getattr(cluster, 'energy', None)
            if energyValue is not None:
                energyValue = float(energyValue)
            energies.append(energyValue)
        return energies

    @property
    def forces(self) -> list[list[list[float]] | None]:
        """按需从团簇对象属性提取力，缺失时返回 None。"""
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

    def export(self, name: str | None = None, format: str = 'xyz') -> str:
        """导出种群数据到文件系统.

        在当前目录'./Populations/'(如不存在则新建)下新建一个文件夹, 名称为name(缺省时为种群名称).
        在文件夹中分别输出所有团簇结构文件(symbol_i.format).
        文件夹还包含一个说明文件(name.json), 包含导出日期, symbol, size, min_energy等信息,
        以及详细的团簇信息(Atoms name, symbol, energies, positions, forces).

        Parameters
        ----------
        name : str, optional
            导出文件夹的名称. 默认使用种群的 symbol.
        format : str, optional
            导出文件格式 (e.g., 'xyz', 'vasp', 'cif', 'data'). Default 'xyz'.

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

        # Check for existing directory
        if os.path.exists(target_dir):
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            name = f"{name}_{timestamp}"
            target_dir = os.path.join(base_dir, name)
            print(f'Duplicate file found. New export directory: {name}!')

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Map format to extension and ASE format string if necessary
        ext = format
        ase_format = format
        if format == 'data':
            ase_format = 'lammps-data'

        # Export clusters
        for i, cluster in enumerate(self._clusters):
            # filename: symbol_i.format
            fname = f"{self.symbol}_{i}.{ext}"
            fpath = os.path.join(target_dir, fname)
            
            # Write file
            if self.energies[i] is not None:
                setattr(cluster, 'energy', self.energies[i])
            
            # Check cell for formats that require it
            original_cell = cluster.get_cell()
            original_pbc = cluster.get_pbc()
            temp_cell_set = False

            if ext in ['vasp', 'cif', 'res', 'castep'] and np.allclose(original_cell.lengths(), 0):
                 # Set a large dummy cell for cluster output
                 cluster.set_cell([30.0, 30.0, 30.0])
                 cluster.center()
                 cluster.set_pbc(True)
                 temp_cell_set = True

            try:
                write(fpath, cluster, format=ase_format)
            except Exception as e:
                print(f"Warning: Failed to write {fname}: {e}")
            
            # Restore original cell if we changed it
            if temp_cell_set:
                 cluster.set_cell(original_cell)
                 cluster.set_pbc(original_pbc)

        # Summary file (JSON)
        summary_path = os.path.join(target_dir, f"{name}.json")
        
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
                  optimizer: Any = None, 
                  fmax: float = 0.05, 
                  steps: int = 100) -> Population | Atoms:
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
    optimizer : str or Optimizer, optional
        优化器类型. 支持字符串 'BFGS', 'LBFGS', 'CG', 'FIRE' 或 ASE优化器类. Default 'BFGS'.
    fmax : float, optional
        收敛判据, 最大力阈值 (eV/Ang). Default 0.05.
    steps : int, optional
        最大优化步数. Default 100.
        
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
        newClusters = copy.deepcopy(pop)
        isSingle = False
    elif isinstance(pop, Atoms):
        newClusters = Population()
        newClusters.addCluster(pop)
        isSingle = True
    else:
        raise TypeError("pop must be a Population or an ase.Atoms object")

    for atoms in newClusters:
        # cell 自动添加到团簇对象上
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        atoms.center()

        # 计算器
        if potential is not None:
            atoms.calc = potential

        # Optimizer selection
        opt_class = BFGS 
        if optimizer == 'CG':
             # Dynamic import to bypass static analysis issues
             import importlib
             try:
                 ase_opt = importlib.import_module('ase.optimize')
                 opt_class = getattr(ase_opt, 'ConjugateGradient')
             except (ImportError, AttributeError):
                 print("ConjugateGradient not found, using BFGS.")
                 opt_class = BFGS
        elif optimizer == 'LBFGS':
             opt_class = LBFGS
        elif optimizer == 'FIRE':
             opt_class = FIRE
        elif optimizer is not None and not isinstance(optimizer, str):
             opt_class = optimizer

        if atoms.calc is None:
             continue

        try:
            dyn = opt_class(atoms, logfile=None) # type: ignore 
            dyn.run(fmax=fmax, steps=steps)
        except Exception as e:
            print(f"Optimization failed: {e}")

        # 只同步到团簇对象属性，Population.energies/forces 按需读取这里。
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
        atoms.set_pbc(False)
        atoms.set_cell([0.0, 0.0, 0.0]) # Effectively clear cell dimensions

    if isSingle:
        return newClusters.clusters[0]
    return newClusters


def randomCluster(pop: Population, 
                  a: float, 
                  delta: float, 
                  cell: list[float] | np.ndarray, 
                  potential: Calculator | None, 
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
    optimizedCluster = localOptimize(cluster, cell=cell, potential=potential, optimizer=optimizer, fmax=fmax, steps=steps)
    if not isinstance(optimizedCluster, Atoms):
        raise TypeError("localOptimize should return Atoms when input is Atoms")
    
    # Add to pop
    pop.addCluster(optimizedCluster)
    
    # Update energy and forces
    energyValue = getattr(optimizedCluster, 'energy', None)
    forceValue = getattr(optimizedCluster, 'forces', None)
    setattr(pop.clusters[len(pop) - 1], 'energy', energyValue)
    setattr(pop.clusters[len(pop) - 1], 'forces', forceValue)


def mutate(pop: Population) -> Population:
    """对种群执行突变，返回突变后的暂定种群。"""
    if len(pop) < 4:
        raise ValueError("Mutation requires at least 4 clusters in population")

    tempClusters = copy.deepcopy(pop)

    # 选取当前最优个体索引（最低能量）
    energyPairs = [(i, e) for i, e in enumerate(pop.energies) if e is not None]
    if not energyPairs:
        raise ValueError("Population energies are required before mutation")
    indexBest = min(energyPairs, key=lambda item: item[1])[0]

    for i in range(len(pop)):
        # 采样缩放因子 F：按 fp 在高斯/柯西分布之间切换
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
    """对原种群与突变种群执行交叉，返回交叉后种群。"""
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for crossover")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for crossover")

    tempClusters = Population(symbol=pop1.symbol)
    tempClusters._fp = pop1.fp
    tempClusters._p = pop1.p
    tempClusters._CRm = pop1.CRm

    for i in range(len(pop1)):
        # 采样当前个体交叉率，并裁剪到 [0, 1]
        crossRate = float(np.random.normal(pop1.CRm, 0.1))
        crossRate = float(np.clip(crossRate, 0.0, 1.0))

        # 至少保留一个原子来自突变种群
        jrand = np.random.randint(0, pop1.num)

        basePositions = pop1[i].get_positions().copy()
        mutPositions = pop2[i].get_positions()
        trialPositions = basePositions.copy()

        # 逐原子交叉：按 CR 或 jrand 选择保留突变位姿
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


def select(
    pop1: Population,
    pop2: Population,
) -> Population:
    """对原种群和候选种群执行选择并更新自适应参数。"""
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for selection")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for selection")

    tempClusters = Population(symbol=pop1.symbol)
    tempClusters._fp = pop1.fp
    tempClusters._p = pop1.p
    tempClusters._CRm = pop1.CRm

    # 局域优化偏函数：固定常用参数，便于在选择环节统一调用
    relax = functools.partial(
        localOptimize,
        cell=[15, 15, 15],
        potential=LennardJones(),
        optimizer=BFGS,
        fmax=0.05,
        steps=100,
    )

    # 先对两组候选做局域优化，再比较能量
    tempPop1 = relax(pop1)
    tempPop2 = relax(pop2)

    if not isinstance(tempPop1, Population) or not isinstance(tempPop2, Population):
        raise TypeError("relax must return Population when input is Population")

    crRec: list[float] = []
    deltaRec: list[float] = []
    ns1, ns2, nf1, nf2 = 0, 0, 0, 0

    for i in range(len(tempPop1)):
        energy1 = tempPop1.energies[i]
        energy2 = tempPop2.energies[i]
        mutTag = getattr(tempPop2[i], 'mut', None)

        # 缺失能量时，优先保留有能量的一方
        choosePop2 = False
        if energy1 is None and energy2 is not None:
            choosePop2 = True
        elif energy1 is not None and energy2 is not None and energy2 < energy1:
            choosePop2 = True

        if choosePop2:
            tempClusters.addCluster(tempPop2[i], deepcopy=True)
            currentCR = getattr(tempPop2[i], 'CR', None)
            if currentCR is not None and energy1 is not None and energy2 is not None:
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

    # 更新 CRm：使用加权平均（仅统计成功个体）
    if crRec and sum(deltaRec) > 0:
        tempClusters._CRm = float(np.average(crRec, weights=deltaRec))

    # 更新 p：按成功/失败统计更新策略选择概率
    denominator = ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2)
    if denominator > 0:
        tempClusters._p = float((ns1 * (ns2 + nf2)) / denominator)

    return tempClusters
