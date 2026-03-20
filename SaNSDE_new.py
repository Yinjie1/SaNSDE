"""SaNSDE_new.py
python 3.13.12
"""
import os
import json
import numpy as np
from typing import Any
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, GPMin, FIRE
from ase.calculators.calculator import Calculator
from datetime import datetime


class Population:
    """种群对象
    
    包含一组Atoms团簇对象, 管理其能量和力. 
    种群中所有团簇必须具有相同的化学式(symbol).

    Parameters
    ----------
    symbol : str | None = None
        团簇的化学式 (e.g., "Au13"). 如果未指定, 将在添加第一个团簇时自动设定.
        
    Examples
    --------
    初始化种群:
    >>> pop = Population(symbol="Au13")

    添加团簇:
    >>> cluster = Atoms('Au13', positions=...)
    >>> pop.addCluster(cluster)

    导出种群:
    >>> pop.export(name="Gen_1")

    导入种群:
    >>> pop_new = Population()
    >>> pop_new.import_population(name="Gen_1")
    """
    def __init__(self, symbol: str | None = None):
        self._clusters: list[Atoms] = []
        self._symbol: str | None = symbol
        self.energies: list[float | None] = []
        self.forces: list[np.ndarray | None] = []

    @property
    def clusters(self) -> list[Atoms]:
        return self._clusters

    @property
    def symbol(self) -> str | None:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        if self._symbol is not None:
            raise ValueError("symbol已经被设置!")
        self._symbol = value

    @property
    def size(self) -> int:
        return len(self._clusters)

    @property
    def num(self) -> int:
        if self._symbol is None:
            return 0
        # 根据化学式symbol推出一个团簇的原子数(int)
        temp = Atoms(self._symbol)
        return len(temp)

    def addCluster(self, cluster: Atoms):
        """导入一个团簇对象(Atoms类型)到_clusters列表末尾.

        必须确保种群中所有团簇化学式相同(symbol), 即每种元素的原子数分别相同.
        团簇类型(symbol)可作为种群的一个属性, 可以在实例化种群时添加, 
        如果为空, 则在种群添加了第一个团簇时, 设为该团簇的化学式.

        Parameters
        ----------
        cluster : Atoms
            ASE Atoms 对象, 代表一个团簇.
            其化学式必须与种群设定的 symbol 一致.
        """
        if not isinstance(cluster, Atoms):
            raise TypeError("Cluster must be an ase.Atoms object")

        current_symbol = cluster.get_chemical_formula()
        
        if self._symbol is None:
            self.symbol = current_symbol 
        else:
            # 必须确保种群中所有团簇化学式相同
            # Simple check based on atom count first
            if len(cluster) != self.num:
                 raise ValueError(f"Cluster size {len(cluster)} does not match population size {self.num}")
            
        self._clusters.append(cluster)
        self.energies.append(None)
        self.forces.append(None)

    def export(self, name: str | None = None, format: str = 'xyz'):
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
                cluster.info['energy'] = self.energies[i]
            
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

        clusters_info = []
        for i, cluster in enumerate(self._clusters):
            # 准备单个团簇的信息
            force_data = self.forces[i]
            if force_data is not None and isinstance(force_data, np.ndarray):
                force_data = force_data.tolist()
            elif force_data is None:
                force_data = None

            cluster_info = {
                "name": f"{self.symbol}_{i}",
                "symbol": cluster.get_chemical_formula(),
                "energy": self.energies[i],
                "positions": cluster.get_positions().tolist(),
                "forces": force_data
            }
            clusters_info.append(cluster_info)

        summary_data = {
            "date": datetime.now().strftime('%Y%m%d%H%M%S'),
            "symbol": self.symbol,
            "size": self.size,
            "min_energy": min_e,
            "min_energy_index": min_idx,
            "clusters": clusters_info
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)

    def import_population(self, name: str, confirm: bool = False):
        """从JSON文件导入种群数据.

        读取'./Populations/name/name.json'文件.
        直接使用JSON文件中的数据重建Atoms对象, 能量和力.
        如果种群非空, 需要设置 confirm=True 才能覆盖现有数据.

        Parameters
        ----------
        name : str
            种群文件夹名称 (位于 ./Populations/ 下).
        confirm : bool, optional
            是否确认覆盖当前种群数据. Default False.
        """
        target_dir = os.path.join('./Populations', name)
        json_path = os.path.join(target_dir, f"{name}.json")
        
        if not os.path.exists(json_path):
             print(f"JSON file {json_path} not found.")
             return

        if self.size > 0:
            if not confirm:
                print("Population is not empty. Set confirm=True to overwrite.")
                return 
            else:
                self._clusters = []
                self.energies = []
                self.forces = []
                self._symbol = None

        try:
            with open(json_path, 'r') as f:
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
                
                # Set energy and forces directly
                energy = c_data.get('energy')
                forces = c_data.get('forces')
                
                if forces is not None:
                    forces = np.array(forces)
                
                self.energies.append(energy)
                self.forces.append(forces)
                
                # Optional: Attach info to Atoms object for convenience
                if energy is not None:
                     cluster.info['energy'] = energy
                
            except Exception as e:
                print(f"Error loading cluster from JSON: {e}")



def localOptimize(pop: Population | Atoms | list[Atoms], 
                  cell: list[float] | np.ndarray = [15, 15, 15], 
                  potential: Calculator | None = None, 
                  optimizer: Any = None, 
                  fmax: float = 0.05, 
                  steps: int = 100):
    """进行局域优化(结构弛豫)
    
    对种群中的所有团簇或单个团簇进行结构优化.
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
        
    Examples
    --------
    >>> localOptimize(pop, potential=calc, optimizer='LBFGS')
    """
    targets = []
    if isinstance(pop, Population):
        targets = pop.clusters
    elif isinstance(pop, Atoms):
        targets = [pop]
    elif isinstance(pop, list):
        targets = pop
    else:
        # Fallback or error, assume it's iterable of Atoms
        targets = pop

    for atoms in targets:
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

        # 计算后清除 cell
        # del atoms.cell # cell attribute deletion is not supported
        atoms.set_pbc(False)
        atoms.set_cell([0.0, 0.0, 0.0]) # Effectively clear cell dimensions


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
    localOptimize(cluster, cell=cell, potential=potential, optimizer=optimizer, fmax=fmax, steps=steps)
    
    # Add to pop
    pop.addCluster(cluster)
    
    # Update energy and forces
    if cluster.calc:
        e = cluster.get_potential_energy()
        f = cluster.get_forces()
        pop.energies[pop.size - 1] = e
        pop.forces[pop.size - 1] = f
