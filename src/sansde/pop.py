"""pop.py
python 3.13.12
"""
import copy
from datetime import datetime
from pathlib import Path
from typing import Self, overload

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz
from ase.io.jsonio import read_json, write_json
from ase.io.trajectory import Trajectory


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
    candidate : list[Atoms]
        历史候选团簇列表, 按能量升序排列(最优在前), 长度由
        ``add_candidate`` 的 candidate_size 参数控制.
    symbol : str | None
        种群化学式. 仅允许在未设置时初始化一次.
    num : int
        单个团簇的原子数. 当 symbol 未设置时为 0.
    fp : float
        变异缩放因子采样分布的选择概率参数.
    p : float
        变异策略选择概率参数.
    crm : float
        交叉率中心参数.
    energies : list[float | None]
        按需从 ``cluster.info['energy']`` 读取的能量列表.
        若某团簇缺失该键, 对应位置为 None.
    forces : list[list[list[float]] | None]
        按需从 ``cluster.arrays['forces']`` 读取的力列表.
        若某团簇缺失该键, 对应位置为 None.
    PGs : list[str | None]
        按需从 ``cluster.info['PG']`` 读取的点群(熊夫利符号)列表.
        若某团簇缺失该键, 对应位置为 None.
    best_cluster : Atoms
        当前最低能量团簇. 空种群或全体缺失能量时抛 ValueError.

    Examples
    --------
    初始化种群:
    >>> pop = Population(symbol="Au13")

    添加团簇:
    >>> cluster = Atoms('Au13', positions=...)
    >>> pop.add_cluster(cluster)

    切片:
    >>> sub_pop = pop[::2]
    >>> isinstance(sub_pop, Population)
    True

    索引:
    >>> first_cluster = pop[0]

    迭代:
    >>> for cluster in pop:
    ...     print(cluster.get_chemical_formula())

    最优团簇:
    >>> best = pop.best_cluster
    >>> best_energy = best.info['energy']

    导出种群:
    >>> json_path = pop.export(name="Gen_1")

    导出最优团簇轨迹:
    >>> traj_path = pop.dump(name="Au13_best")

    导入种群:
    >>> pop_new = Population()
    >>> pop_new.import_population(filename=json_path)
    """
    def __init__(self, symbol: str | None = None):
        self._clusters: list[Atoms] = []
        self._symbol: str | None = symbol
        self._fp: float = 0.5
        self._p: float = 0.5
        self._crm: float = 0.5
        self._candidate: list[Atoms] = []

    @property
    def clusters(self) -> list[Atoms]:
        return self._clusters

    @property
    def candidate(self) -> list[Atoms]:
        return self._candidate

    @property
    def fp(self) -> float:
        return self._fp

    @property
    def p(self) -> float:
        return self._p

    @property
    def crm(self) -> float:
        return self._crm

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

    @overload
    def __getitem__(self, index: int) -> Atoms: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> Atoms | Self:
        if isinstance(index, slice):
            sub_pop = Population(symbol=self._symbol)
            sub_pop._fp = self._fp
            sub_pop._p = self._p
            sub_pop._crm = self._crm
            for cluster in self._clusters[index]:
                sub_pop.add_cluster(cluster, deepcopy=True)
            return sub_pop
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
        """按需从团簇 info 提取能量,缺失时返回 None."""
        energies: list[float | None] = []
        for cluster in self:
            energy_value = cluster.info.get('energy')
            if energy_value is not None:
                energy_value = float(energy_value)
            energies.append(energy_value)
        return energies

    @property
    def forces(self) -> list[list[list[float]] | None]:
        """按需从团簇 arrays 提取力,缺失时返回 None."""
        forces: list[list[list[float]] | None] = []
        for cluster in self:
            force_value = cluster.arrays.get('forces')
            if force_value is not None:
                force_value = np.asarray(force_value, dtype=float).tolist()
            forces.append(force_value)
        return forces

    @property
    def PGs(self) -> list[str | None]:
        """按需从团簇 info 提取点群(熊夫利符号), 缺失时返回 None."""
        point_groups: list[str | None] = []
        for cluster in self:
            point_groups.append(cluster.info.get('PG'))
        return point_groups

    @property
    def best_cluster(self) -> Atoms:
        """返回当前最优团簇, 即最低能量团簇."""
        if len(self._clusters) == 0:
            raise ValueError("Population is empty, best_cluster is unavailable")

        energies = self.energies
        best_index: int | None = None
        best_energy: float | None = None
        for i, energy_value in enumerate(energies):
            if energy_value is None:
                continue
            if best_energy is None or energy_value < best_energy:
                best_energy = energy_value
                best_index = i

        if best_index is None:
            raise ValueError("No valid energy in population, best_cluster is unavailable")

        return self._clusters[best_index]

    def add_cluster(self, cluster: Atoms, deepcopy: bool = True):
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

        cluster_to_add = copy.deepcopy(cluster) if deepcopy else cluster
        # 平移质心到原点
        cluster_to_add.set_center_of_mass([0.0, 0.0, 0.0])

        current_symbol = cluster_to_add.get_chemical_formula()
        
        if self._symbol is None:
            self.symbol = current_symbol 
        else:
            # 必须确保种群中所有团簇化学式相同
            if current_symbol != self._symbol:
                 raise ValueError(f"Cluster symbol {current_symbol} does not match population symbol {self._symbol}")

        self._clusters.append(cluster_to_add)

    def add_candidate(self, cluster: Atoms, candidate_size: int = 2) -> None:
        """把团簇加入候选列表首位, 并保持列表长度不超过 candidate_size.

        必须确保候选团簇与种群化学式相同.
        若种群 symbol 为空, 则在添加第一个候选团簇时自动设置.

        Parameters
        ----------
        cluster : Atoms
            待加入候选列表的团簇. 其化学式必须与种群设定的 symbol 一致.
        candidate_size : int, optional
            候选列表的最大长度. Default 2, 即只保留最优和次优团簇.
        """
        if not isinstance(cluster, Atoms):
            raise TypeError("Cluster must be an ase.Atoms object")
        if candidate_size < 0:
            raise ValueError("candidate_size must be >= 0")

        candidate_copy = cluster.copy()
        candidate_copy.calc = None
        current_symbol = candidate_copy.get_chemical_formula()
        if self._symbol is None:
            self.symbol = current_symbol
        elif current_symbol != self._symbol:
            raise ValueError(
                f"Cluster symbol {current_symbol} does not match "
                f"population symbol {self._symbol}"
            )

        self._candidate.insert(0, candidate_copy)
        del self._candidate[candidate_size:]

    def export(
        self,
        name: str | None = None,
        xyz: bool = False,
        dir: str | Path = "./Populations",
    ) -> Path:
        """导出种群数据到文件系统.

        在 dir 指定的目录(默认 './Populations/', 不存在则新建)下输出一个 JSON 文件.
        团簇以 ASE JSON 格式(Atoms 对象)存入 ``clusters`` 列表.
        当 xyz=True 时, 会同时输出同名的多帧 extxyz 文件,
        包含能量、力、pbc 以及 info 中的其他字段.

        Parameters
        ----------
        name : str, optional
            导出名称(不含扩展名).
            命名规则:
            1) 当 name 为 None 时, 默认使用种群 symbol; 若 symbol 为空则使用 "Population".
            2) JSON 文件名固定为 "{name}.json".
            3) 当 "{name}.json" 已存在时, 自动追加时间戳后缀,
               生成 "{name}_YYYYMMddhhmmss.json" 以避免覆盖.
            4) 当 xyz=True 时, xyz 文件名与最终 JSON 同名(不含扩展名).
        xyz : bool, optional
            是否同时导出包含所有团簇的多帧 extxyz 文件. Default False.
        dir : str | pathlib.Path, optional
            输出目录. Default "./Populations".

        Returns
        -------
        pathlib.Path
            导出 JSON 文件的完整绝对路径.
        """
        if name is None:
            name = self.symbol if self.symbol else "Population"

        base_dir = Path(dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Check for duplicate JSON filename
        summary_path = base_dir / f"{name}.json"
        if summary_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            name = f"{name}_{timestamp}"
            summary_path = base_dir / f"{name}.json"
            print(f'Duplicate file found. New export name: {name}!')

        # Optional extxyz snapshot: 只用 info/arrays 里已有的 energy/forces,
        # 不读取或重算 calculator; 缺失的字段留空.
        if xyz:
            xyz_path = base_dir / f"{name}.xyz"
            try:
                with xyz_path.open('w', encoding='utf-8') as f:
                    write_extxyz(f, self._clusters, write_results=False)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Failed to write extxyz file {xyz_path}: {e}")
        
        min_e = None
        min_idx = None
        
        valid_energies = [(i, e) for i, e in enumerate(self.energies) if e is not None]
        if valid_energies:
            min_idx, min_e = min(valid_energies, key=lambda x: x[1])
            if isinstance(min_e, np.generic):
                min_e = min_e.item()

        summary_data = {
            "date": datetime.now().strftime('%Y%m%d%H%M%S'),
            "symbol": self.symbol,
            "size": len(self),
            "fp": self.fp,
            "p": self.p,
            "CRm": self.crm,
            "min_energy": min_e,
            "min_energy_index": min_idx,
            # 每个团簇直接以 ASE JSON 格式(Atoms)存入列表.
            "clusters": self._clusters,
            "candidate": self._candidate,
        }

        with summary_path.open('w', encoding='utf-8') as f:
            write_json(f, summary_data)

        return summary_path.resolve()

    def dump(self, name: str | None = None) -> Path:
        """导出当前最优团簇到 ASE 轨迹文件(.traj).

        使用 ASE Trajectory 接口写入 ./Trajectories/ 下的轨迹文件.
        若文件已存在则追加写入新帧.
        能量与力仅从团簇的 ASE 原生容器读取:
        - 若存在 cluster.info['energy'], 则写入 energy.
        - 若存在 cluster.arrays['forces'], 则写入 forces.
        - 不会触发额外计算; 缺失字段则不写入.

        Parameters
        ----------
        name : str, optional
            轨迹文件名(不含扩展名). 默认使用 "{symbol}_best".

        Returns
        -------
        pathlib.Path
            导出轨迹文件的完整绝对路径.
        """
        best_cluster = self.best_cluster

        base_dir = Path("./Trajectories")
        base_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            base_name = f"{self.symbol}_best" if self.symbol else "Population_best"
        else:
            base_name = name

        traj_path = base_dir / f"{base_name}.traj"
        mode = 'a' if traj_path.exists() else 'w'

        export_cluster = best_cluster.copy()
        write_kwargs: dict[str, float | np.ndarray] = {}

        energy_value = best_cluster.info.get('energy')
        if energy_value is not None:
            write_kwargs['energy'] = float(energy_value)

        force_value = best_cluster.arrays.get('forces')
        if force_value is not None:
            write_kwargs['forces'] = np.array(force_value, dtype=float)

        with Trajectory(traj_path, mode) as traj:
            traj.write(export_cluster, **write_kwargs)

        return traj_path.resolve()

    def dump_xyz(
        self,
        name: str | None = None,
        candidate: bool = False,
        dir: str | Path = "./Trajectories",
    ) -> Path:
        """导出团簇到 extxyz 文件(.xyz).

        ``candidate=False`` 时只写当前最优团簇, 文件已存在则追加, 用于记录
        最优团簇演化轨迹. ``candidate=True`` 时按顺序写出整个 ``candidate``
        列表, 并覆盖同名文件.

        只使用 ``info``/``arrays`` 中已有的能量和力, 不读取或重算计算器.

        Parameters
        ----------
        name : str, optional
            文件名(不含扩展名). 默认使用 "{symbol}_best".
        candidate : bool, optional
            是否导出整个候选列表而不是当前最优团簇. Default False.
        dir : str | pathlib.Path, optional
            输出目录. Default "./Trajectories".

        Returns
        -------
        pathlib.Path
            导出 extxyz 文件的完整绝对路径.
        """
        base_dir = Path(dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            base_name = f"{self.symbol}_best" if self.symbol else "Population_best"
        else:
            base_name = name

        xyz_path = base_dir / f"{base_name}.xyz"

        if candidate:
            mode = 'w'
            clusters = self._candidate
        else:
            mode = 'a'
            clusters = [self.best_cluster]

        with xyz_path.open(mode, encoding='utf-8') as f:
            write_extxyz(f, clusters, write_results=False)

        return xyz_path.resolve()

    def import_population(self, filename: str | Path, confirm: bool = False):
        """从JSON文件导入种群数据.

        直接读取指定的JSON文件.
        ``clusters`` 中的 ASE JSON (Atoms) 直接重建为 Atoms 对象;
        同时兼容旧版手写 dict 格式.
        如果种群非空, 需要设置 confirm=True 才能覆盖现有数据.

        Parameters
        ----------
        filename : str | pathlib.Path
            JSON文件的完整路径.
        confirm : bool, optional
            是否确认覆盖当前种群数据. Default False.
        """
        json_path = Path(filename)
        if not json_path.is_file():
             print(f"JSON file {json_path} not found.")
             return

        if len(self) > 0:
            if not confirm:
                print("Population is not empty. Set confirm=True to overwrite.")
                return 
            else:
                self._clusters = []
                self._symbol = None
                self._candidate = []

        try:
            with json_path.open('r', encoding='utf-8') as f:
                data = read_json(f)
        except (OSError, ValueError) as e:
            print(f"Failed to read JSON file: {e}")
            return
            
        # Set population attributes from JSON
        if self._symbol is None:
             self._symbol = data.get('symbol')
        
        if self._symbol != data.get('symbol'):
             print(f"Warning: JSON symbol ({data.get('symbol')}) does not match population symbol ({self._symbol})")
             # Proceeding anyway, but this is a potential issue if user intends strict type checking

        # Load adaptive parameters if present; keep current values when missing.
        fp_value = data.get('fp')
        p_value = data.get('p')
        crm_value = data.get('CRm')
        if fp_value is not None:
            self._fp = float(fp_value)
        if p_value is not None:
            self._p = float(p_value)
        if crm_value is not None:
            self._crm = float(crm_value)

        # Import any extra attributes beyond the preset ones
        preset_keys = {'symbol', 'fp', 'p', 'CRm', 'clusters', 'candidate'}
        for key, value in data.items():
            if key not in preset_keys:
                setattr(self, key, value)

        candidate_data = data.get('candidate', [])
        self._candidate = [
            atoms.copy() for atoms in candidate_data if isinstance(atoms, Atoms)
        ]

        clusters_data = data.get('clusters', [])
        
        for c_data in clusters_data:
            if isinstance(c_data, Atoms):
                # ASE JSON 快照: 能量/点群在 info, 力在 arrays 中, 直接收编.
                self._clusters.append(c_data)
                continue

            try:
                # 兼容旧版手写 dict 格式.
                # Reconstruct Atoms object
                symbol = c_data.get('symbol')
                positions = c_data.get('positions')
                
                # Ensure positions is a list or numpy array
                if positions is None:
                    print(f"Skipping cluster {c_data.get('name')} due to missing positions.")
                    continue
                    
                cluster = Atoms(symbol, positions=positions)
                
                # Add to population
                # Note: self.add_cluster appends None to energies/forces, so we overwrite them below
                self._clusters.append(cluster)
                
                # Set energy/forces/PG into ASE native containers
                energy = c_data.get('energy')
                forces = c_data.get('forces')
                
                if energy is not None:
                     cluster.info['energy'] = float(energy)
                if forces is not None:
                     cluster.arrays['forces'] = np.asarray(forces, dtype=float)
                point_group = c_data.get('PG')
                if point_group is not None:
                     cluster.info['PG'] = point_group
                
            except Exception as e:  # noqa: BLE001
                print(f"Error loading cluster from JSON: {e}")
