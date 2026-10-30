"""operators.py

Population-level operators for SaNSDE: cluster generation, local relaxation,
mutation, crossover, and selection. The operators act on ``Population``
instances while keeping the population container free of algorithmic code.
"""
import contextlib
import functools
import signal
from collections.abc import Callable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from sansde.pop import Population

with contextlib.suppress(Exception):
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

MIN_ATOM_DISTANCE = 0.5

# 作用于团簇

def compute_point_group(
    cluster: Atoms,
    tolerance: float = 0.3,
    timeout: float = 5.0,
) -> str | None:
    """计算团簇点群(熊夫利符号), 失败或超时返回 None.

    pymatgen 的 ``PointGroupAnalyzer`` 在近简并结构上可能长时间生成对称操作,
    这里用 ``SIGALRM`` 限制单次计算时长. 首次超时后用 2 倍 tolerance 重试;
    仍然超时则放弃计算、返回 None 并发出警告.

    Parameters
    ----------
    cluster : Atoms
        待计算点群的团簇.
    tolerance : float, optional
        传给 pymatgen ``PointGroupAnalyzer`` 的坐标容差. Default 0.3.
        首次超时后会用 ``2 * tolerance`` 重试一次.
    timeout : float, optional
        每次点群计算允许的最长时间(秒). Default 5.0.

    Returns
    -------
    str | None
        点群熊夫利符号; pymatgen 不可用、计算失败或两次尝试均超时时返回 None.
    """
    try:
        molecule = Molecule(cluster.get_chemical_symbols(), cluster.get_positions())
    except Exception:  # noqa: BLE001
        return None

    def handle_timeout(signum: int, frame: object) -> None:
        raise TimeoutError

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    try:
        for attempt, current_tolerance in enumerate((tolerance, tolerance * 2), 1):
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return str(
                    PointGroupAnalyzer(molecule, tolerance=current_tolerance)
                    .get_pointgroup()
                    .sch_symbol
                )
            except TimeoutError:
                if attempt == 2:
                    print(
                        f"Warning: PointGroupAnalyzer timed out after {timeout}s "
                        f"with tolerance {current_tolerance}; "
                        "point group set to None."
                    )
                    return None
                # 第一次超时: 用 2 倍 tolerance 重试.
            except Exception:  # noqa: BLE001
                return None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)


def min_pair_distance(positions: np.ndarray) -> float:
    """计算一组坐标的最小原子间距.

    Parameters
    ----------
    positions : numpy.ndarray
        原子坐标, 形状为 ``(natoms, 3)``.

    Returns
    -------
    float
        最小两两间距. 原子数少于 2 时返回 ``inf``;
        坐标含非有限值时返回 ``-inf``, 以便调用方判为非法几何.
    """
    coords = np.asarray(positions, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("positions must have shape (natoms, 3)")
    if len(coords) < 2:
        return float("inf")
    if not np.isfinite(coords).all():
        return -float("inf")
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def similarity_function(atoms_a: Atoms, atoms_b: Atoms) -> float:
    """计算两个团簇的相似性函数 S = 1 / (1 + q).

    自动从 ``Atoms`` 对象读取坐标并构造升序原子间距列表, 不依赖预先存储的
    距离列表. 公式见 docs/similarity_function.md.

    Parameters
    ----------
    atoms_a : Atoms
        第一个团簇.
    atoms_b : Atoms
        第二个团簇.

    Returns
    -------
    float
        相似性函数 S = 1 / (1 + q), 取值 (0, 1].
        原子数不一致, 原子数少于 2, 坐标含非有限值, 或最短间距为 0 时
        返回 0.0.
    """
    if len(atoms_a) != len(atoms_b) or len(atoms_a) < 2:
        return 0.0

    positions_a = np.asarray(atoms_a.get_positions(), dtype=float)
    positions_b = np.asarray(atoms_b.get_positions(), dtype=float)
    if positions_a.shape != positions_b.shape:
        return 0.0
    if not np.isfinite(positions_a).all() or not np.isfinite(positions_b).all():
        return 0.0

    pair_vectors_a = positions_a[:, None, :] - positions_a[None, :, :]
    pair_vectors_b = positions_b[:, None, :] - positions_b[None, :, :]
    upper_indices = np.triu_indices(len(positions_a), k=1)
    distances_a = np.sort(np.linalg.norm(pair_vectors_a, axis=-1)[upper_indices])
    distances_b = np.sort(np.linalg.norm(pair_vectors_b, axis=-1)[upper_indices])

    if distances_a.size == 0:
        return 0.0
    if distances_a[0] <= 0.0 or distances_b[0] <= 0.0:
        return 0.0

    normalized_a = distances_a / distances_a[0]
    normalized_b = distances_b / distances_b[0]
    q_value = float(np.sqrt(np.mean((normalized_a - normalized_b) ** 2)))
    return 1.0 / (1.0 + q_value)


# 作用于种群

def local_optimize(pop: Population | Atoms, 
                   cell: list[float] | np.ndarray | None = None, 
                   potential: Calculator | None = None,
                   optimizer: type[Optimizer] = BFGS,
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
    cell : list or np.ndarray or None, optional
        用于计算的模拟盒子大小, default None.
    potential : Calculator, optional
        ASE计算器对象(e.g., LJ, EMT, DeepMD等).
        不允许为 None; 若为 None 直接报错.
    optimizer : type[Optimizer], optional
        ASE 优化器类(Optimizer 子类), 例如 BFGS/LBFGS/FIRE.
        Default BFGS.
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
    >>> popNew = local_optimize(pop, potential=calc, optimizer='LBFGS')
    """
    if potential is None:
        raise ValueError("potential must be an ASE calculator object, None is not allowed")

    if isinstance(pop, Population):
        new_clusters = pop.copy()
        is_single = False
    elif isinstance(pop, Atoms):
        new_clusters = Population()
        new_clusters.add_cluster(pop)
        is_single = True
    else:
        raise TypeError("pop must be a Population or an ase.Atoms object")

    for atoms in new_clusters:
        # cell 自动添加到团簇对象上
        if cell:
            atoms.set_cell(cell)
            atoms.set_pbc(False)

        # 重合几何会让势函数返回 NaN, 直接判为无效个体, 不调用计算器.
        if min_pair_distance(atoms.get_positions()) < MIN_ATOM_DISTANCE:
            atoms.info['energy'] = float('inf')
            atoms.arrays['forces'] = np.zeros((len(atoms), 3))
            atoms.set_cell([0.0, 0.0, 0.0])
            atoms.calc = None
            continue

        # 计算器
        atoms.calc = potential

        # Optimizer selection: 统一使用 ASE Optimizer 子类
        opt_class = optimizer
        if not isinstance(opt_class, type) or not issubclass(opt_class, Optimizer):
            raise TypeError("optimizer must be an ASE Optimizer subclass")

        try:
            dyn = opt_class(atoms, logfile=None) # type: ignore
            dyn.run(fmax=fmax, steps=steps)
        except Exception as e:  # noqa: BLE001
            print(f"Optimization failed: {e}")

        # 只同步到 atoms.info / atoms.arrays, Population.energies/forces 按需读取这里.
        # 仅复用优化器最后一步的 calculator.results, 不再触发额外能量/力计算.
        energy_value = None
        force_value = None
        if atoms.calc is not None:
            calc_results = getattr(atoms.calc, 'results', None)
            if isinstance(calc_results, dict):
                energy_value = calc_results.get('energy')
                force_value = calc_results.get('forces')

        if energy_value is None or force_value is None:
            # 优化失败(含中途出现 NaN 坐标): 记为无效个体, 不中断整个流程.
            atoms.info['energy'] = float('inf')
            atoms.arrays['forces'] = np.zeros((len(atoms), 3))
            positions = np.asarray(atoms.get_positions(), dtype=float)
            if not np.isfinite(positions).all():
                atoms.set_positions(
                    np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
                )
        else:
            atoms.info['energy'] = float(energy_value)
            atoms.arrays['forces'] = np.asarray(force_value, dtype=float)

        # 计算后清除 cell
        # del atoms.cell # cell attribute deletion is not supported
        atoms.set_cell([0.0, 0.0, 0.0]) # Effectively clear cell dimensions
        atoms.calc = None

    if is_single:
        return new_clusters.clusters[0]
    return new_clusters


# 局域优化偏函数:固定常用参数,供 select 等流程直接复用
relax = functools.partial(
    local_optimize,
    cell=None,
    potential=LennardJones(),
    optimizer=BFGS,
    fmax=0.05,
    steps=100,
)


def random_cluster(pop: Population, 
                   a: float, 
                   delta: float, 
                   relax: Callable[[Population | Atoms], Population | Atoms],
                   is_2d: bool = False) -> None:
    """随机生成一个团簇并添加进种群.
    
    使用生长法随机生成团簇初始坐标, 并进行局域优化后加入种群.
    生成过程: 
    1. 第一个原子置于原点.
    2. 后续原子在已生成原子的附近随机生成 (键长 bond_length, 方向随机).
    3. 检查新原子与已有原子的距离, 若过近(<0.75a)则重新生成.

    Parameters
    ----------
    pop : Population
        目标种群对象.
    a : float
        晶格常数或第一近邻距离 (Angstrom). 用于控制生成团簇的致密程度.
    delta : float
        键长扰动因子 (0~1). bond_length = (1 + delta * uniform(-0.25, 0.75)) * a.
    relax : Callable
        局域优化函数. 可传入偏函数 relax() 或函数对象 local_optimize().
    is_2d : bool, optional
        Whether to generate a 2D cluster in the xy plane. Default False.
        When True, atoms are generated inside a disk with radius
        R = a * (0.5 + sqrt((sqrt(3) / (2 * pi)) * n_atoms)), and z is fixed to 0.
        
    Examples
    --------
    >>> random_cluster(pop, a=2.5, delta=0.5, relax=relax)
    """
    n_atoms = pop.num
    if n_atoms == 0:
        if pop.symbol:
             n_atoms = len(Atoms(pop.symbol))
        else:
             raise ValueError("Population symbol not set, cannot determine cluster size.")

    # Radius limit for random growth.
    if is_2d:
        r0 = a * (0.5 + np.sqrt((np.sqrt(3) / (2 * np.pi)) * n_atoms))
    else:
        r0 = 2 * a * (0.5 + (3 * n_atoms / (4 * np.pi * np.sqrt(2)))**(1.0/3.0))

    # Initialize positions
    dummy = Atoms(pop.symbol)
    symbols = dummy.get_chemical_symbols()
    
    positions = np.zeros((n_atoms, 3))
    positions[0] = [0.0, 0.0, 0.0]

    for i in range(1, n_atoms):
        while True:
            idx_bond_atom = np.random.randint(0, i)
            
            # bond_length = (1+delta*np.random.uniform(-0.25,0.75))*a
            bond_length = (1 + delta * np.random.uniform(-0.25, 0.75)) * a
            
            if is_2d:
                # 2D mode: random direction in xy plane, z fixed at 0.
                bond_phi = np.random.rand() * 2 * np.pi
                x = positions[idx_bond_atom][0] + bond_length * np.cos(bond_phi)
                y = positions[idx_bond_atom][1] + bond_length * np.sin(bond_phi)
                z = 0.0
            else:
                # 3D mode: random spherical direction.
                bond_theta = np.random.rand() * np.pi
                bond_phi = np.random.rand() * 2 * np.pi
                x = positions[idx_bond_atom][0] + bond_length * np.cos(bond_theta) * np.cos(bond_phi)
                y = positions[idx_bond_atom][1] + bond_length * np.cos(bond_theta) * np.sin(bond_phi)
                z = positions[idx_bond_atom][2] + bond_length * np.sin(bond_theta)
            
            trial_pos = np.array([x, y, z])

            # 新原子必须位于半径 r0 球/圆内.
            if np.linalg.norm(trial_pos) > r0:
                continue
            
            # if 该原子和其他原子距离 < 0.75*a: 重新生成
            distances = np.linalg.norm(positions[:i] - trial_pos, axis=1)
            if np.any(distances < 0.75 * a):
                continue
            
            positions[i] = trial_pos
            break

    # Create cluster
    cluster = Atoms(symbols=symbols, positions=positions)
    
    # local_optimize
    optimized_cluster = relax(cluster)
    if not isinstance(optimized_cluster, Atoms):
        raise TypeError("local_optimize should return Atoms when input is Atoms")
    
    # Add to pop
    pop.add_cluster(optimized_cluster)
    
    # Update energy and forces
    energy_value = optimized_cluster.info.get('energy')
    force_value = optimized_cluster.arrays.get('forces')
    added_cluster = pop.clusters[len(pop) - 1]
    added_cluster.info['energy'] = energy_value
    if force_value is None:
        added_cluster.arrays.pop('forces', None)
    else:
        added_cluster.arrays['forces'] = np.asarray(force_value, dtype=float)


def is_connected_graph(cluster: Atoms, cutoff: float) -> bool:
    """判断单个团簇是否为连通图.

    Parameters
    ----------
    cluster : Atoms
        待检查团簇.
    cutoff : float
        连边阈值(Ang). 两原子距离 <= cutoff 视为有边.

    Returns
    -------
    bool
        团簇连通返回 True, 否则返回 False.
    """
    # Step 1: 参数检查, cutoff 必填且必须为正数.
    if cutoff is None:
        raise ValueError("cutoff is required")
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    if not isinstance(cluster, Atoms):
        raise TypeError("cluster must be an ase.Atoms object")

    n_atoms = len(cluster)

    # Step 2: 0 或 1 个原子的团簇视为连通.
    if n_atoms <= 1:
        return True

    # Step 3: 取坐标并初始化 DFS 状态.
    positions = np.array(cluster.get_positions(), dtype=float)
    visited = [False] * n_atoms
    stack = [0]
    visited[0] = True

    # Step 4: 基于 cutoff 建边并做 DFS 遍历.
    while stack:
        i = stack.pop()
        diff = positions - positions[i]
        distances = np.linalg.norm(diff, axis=1)

        # 距离 <= cutoff 视为邻居, 距离 > 0 排除自身.
        neighbors = np.where((distances <= cutoff) & (distances > 0.0))[0]
        for j in neighbors.tolist():
            if not visited[j]:
                visited[j] = True
                stack.append(j)

    # Step 5: 所有原子都被访问到则连通.
    return all(visited)


def mutate(pop: Population, a: float = 4, n: int = 10, debug: bool = False) -> Population:
    """对种群执行突变,返回突变后的暂定种群.

    Parameters
    ----------
    pop : Population
        输入种群.
    a : float, optional
        元素对应晶格常数(Ang). Default 4.
        仅在柯西分支用于半径约束阈值计算.
    n : int, optional
        柯西分支重采样最大次数. Default 10.
        超过上限后将停止重采样,并使用最后一次候选坐标.
    debug : bool, optional
        是否输出柯西分支重采样过程信息. Default False.
    """
    if len(pop) < 4:
        raise ValueError("Mutation requires at least 4 clusters in population")
    if a <= 0:
        raise ValueError("a must be > 0")
    if n <= 0:
        raise ValueError("n must be > 0")

    temp_clusters = pop.copy()

    # 选取当前最优个体索引(最低能量),要求所有个体能量都已就绪
    energies = pop.energies
    for i, energy_value in enumerate(energies):
        if energy_value is None:
            raise ValueError(f"All clusters must have energy before mutation. Missing index: {i}")
    index_best = int(np.argmin(np.array(energies, dtype=float)))

    n_atoms = pop.num
    # WARNING: 系数过小会导致柯西分支重采样次数激增, 甚至表现为长时间循环.
    # 这里使用 10*a 对应阈值 5*r0, 放宽失败判据以降低卡顿风险.
    radius_limit = 10 * a * (0.5 + (3 * n_atoms / (4 * np.pi * np.sqrt(2)))**(1.0 / 3.0))

    for i in range(len(pop)):
        # 采样缩放因子分支:按 fp 在高斯/柯西之间切换.
        use_gaussian = np.random.uniform(0.0, 1.0) < temp_clusters.fp
        # mut_tag 在分布分支确定后立即确定: 0=Gaussian, 1=Cauchy.
        mut_tag = 0 if use_gaussian else 1

        # 从除 i 外的个体中采样互异索引
        candidate_indices = [idx for idx in range(len(pop)) if idx != i]
        index1, index2, index3 = np.random.choice(candidate_indices, size=3, replace=False)

        # 按 p 在两种突变策略之间切换
        use_rand_strategy = np.random.uniform(0.0, 1.0) < temp_clusters.p

        def build_positions(
            scale_factor: float,
            current_index: int,
            best_index: int,
            idx1: int,
            idx2: int,
            idx3: int,
            rand_strategy: bool,
        ) -> np.ndarray:
            # Pass indices and strategy explicitly to avoid capturing loop variables.
            if rand_strategy:
                # DE/rand/1
                return (
                    pop[idx1].get_positions()
                    + scale_factor * (pop[idx2].get_positions() - pop[idx3].get_positions())
                )
            # DE/current-to-best/2
            return (
                pop[current_index].get_positions()
                + scale_factor
                * (pop[best_index].get_positions() - pop[current_index].get_positions())
                + scale_factor * (pop[idx1].get_positions() - pop[idx2].get_positions())
            )

        if use_gaussian:
            # 高斯分支保持原逻辑.
            scale_factor = float(np.random.normal(0.5, 0.3))
            new_positions = build_positions(
                scale_factor, i, index_best, index1, index2, index3, use_rand_strategy
            )
        else:
            # 柯西分支增加半径约束: 若超出 2*r0 则重采样柯西分布.
            retry_count = 0
            last_positions: np.ndarray | None = None
            while True:
                scale_factor = float(np.random.standard_cauchy())
                trial_positions = build_positions(
                    scale_factor, i, index_best, index1, index2, index3, use_rand_strategy
                )
                last_positions = trial_positions
                max_distance = float(np.linalg.norm(trial_positions, axis=1).max())
                if max_distance <= radius_limit:
                    new_positions = trial_positions
                    break
                if debug:
                    print(f"Mutation failed, atom at distance {max_distance:.2f} from origin")
                retry_count += 1
                if retry_count >= n:
                    if last_positions is None:
                        raise RuntimeError("mutate internal error: last_positions is unavailable")
                    if debug:
                        print(f"Warning: Cauchy retry limit reached n={n}, using last candidate positions")
                    new_positions = last_positions
                    break

        # 突变产生重合几何时退回父代坐标, 避免把非法个体送入优化.
        if min_pair_distance(new_positions) < MIN_ATOM_DISTANCE:
            new_positions = pop[i].get_positions()

        # 记录本个体突变标签到 info['mut'] (0/1).
        temp_clusters[i].set_positions(new_positions)
        temp_clusters[i].info['mut'] = mut_tag
        temp_clusters[i].info['energy'] = None
        temp_clusters[i].arrays.pop('forces', None)

    return temp_clusters


def crossover(pop1: Population, pop2: Population) -> Population:
    """对原种群与突变种群执行交叉,返回交叉后种群."""
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for crossover")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for crossover")

    temp_clusters = Population(symbol=pop1.symbol)
    temp_clusters._fp = pop1.fp
    temp_clusters._p = pop1.p
    temp_clusters._crm = pop1.crm

    for i in range(len(pop1)):
        # 采样当前个体交叉率,并裁剪到 [0, 1].
        # 裁剪是为了保证交叉概率合法,避免出现负值或大于 1.
        cross_rate = np.random.normal(pop1.crm, 0.1)
        cross_rate = np.clip(cross_rate, 0.0, 1.0)

        # 至少保留一个原子来自突变种群
        j_rand = np.random.randint(0, pop1.num)

        # 使用坐标拷贝进行交叉,保证不原地改动输入种群 pop1/pop2.
        base_positions = pop1[i].get_positions().copy()
        mut_positions = pop2[i].get_positions().copy()
        trial_positions = base_positions.copy()

        # 逐原子交叉:按 cr 或 j_rand 选择保留突变位姿.
        # 父代原子顺序不一致时可能选中同一物理原子, 导致坐标重合, 故重采.
        for _attempt in range(10):
            trial_positions = base_positions.copy()
            for j in range(pop1.num):
                if np.random.uniform(0.0, 1.0) <= cross_rate or j == j_rand:
                    trial_positions[j] = mut_positions[j]
            if min_pair_distance(trial_positions) >= MIN_ATOM_DISTANCE:
                break
        else:
            trial_positions = base_positions.copy()

        trial_cluster = Atoms(
            symbols=pop1[i].get_chemical_symbols(),
            positions=trial_positions,
        )
        # 记录本个体交叉率 cr,供 select 在成功样本上更新 crm.
        trial_cluster.info['cr'] = cross_rate
        # 传递 mut_tag,供 select 统计对应策略成功/失败并更新 p.
        trial_cluster.info['mut'] = pop2[i].info.get('mut')

        temp_clusters.add_cluster(trial_cluster)

    return temp_clusters


def sphere_cut_splice(
    pop1: Population,
    pop2: Population,
    r: float = 0.8,
    relax: Callable[[Population | Atoms], Population | Atoms] | None = None,
    debug: bool = False,
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
        球面半径定义为 R = dmax_a * r.
    relax : Callable, optional
        局域优化函数. 可传入偏函数 relax() 或函数对象 local_optimize().
        若使用默认值 None, 则直接报错.
    debug : bool, optional
        是否输出重选过程信息. Default False.
    """
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for sphere_cut_splice")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for sphere_cut_splice")
    if len(pop1) < 2:
        raise ValueError("sphere_cut_splice requires at least 2 clusters per population")
    if not (0.0 < r <= 1.0):
        raise ValueError("r must be in (0, 1]")

    temp_clusters = Population(symbol=pop1.symbol)
    temp_clusters._fp = pop1.fp
    temp_clusters._p = pop1.p
    temp_clusters._crm = pop1.crm

    if relax is None:
        raise ValueError("sphere_cut_splice requires explicit relax callable")

    def splice_one(cluster_a: Atoms, cluster_b: Atoms) -> Atoms | None:
        work_a = cluster_a.copy()
        work_b = cluster_b.copy()

        # 这里按需求使用质心(center of mass)移动到原点,不使用几何中心.
        work_a.set_center_of_mass([0.0, 0.0, 0.0])
        work_b.set_center_of_mass([0.0, 0.0, 0.0])

        positions_a = work_a.get_positions().copy()
        positions_b = work_b.get_positions().copy()
        symbols_a = work_a.get_chemical_symbols()
        symbols_b = work_b.get_chemical_symbols()

        dist_a = np.linalg.norm(positions_a, axis=1)
        dist_b = np.linalg.norm(positions_b, axis=1)

        dmax_a = float(np.max(dist_a))
        radius = dmax_a * r

        symbol_set = sorted(set(symbols_a))
        swap_pairs: list[tuple[int, int]] = []

        for symbol in symbol_set:
            outer_indices_a = [
                i for i, atom_symbol in enumerate(symbols_a)
                if atom_symbol == symbol and dist_a[i] > radius
            ]

            count_outer = len(outer_indices_a)
            if count_outer == 0:
                continue

            candidate_indices_b = [
                i for i, atom_symbol in enumerate(symbols_b)
                if atom_symbol == symbol
            ]
            candidate_indices_b.sort(key=lambda i: dist_b[i], reverse=True)
            selected_indices_b = candidate_indices_b[:count_outer]

            if len(selected_indices_b) < count_outer:
                return None

            # 若团簇2被选中交换的原子里出现 <= dmax_a*r/2,则本次交换失败并重选.
            threshold = radius / 2.0
            if any(dist_b[idx] <= threshold for idx in selected_indices_b):
                return None

            outer_indices_a.sort(key=lambda i: dist_a[i], reverse=True)
            selected_indices_b.sort(key=lambda i: dist_b[i], reverse=True)
            swap_pairs.extend(zip(outer_indices_a, selected_indices_b))

        trial_a = work_a.copy()
        trial_b = work_b.copy()
        trial_pos_a = trial_a.get_positions().copy()
        trial_pos_b = trial_b.get_positions().copy()

        for index_a, index_b in swap_pairs:
            cache = trial_pos_a[index_a].copy()
            trial_pos_a[index_a] = trial_pos_b[index_b]
            trial_pos_b[index_b] = cache

        trial_a.set_positions(trial_pos_a)
        trial_b.set_positions(trial_pos_b)

        relaxed_a = relax(trial_a)
        relaxed_b = relax(trial_b)
        if not isinstance(relaxed_a, Atoms) or not isinstance(relaxed_b, Atoms):
            raise TypeError("relax must return Atoms when input is Atoms")

        # 与 crossover 一致: 传递 mut_tag 供后续 select 更新 p/crm.
        mut_tag = cluster_b.info.get('mut')
        if mut_tag is not None:
            relaxed_a.info['mut'] = mut_tag
            relaxed_b.info['mut'] = mut_tag

        # cr 生命周期透传: 供 select 在成功样本上更新 crm.
        cross_rate = cluster_b.info.get('cr')
        if cross_rate is not None:
            relaxed_a.info['cr'] = cross_rate
            relaxed_b.info['cr'] = cross_rate

        energy_a = relaxed_a.info.get('energy')
        energy_b = relaxed_b.info.get('energy')
        if energy_a is None or energy_b is None:
            raise ValueError("relax result must contain energy for sphere_cut_splice")

        return relaxed_a if float(energy_a) <= float(energy_b) else relaxed_b

    target_size = len(pop1)
    max_retries = target_size * 200
    retry_count = 0

    while len(temp_clusters) < target_size:
        if retry_count >= max_retries:
            raise RuntimeError("sphere_cut_splice failed to generate enough offspring after retries")
        retry_count += 1

        selected_1 = np.random.choice(len(pop1), size=2, replace=False)
        selected_2 = np.random.choice(len(pop2), size=2, replace=False)

        winners: list[Atoms] = []
        round_failed = False
        for local_index in range(2):
            winner = splice_one(pop1[int(selected_1[local_index])], pop2[int(selected_2[local_index])])
            if winner is None:
                round_failed = True
                break
            winners.append(winner)

        if round_failed:
            if debug:
                print(f"SphereCutSplice failed, retry_count={retry_count}")
            continue

        for winner in winners:
            if len(temp_clusters) >= target_size:
                break
            temp_clusters.add_cluster(winner)

    return temp_clusters


def local_transformation(
    pop: Population,
    r: float = 0.25,
    relax: Callable[[Population | Atoms], Population | Atoms] | None = None,
) -> Population:
    """局部变换:随机扰动后择优保留更低能量个体.

    对每个团簇原子坐标施加 [-r, r] 的均匀随机扰动.
    分别对扰动前后团簇调用 relax, 选择能量较低者进入下一代.

    Parameters
    ----------
    pop : Population
        输入种群.
    r : float, optional
        扰动幅度, 每个坐标分量在 [-r, r] 取样. Default 0.25.
    relax : Callable, optional
        局域优化函数. 可传偏函数 relax() 或函数对象 local_optimize().

    Returns
    -------
    Population
        经过局部变换与选择后的新种群.
    """
    if relax is None:
        raise ValueError("local_transformation requires explicit relax callable")
    if r < 0:
        raise ValueError("r must be >= 0")

    source_pop = pop
    output_pop = Population(symbol=source_pop.symbol)
    output_pop._fp = source_pop.fp
    output_pop._p = source_pop.p
    output_pop._crm = source_pop.crm

    base_energy_values: list[float | None] = source_pop.energies.copy()
    base_candidates: list[Atoms | None] = [None] * len(source_pop)

    for i, base_energy in enumerate(base_energy_values):
        if base_energy is None:
            base_cluster = source_pop[i].copy()
            relaxed_base = relax(base_cluster)
            if not isinstance(relaxed_base, Atoms):
                raise TypeError("relax must return Atoms when input is Atoms")
            relaxed_energy = relaxed_base.info.get('energy')
            if relaxed_energy is None:
                raise ValueError("relax result must contain energy for local_transformation")
            base_energy_values[i] = float(relaxed_energy)
            base_candidates[i] = relaxed_base
        else:
            base_candidate = source_pop[i].copy()
            base_candidate.info['energy'] = float(base_energy)
            base_candidates[i] = base_candidate

    for i in range(len(source_pop)):
        base_energy = base_energy_values[i]
        base_candidate = base_candidates[i]
        if base_energy is None or base_candidate is None:
            raise ValueError("base energy completion failed in local_transformation")

        # 传递 mut_tag: local_transformation 前后保持策略标签不丢失.
        mut_tag = source_pop[i].info.get('mut')
        if mut_tag is not None:
            base_candidate.info['mut'] = mut_tag

        # cr 生命周期透传: local_transformation 前后保持交叉率标签不丢失.
        cross_rate = source_pop[i].info.get('cr')
        if cross_rate is not None:
            base_candidate.info['cr'] = cross_rate

        perturbed_cluster = source_pop[i].copy()
        positions = perturbed_cluster.get_positions().copy()
        noise = np.random.uniform(-r, r, size=positions.shape)
        perturbed_cluster.set_positions(positions + noise)

        relaxed_perturbed = relax(perturbed_cluster)
        if not isinstance(relaxed_perturbed, Atoms):
            raise TypeError("relax must return Atoms when input is Atoms")

        energy_perturbed = relaxed_perturbed.info.get('energy')
        if energy_perturbed is None:
            raise ValueError("relax result must contain energy for local_transformation")

        if mut_tag is not None:
            relaxed_perturbed.info['mut'] = mut_tag
        if cross_rate is not None:
            relaxed_perturbed.info['cr'] = cross_rate

        winner = relaxed_perturbed if float(energy_perturbed) < float(base_energy) else base_candidate
        output_pop.add_cluster(winner)

    return output_pop


def select(
    pop1: Population,
    pop2: Population,
    relax: Callable[[Population | Atoms], Population | Atoms],
    pg_tolerance: float = 0.3,
) -> Population:
    """对原种群和候选种群执行选择并更新自适应参数.

    被选中的个体在进入下一代时计算点群(PG, 熊夫利符号);
    若 pymatgen 不可用或点群分析失败, 对应个体 PG 记为 None.

    Parameters
    ----------
    pop1 : Population
        原种群.
    pop2 : Population
        候选种群.
    relax : Callable
        局域优化函数. 可传入偏函数 relax() 或函数对象 local_optimize().
    pg_tolerance : float, optional
        计算点群时传给 pymatgen ``PointGroupAnalyzer`` 的坐标容差. Default 0.3.
    """
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for selection")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for selection")

    temp_clusters = Population(symbol=pop1.symbol)
    temp_clusters._fp = pop1.fp
    temp_clusters._p = pop1.p
    temp_clusters._crm = pop1.crm
    temp_clusters._candidate = [atoms.copy() for atoms in pop1.candidate]

    # pop1: 先读取现有能量,仅对缺失能量个体做 relax 补全.
    pop1_energies_raw: list[float | None] = pop1.energies.copy()
    pop1_candidates: list[Atoms | None] = [None] * len(pop1)

    for i, energy_value in enumerate(pop1_energies_raw):
        if energy_value is None:
            relaxed_cluster = relax(pop1[i].copy())
            if not isinstance(relaxed_cluster, Atoms):
                raise TypeError("relax must return Atoms when input is Atoms")
            relaxed_energy = relaxed_cluster.info.get('energy')
            if relaxed_energy is None:
                raise ValueError(f"pop1 relax result missing energy at index {i}")
            pop1_energies_raw[i] = float(relaxed_energy)
            pop1_candidates[i] = relaxed_cluster
        else:
            preserved_cluster = pop1[i].copy()
            preserved_cluster.info['energy'] = float(energy_value)
            pop1_candidates[i] = preserved_cluster

    # pop2: 与 pop1 一致,先读取能量,仅对缺失能量个体做 relax 补全.
    pop2_energies_raw: list[float | None] = pop2.energies.copy()
    pop2_candidates: list[Atoms | None] = [None] * len(pop2)

    for i, energy_value in enumerate(pop2_energies_raw):
        if energy_value is None:
            relaxed_cluster = relax(pop2[i].copy())
            if not isinstance(relaxed_cluster, Atoms):
                raise TypeError("relax must return Atoms when input is Atoms")
            relaxed_energy = relaxed_cluster.info.get('energy')
            if relaxed_energy is None:
                raise ValueError(f"pop2 relax result missing energy at index {i}")
            pop2_energies_raw[i] = float(relaxed_energy)
            pop2_candidates[i] = relaxed_cluster
        else:
            preserved_cluster = pop2[i].copy()
            preserved_cluster.info['energy'] = float(energy_value)
            pop2_candidates[i] = preserved_cluster

    cr_rec: list[float] = []
    delta_rec: list[float] = []
    ns1, ns2, nf1, nf2 = 0, 0, 0, 0

    # 选择环节要求两组种群都具备完整能量信息
    pop1_energies = pop1_energies_raw
    pop2_energies = pop2_energies_raw
    pop1_energy_values: list[float] = []
    pop2_energy_values: list[float] = []

    for i, energy_value in enumerate(pop1_energies):
        if energy_value is None:
            raise ValueError(f"pop1 is missing energy at index {i}")
        pop1_energy_values.append(float(energy_value))
    for i, energy_value in enumerate(pop2_energies):
        if energy_value is None:
            raise ValueError(f"pop2 is missing energy at index {i}")
        pop2_energy_values.append(float(energy_value))

    # mut_tag 生命周期终点: select 在此消费 mut_tag,用于统计并更新 p.
    # 选择环节要求 pop2 的 info 含 mut_tag,否则无法按策略统计并更新 p.
    for i in range(len(pop2_candidates)):
        pop2_candidate = pop2_candidates[i]
        if pop2_candidate is None:
            raise ValueError(f"pop2 candidate missing at index {i}")
        if pop2_candidate.info.get('mut') is None:
            raise ValueError(f"pop2 is missing mut_tag in info at index {i}")

    for i in range(len(pop1_candidates)):
        pop1_candidate = pop1_candidates[i]
        pop2_candidate = pop2_candidates[i]
        if pop1_candidate is None:
            raise ValueError(f"pop1 candidate missing at index {i}")
        if pop2_candidate is None:
            raise ValueError(f"pop2 candidate missing at index {i}")

        energy1 = pop1_energy_values[i]
        energy2 = pop2_energy_values[i]
        mut_tag = pop2_candidate.info.get('mut')

        choose_pop2 = energy2 < energy1

        if choose_pop2:
            temp_clusters.add_cluster(pop2_candidate)
            # 成功个体记录 cr 和能量改变量,用于后续更新 crm.
            current_cr = pop2_candidate.info.get('cr')
            if current_cr is not None:
                cr_rec.append(float(current_cr))
                delta_rec.append(float(energy1 - energy2))
            if mut_tag == 0:
                ns1 += 1
            elif mut_tag == 1:
                ns2 += 1
        else:
            temp_clusters.add_cluster(pop1_candidate)
            if mut_tag == 0:
                nf1 += 1
            elif mut_tag == 1:
                nf2 += 1

        # 个体通过选择后计算点群(熊夫利符号); 失败或超时记为 None.
        chosen_cluster = temp_clusters[-1]
        chosen_cluster.info['PG'] = compute_point_group(
            chosen_cluster, tolerance=pg_tolerance
        )

    # 更新 crm:使用成功个体的 cr 按能量改变量加权平均.
    if cr_rec and sum(delta_rec) > 0:
        temp_clusters._crm = float(np.average(cr_rec, weights=delta_rec))

    # 更新 p:按 mut_tag 两类策略的成功/失败统计更新选择概率.
    denominator = ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2)
    if denominator > 0:
        temp_clusters._p = float((ns1 * (ns2 + nf2)) / denominator)

    return temp_clusters
