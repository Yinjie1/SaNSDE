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
    matrix_tolerance: float = 0.1,
) -> str | None:
    """计算团簇点群(熊夫利符号), 失败或超时返回 None.

    pymatgen 的 ``PointGroupAnalyzer`` 在近简并结构上可能长时间生成对称操作,
    这里用 ``SIGALRM`` 限制单次计算时长. 首次超时后同时用 2 倍 tolerance
    和 2 倍 matrix_tolerance 重试; 仍然超时则放弃计算、返回 None 并发出警告.

    Parameters
    ----------
    cluster : Atoms
        待计算点群的团簇.
    tolerance : float, optional
        传给 pymatgen ``PointGroupAnalyzer`` 的坐标容差. Default 0.3.
        首次超时后会用 ``2 * tolerance`` 重试一次.
    timeout : float, optional
        每次点群计算允许的最长时间(秒). Default 5.0.
    matrix_tolerance : float, optional
        传给 pymatgen ``PointGroupAnalyzer`` 的矩阵闭包容差. Default 0.1.
        首次超时后会用 ``2 * matrix_tolerance`` 重试一次.

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
        attempts = (
            (tolerance, matrix_tolerance),
            (tolerance * 2, matrix_tolerance * 2),
        )
        for attempt, (current_tolerance, current_matrix_tolerance) in enumerate(attempts, 1):
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return str(
                    PointGroupAnalyzer(
                        molecule,
                        tolerance=current_tolerance,
                        matrix_tolerance=current_matrix_tolerance,
                    )
                    .get_pointgroup()
                    .sch_symbol
                )
            except TimeoutError:
                outcome = "point group set to None." if attempt == 2 else "retrying..."
                print(
                    f"Warning: PointGroupAnalyzer timed out after {timeout}s "
                    f"with tolerance {current_tolerance} and "
                    f"matrix_tolerance {current_matrix_tolerance}; {outcome}"
                )
                if attempt == 2:
                    return None
                # 第一次超时: 同时放宽 tolerance 和 matrix_tolerance 重试.
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

def local_optimize(
    pop: Population | Atoms, 
    cell: list[float] | np.ndarray | None = None, 
    potential: Calculator | None = None,
    optimizer: type[Optimizer] = BFGS,
    fmax: float = 0.05, 
    steps: int = 100
    ) -> Population | Atoms:
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


def random_cluster(
    pop: Population, 
    a: float, 
    delta: float, 
    relax: Callable[[Population | Atoms], Population | Atoms],
    is_2d: bool = False
    ) -> None:
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
        # scale_factor_tag 在分布分支确定后立即确定: 0=Gaussian, 1=Cauchy.
        scale_factor_tag = 0 if use_gaussian else 1

        # 从除 i 外的个体中采样互异索引
        candidate_indices = [idx for idx in range(len(pop)) if idx != i]
        index1, index2, index3 = np.random.choice(candidate_indices, size=3, replace=False)

        # 按 p 在两种突变策略之间切换; 0=DE/rand/1, 1=DE/current-to-best/2.
        use_rand_strategy = np.random.uniform(0.0, 1.0) < temp_clusters.p
        strategy_tag = 0 if use_rand_strategy else 1

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

        # Record the distribution tag, strategy tag and the sampled scaling factor.
        temp_clusters[i].set_positions(new_positions)
        temp_clusters[i].info['scale_factor_tag'] = scale_factor_tag
        temp_clusters[i].info['strategy_tag'] = strategy_tag
        temp_clusters[i].info['F'] = scale_factor
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
    temp_clusters.info = pop1.info.copy()

    for i in range(len(pop1)):
        # Reuse the CR kept by the individual; sample a new one only when missing.
        cross_rate = pop1[i].info.get('cr')
        if cross_rate is None:
            cross_rate = np.random.normal(pop1.crm, 0.1)
        cross_rate = float(np.clip(cross_rate, 0.0, 1.0))

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
                    trial_positions[j] = mut_positions[j] # type: ignore
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
        # Propagate scale_factor_tag, strategy_tag and F so select can update p and fp.
        trial_cluster.info['scale_factor_tag'] = pop2[i].info.get('scale_factor_tag')
        trial_cluster.info['strategy_tag'] = pop2[i].info.get('strategy_tag')
        trial_cluster.info['F'] = pop2[i].info.get('F')

        temp_clusters.add_cluster(trial_cluster)

    return temp_clusters


def sphere_cut_splice(
    pop: Population,
    trial: Population,
    relax: Callable[[Population | Atoms], Population | Atoms] | None = None,
) -> Population:
    """Sphere-cut crossover driven by the sum of atomic coordinates.

    For every individual the parent ``pop[i]`` is the father and the crossover
    offspring ``trial[i]`` is the mother. Both structures are moved to the
    origin, every atom gets the scalar ``x + y + z`` of its position, and both
    atom lists are sorted by that scalar. A cut position is a rank where the
    relative order of the two sorted lists flips; one such rank is drawn at
    random and used as the threshold. One child keeps the father core and takes
    the mother shell, the other keeps the mother core and takes the father
    shell. Both children are relaxed and the lower-energy one is kept.

    Only coordinates are exchanged, so each child keeps the chemical
    composition of its template. Every individual always produces one
    offspring, so there is no failure or retry branch.

    Parameters
    ----------
    pop : Population
        Parent population.
    trial : Population
        Crossover offspring; ``trial[i]`` is crossed with ``pop[i]``.
    relax : Callable, optional
        Local optimization callable. ``None`` raises ``ValueError``.

    Returns
    -------
    Population
        Offspring population with the same length as ``pop``.

    Raises
    ------
    ValueError
        If the populations differ in symbol or length, or ``relax`` is missing.
    TypeError
        If ``relax`` does not return an ``Atoms`` object.
    """
    if pop.symbol != trial.symbol:
        raise ValueError("Population symbols must match for sphere_cut_splice")
    if len(pop) != len(trial):
        raise ValueError("Population sizes must match for sphere_cut_splice")
    if relax is None:
        raise ValueError("sphere_cut_splice requires explicit relax callable")

    temp_clusters = Population(symbol=pop.symbol)
    temp_clusters.info = pop.info.copy()

    for index in range(len(pop)):
        father = pop[index]
        mother = trial[index]

        father_positions = np.asarray(father.get_positions(), dtype=float)
        mother_positions = np.asarray(mother.get_positions(), dtype=float)
        father_positions = father_positions - np.mean(father_positions, axis=0)
        mother_positions = mother_positions - np.mean(mother_positions, axis=0)

        father_sum = np.sum(father_positions, axis=1)
        mother_sum = np.sum(mother_positions, axis=1)
        father_rank = np.copy(father_sum)
        mother_rank = np.copy(mother_sum)

        n_atoms = len(father)
        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                if father_sum[i] == father_sum[j]:
                    father_sum[i] += (2.0 + np.random.random()) * 1e-16
                    father_rank[i] = father_sum[i]
                if mother_sum[i] == mother_sum[j]:
                    mother_sum[i] += (2.0 + np.random.random()) * 1e-16
                    mother_rank[i] = mother_sum[i]

        father_rank.sort()
        mother_rank.sort()

        crossings = [
            k
            for k in range(1, n_atoms)
            if mother_rank[k - 1] < father_rank[k - 1]
            and father_rank[k] < mother_rank[k]
        ]

        child1_positions = father_positions.copy()
        child2_positions = mother_positions.copy()
        if crossings:
            middle = int(np.ceil(len(crossings) * np.random.random()))
            if middle == 0:
                middle = 1
            cut = crossings[middle - 1]
            father_cut = father_rank[cut]
            mother_cut = mother_rank[cut]

            mother_shell = mother_sum > mother_cut + 1e-16
            child1_positions[mother_shell] = mother_positions[mother_shell]
            father_shell = father_sum > father_cut + 1e-16
            child2_positions[father_shell] = father_positions[father_shell]

        children: list[Atoms] = []
        for template, positions in (
            (father, child1_positions),
            (mother, child2_positions),
        ):
            cluster = template.copy()
            cluster.set_positions(positions)
            relaxed = relax(cluster)
            if not isinstance(relaxed, Atoms):
                raise TypeError("relax must return Atoms when input is Atoms")
            if relaxed.info.get('energy') is None:
                raise ValueError(
                    "relax result must contain energy for sphere_cut_splice"
                )
            children.append(relaxed)

        winner = children[1]
        if float(children[0].info['energy']) < float(children[1].info['energy']):
            winner = children[0]

        for key in ('cr', 'scale_factor_tag', 'strategy_tag', 'F'):
            value = mother.info.get(key)
            if value is not None:
                winner.info[key] = value

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
    output_pop.info = source_pop.info.copy()

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

        # Propagate scale_factor_tag and F through local_transformation.
        scale_factor_tag = source_pop[i].info.get('scale_factor_tag')
        if scale_factor_tag is not None:
            base_candidate.info['scale_factor_tag'] = scale_factor_tag
        f_value = source_pop[i].info.get('F')
        if f_value is not None:
            base_candidate.info['F'] = f_value
        strategy_tag = source_pop[i].info.get('strategy_tag')
        if strategy_tag is not None:
            base_candidate.info['strategy_tag'] = strategy_tag

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

        if scale_factor_tag is not None:
            relaxed_perturbed.info['scale_factor_tag'] = scale_factor_tag
        if f_value is not None:
            relaxed_perturbed.info['F'] = f_value
        if strategy_tag is not None:
            relaxed_perturbed.info['strategy_tag'] = strategy_tag
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
    update_crm: bool = False,
    ediff: float = 1e-5,
) -> Population:
    """Select the next generation and accumulate adaptive statistics.

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
    update_crm : bool, optional
        Whether to update crm from the recorded successful CR values.
        Default False; the caller enables it at the end of a CRm period.
    ediff : float, optional
        能量容差. 候选个体的能量比父代低超过 ediff 时才算胜出并进入下一代;
        ``ediff = 0`` 时退化为 ``energy2 < energy1``. Default 1e-5.
    """
    if pop1.symbol != pop2.symbol:
        raise ValueError("Population symbols must match for selection")
    if len(pop1) != len(pop2):
        raise ValueError("Population sizes must match for selection")

    temp_clusters = Population(symbol=pop1.symbol)
    temp_clusters.info = pop1.info.copy()
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
    nsg, nfg, nsc, nfc = 0, 0, 0, 0

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

    # scale_factor_tag and strategy_tag are consumed here: scale_factor_tag tracks the F
    # distribution (Gaussian/Cauchy) for fp, strategy_tag tracks the mutation
    # strategy (rand/1 or current-to-best/2) for p.
    for i in range(len(pop2_candidates)):
        pop2_candidate = pop2_candidates[i]
        if pop2_candidate is None:
            raise ValueError(f"pop2 candidate missing at index {i}")
        if pop2_candidate.info.get('scale_factor_tag') is None:
            raise ValueError(f"pop2 is missing scale_factor_tag in info at index {i}")
        if pop2_candidate.info.get('strategy_tag') is None:
            raise ValueError(f"pop2 is missing strategy_tag in info at index {i}")

    for i in range(len(pop1_candidates)):
        pop1_candidate = pop1_candidates[i]
        pop2_candidate = pop2_candidates[i]
        if pop1_candidate is None:
            raise ValueError(f"pop1 candidate missing at index {i}")
        if pop2_candidate is None:
            raise ValueError(f"pop2 candidate missing at index {i}")

        energy1 = pop1_energy_values[i]
        energy2 = pop2_energy_values[i]
        scale_factor_tag = pop2_candidate.info.get('scale_factor_tag')
        strategy_tag = pop2_candidate.info.get('strategy_tag')

        # 候选需要比父代低超过 ediff 才算胜出; ediff=0 时即 energy2 < energy1.
        choose_pop2 = energy2 < energy1 - ediff

        if choose_pop2:
            temp_clusters.add_cluster(pop2_candidate)
            # 成功个体记录 cr 和能量改变量,用于后续更新 crm.
            current_cr = pop2_candidate.info.get('cr')
            if current_cr is not None:
                cr_rec.append(float(current_cr))
                delta_rec.append(float(energy1 - energy2))
            if strategy_tag == 0:
                ns1 += 1
            elif strategy_tag == 1:
                ns2 += 1
            if scale_factor_tag == 0:
                nsg += 1
            elif scale_factor_tag == 1:
                nsc += 1
        else:
            temp_clusters.add_cluster(pop1_candidate)
            if strategy_tag == 0:
                nf1 += 1
            elif strategy_tag == 1:
                nf2 += 1
            if scale_factor_tag == 0:
                nfg += 1
            elif scale_factor_tag == 1:
                nfc += 1

        # 个体通过选择后计算点群(熊夫利符号); 失败或超时记为 None.
        chosen_cluster = temp_clusters[-1]
        chosen_cluster.info['PG'] = compute_point_group(
            chosen_cluster, tolerance=pg_tolerance
        )

    # Record successful CR values and their fitness improvements.
    if cr_rec and sum(delta_rec) > 0:
        temp_clusters.info['cr_rec'].extend(cr_rec)
        temp_clusters.info['delta_rec'].extend(delta_rec)

    # Update crm at the end of the CRm period and reset both record lists.
    if update_crm:
        temp_clusters.update_crm()

    # Accumulate success counts for the learning period. p and fp are updated
    # by Population.update_adaptive() only when the learning period ends.
    temp_clusters.info['ns1'] += ns1
    temp_clusters.info['ns2'] += ns2
    temp_clusters.info['nf1'] += nf1
    temp_clusters.info['nf2'] += nf2
    temp_clusters.info['nsg'] += nsg
    temp_clusters.info['nsc'] += nsc
    temp_clusters.info['nfg'] += nfg
    temp_clusters.info['nfc'] += nfc

    return temp_clusters
