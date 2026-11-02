"""pop.py
python 3.13.12
"""
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Self, overload

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz
from ase.io.jsonio import read_json, write_json
from ase.io.trajectory import Trajectory


class MultiAtoms:
    """List-like container for multiple Atoms clusters.

    The container only stores clusters and does not require the chemical
    formulas to be identical.

    Attributes
    ----------
    clusters : list[Atoms]
        Clusters currently held in the container.
    symbols : list[str]
        Chemical formula list in cluster order.
    energies : list[float | None]
        Energies read from ``cluster.info['energy']``.
    forces : list[list[list[float]] | None]
        Forces read from ``cluster.arrays['forces']``.
    PGs : list[str | None]
        Point groups read from ``cluster.info['PG']``.

    Examples
    --------
    >>> multi = MultiAtoms()
    >>> multi.add_cluster(cluster)
    >>> multi[0]
    >>> multi[::2]
    >>> cluster in multi
    >>> multi[0] = cluster
    >>> del multi[0]
    >>> multi.symbols
    """

    def __init__(self) -> None:
        self._clusters: list[Atoms] = []

    @property
    def clusters(self) -> list[Atoms]:
        return self._clusters

    @property
    def symbols(self) -> list[str]:
        return [cluster.get_chemical_formula() for cluster in self._clusters]

    def __iter__(self) -> Iterator[Atoms]:
        return iter(self._clusters)

    def __len__(self) -> int:
        return len(self._clusters)

    @overload
    def __getitem__(self, index: int) -> Atoms: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> Atoms | Self:
        if isinstance(index, slice):
            sub_atoms = MultiAtoms()
            sub_atoms._clusters = self._clusters[index]
            return sub_atoms # type: ignore
        return self._clusters[index]

    def __delitem__(self, index: int | slice) -> None:
        del self._clusters[index]

    def __contains__(self, item: Atoms) -> bool:
        return item in self._clusters

    @overload
    def __setitem__(self, index: int, value: Atoms) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Atoms]) -> None: ...

    def __setitem__(
        self,
        index: int | slice,
        value: Atoms | Iterable[Atoms],
    ) -> None:
        if isinstance(index, slice):
            clusters = list(value)
            for cluster in clusters:
                if not isinstance(cluster, Atoms):
                    raise TypeError("Cluster must be an ase.Atoms object")
            self._clusters[index] = clusters # type: ignore
            return
        if not isinstance(value, Atoms):
            raise TypeError("Cluster must be an ase.Atoms object")
        self._clusters[index] = value

    @property
    def energies(self) -> list[float | None]:
        return [
            float(cluster.info['energy']) if 'energy' in cluster.info else None
            for cluster in self._clusters
        ]

    @property
    def forces(self) -> list[list[list[float]] | None]:
        return [
            cluster.arrays['forces'].tolist() if 'forces' in cluster.arrays else None
            for cluster in self._clusters
        ]

    @property
    def PGs(self) -> list[str | None]:
        return [cluster.info.get('PG') for cluster in self._clusters]

    def copy(self) -> Self:
        """Return an independent copy of this cluster container.

        Clusters are copied with ``Atoms.copy()``; calculators are not copied.
        """
        new_atoms = type(self)()
        new_atoms._clusters = [cluster.copy() for cluster in self._clusters]
        return new_atoms

    def add_cluster(self, cluster: Atoms) -> None:
        """Append a cluster to the end of the container.

        Parameters
        ----------
        cluster : Atoms
            Cluster to add. It is stored as an ``Atoms.copy()``; the
            calculator is not copied.
        """
        if not isinstance(cluster, Atoms):
            raise TypeError("Cluster must be an ase.Atoms object")
        self._clusters.append(cluster.copy())

    def deduplicate(self, s: float = 0.99, s_c1: float = 0.98) -> None:
        """Remove identical clusters with the same chemical formula.

        Clusters are compared only when their chemical formulas match. Two
        clusters are treated as duplicates when their point groups are equal
        and their similarity is at least ``s``. For ``C1`` clusters, the
        threshold is ``s_c1``. The lower-energy cluster is kept.

        Parameters
        ----------
        s : float, optional
            Similarity threshold for non-``C1`` point groups. Default 0.99.
        s_c1 : float, optional
            Similarity threshold for ``C1`` point groups. Default 0.98.
        """
        from sansde.operators import similarity_function

        kept: list[Atoms] = []
        for cluster in self._clusters:
            formula = cluster.get_chemical_formula()
            duplicate_index: int | None = None
            for index, existing in enumerate(kept):
                if existing.get_chemical_formula() != formula:
                    continue
                pg_existing = existing.info.get('PG')
                pg_cluster = cluster.info.get('PG')
                if pg_existing is None or pg_existing != pg_cluster:
                    continue
                threshold = s_c1 if pg_existing == 'C1' else s
                if similarity_function(existing, cluster) < threshold:
                    continue
                duplicate_index = index
                break

            if duplicate_index is None:
                kept.append(cluster)
                continue

            existing = kept[duplicate_index]
            energy_existing = existing.info.get('energy')
            energy_cluster = cluster.info.get('energy')
            # If the new cluster has no energy, keep the existing duplicate.
            if energy_cluster is None:
                print(f"Removed {formula}, PG={pg_cluster}")
                continue
            # Prefer the cluster that has an energy value.
            if energy_existing is None or float(energy_cluster) < float(energy_existing):
                print(f"Removed {formula}, PG={pg_existing}")
                kept[duplicate_index] = cluster
            else:
                print(f"Removed {formula}, PG={pg_cluster}")

        self._clusters = kept


class Population(MultiAtoms):
    """MultiAtoms subclass with evolutionary state.

    Inherits the cluster container functionality of MultiAtoms and additionally
    maintains the candidate list and adaptive parameters.
    All clusters in the population must share the same chemical formula.

    Parameters
    ----------
    symbol : str | None = None
        Population chemical formula (e.g., "Au13"). When not specified, it is
        set automatically from the first added cluster.

    Attributes
    ----------
    symbol : str | None
        Population chemical formula. It can only be initialized once.
    candidate : list[Atoms]
        History of candidate clusters.
    fp : float
        Selection probability parameter of the mutation scale factor sampling
        distribution.
    p : float
        Selection probability parameter of the mutation strategy.
    crm : float
        Center parameter of the crossover rate.
    num : int
        Number of atoms in a single cluster. It is 0 when symbol is not set.
    best_cluster : Atoms
        Lowest-energy cluster in the population.

    Examples
    --------
    >>> pop = Population(symbol="Au13")
    >>> pop.add_cluster(cluster)
    >>> pop[::2]
    >>> pop.symbol
    'Au13'
    """

    def __init__(self, symbol: str | None = None):
        super().__init__()
        self._symbol: str | None = symbol
        self._fp: float = 0.5
        self._p: float = 0.5
        self._crm: float = 0.5
        self._candidate: list[Atoms] = []
        self._success_counts: dict[str, int] = {
            'ns1': 0,
            'ns2': 0,
            'nf1': 0,
            'nf2': 0,
            'nsg': 0,
            'nsc': 0,
            'nfg': 0,
            'nfc': 0,
        }

    @property
    def best_cluster(self) -> Atoms:
        """Return the cluster with the lowest energy.

        Raises
        ------
        ValueError
            If the population is empty or no cluster has a valid energy.
        """
        if not self._clusters:
            raise ValueError(
                f"{type(self).__name__} is empty, best_cluster is unavailable"
            )
        valid_energies: list[tuple[int, float]] = [
            (index, energy)
            for index, energy in enumerate(self.energies)
            if energy is not None
        ]
        if not valid_energies:
            raise ValueError(
                f"No valid energy in {type(self).__name__}, "
                "best_cluster is unavailable"
            )
        return self._clusters[min(valid_energies, key=lambda item: item[1])[0]]

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

    def update_adaptive(self, adapt_fp: bool = True) -> None:
        """Update p and fp from the accumulated success counts.

        Counts are accumulated by :func:`~sansde.operators.select` over a
        learning period and reset after this call.

        Parameters
        ----------
        adapt_fp : bool, optional
            If True (default), update fp from the Gaussian/Cauchy counts.
            If False, fp is kept unchanged.
        """
        counts = self._success_counts
        ns1, ns2 = counts['ns1'], counts['ns2']
        nf1, nf2 = counts['nf1'], counts['nf2']
        denominator = ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2)
        if denominator > 0:
            self._p = float((ns1 * (ns2 + nf2)) / denominator)

        if adapt_fp:
            nsg, nsc = counts['nsg'], counts['nsc']
            nfg, nfc = counts['nfg'], counts['nfc']
            fp_denominator = nsc * (nsg + nfg) + nsg * (nsc + nfc)
            if fp_denominator > 0:
                self._fp = float((nsg * (nsc + nfc)) / fp_denominator)

        for key in counts:
            counts[key] = 0

    @property
    def symbol(self) -> str | None:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        if self._symbol is not None:
            raise ValueError("symbol has already been set!")
        self._symbol = value

    @property
    def num(self) -> int:
        if self._symbol is None:
            return 0
        return len(Atoms(self._symbol))

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
                sub_pop.add_cluster(cluster)
            return sub_pop # type: ignore
        return super().__getitem__(index)

    def __setitem__(self, index: int | slice, value: Atoms | Iterable[Atoms]) -> None:
        raise TypeError(
            "Population does not support item assignment; use add_cluster()"
        )

    def add_cluster(self, cluster: Atoms) -> None:
        """Add a cluster and validate its chemical formula.

        If the population symbol is not set, it is set from the first cluster;
        otherwise the cluster formula must match the population symbol.
        The center of mass of the added cluster is translated to the origin.

        Parameters
        ----------
        cluster : Atoms
            Cluster to add.
        """
        if not isinstance(cluster, Atoms):
            raise TypeError("Cluster must be an ase.Atoms object")
        current_symbol = cluster.get_chemical_formula()
        if self._symbol is None:
            self.symbol = current_symbol
        elif current_symbol != self._symbol:
            raise ValueError(
                f"Cluster symbol {current_symbol} does not match "
                f"population symbol {self._symbol}"
            )
        super().add_cluster(cluster)
        self._clusters[-1].set_center_of_mass([0.0, 0.0, 0.0])

    def copy(self) -> Self:
        """Return an independent copy of this population and its clusters.

        Clusters are copied with ``Atoms.copy()``, so ASE calculators are not
        copied. The population fp/p/crm, accumulated success counts, cluster
        coordinates, info, arrays and the candidate list are all preserved.
        """
        new_pop = Population(symbol=self._symbol)
        new_pop._fp = self._fp
        new_pop._p = self._p
        new_pop._crm = self._crm
        new_pop._success_counts = self._success_counts.copy()
        # Copy the list directly to avoid add_cluster translating the cluster
        # center of mass again.
        new_pop._clusters = [cluster.copy() for cluster in self._clusters]
        new_pop._candidate = [cluster.copy() for cluster in self._candidate]
        return new_pop # type: ignore


    def add_candidate(self, cluster: Atoms, candidate_size: int = 2) -> None:
        """Insert a cluster at the front of the candidate list, keeping the
        list length at most candidate_size.

        The candidate cluster must have the same chemical formula as the
        population. If the population symbol is empty, it is set when the first
        candidate cluster is added.

        Parameters
        ----------
        cluster : Atoms
            Cluster to add to the candidate list. Its chemical formula must match
            the population symbol.
        candidate_size : int, optional
            Maximum length of the candidate list. Default 2, meaning only the
            best and the second best clusters are kept.
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
        """Export population data to the file system.

        Writes a JSON file under the directory given by dir (default
        './Populations/', created if it does not exist).
        Clusters are stored in the ``clusters`` list in ASE JSON format
        (Atoms objects). When xyz=True, a multi-frame extxyz file with the
        same name is also written, containing energies, forces, pbc and the
        other fields in info.

        Parameters
        ----------
        name : str, optional
            Export name (without extension).
            Naming rules:
            (1) When name is None, the population symbol is used; if the symbol
                is empty, "Population" is used.
            (2) The JSON file name is always "{name}.json".
            (3) When "{name}.json" already exists, a timestamp suffix is appended
                automatically, producing "{name}_YYYYMMddhhmmss.json" to avoid
                overwriting.
            (4) When xyz=True, the xyz file name matches the final JSON name
                (without extension).
        xyz : bool, optional
            Whether to also export a multi-frame extxyz file containing all
            clusters. Default False.
        dir : str | pathlib.Path, optional
            Output directory. Default "./Populations".

        Returns
        -------
        pathlib.Path
            Full absolute path of the exported JSON file.
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

        # Optional extxyz snapshot: only use the energy/forces already present in
        # info/arrays; do not read or recompute the calculator. Missing fields
        # are left empty.
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
            # Each cluster is stored directly in the list in ASE JSON format (Atoms).
            "clusters": self._clusters,
            "candidate": self._candidate,
        }

        with summary_path.open('w', encoding='utf-8') as f:
            write_json(f, summary_data)

        return summary_path.resolve()

    def dump(self, name: str | None = None) -> Path:
        """Export the current best cluster to an ASE trajectory file (.traj).

        The trajectory file is written under ./Trajectories/ via the ASE
        Trajectory interface. If the file already exists, new frames are
        appended.
        Energy and forces are read only from the native ASE containers of the
        cluster:
        - If cluster.info['energy'] exists, energy is written.
        - If cluster.arrays['forces'] exists, forces are written.
        - No extra computation is triggered; missing fields are not written.

        Parameters
        ----------
        name : str, optional
            Trajectory file name (without extension). Defaults to "{symbol}_best".

        Returns
        -------
        pathlib.Path
            Full absolute path of the exported trajectory file.
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
            traj.write(export_cluster, **write_kwargs) # type: ignore

        return traj_path.resolve()

    def dump_xyz(
        self,
        name: str | None = None,
        candidate: bool = False,
        dir: str | Path = "./Trajectories",
    ) -> Path:
        """Export clusters to an extxyz file (.xyz).

        With ``candidate=False`` only the current best cluster is written, and
        the file is appended if it already exists, which records the evolution
        of the best cluster. With ``candidate=True`` the whole ``candidate``
        list is written in order and the file with the same name is overwritten.

        Only the energies and forces already present in ``info``/``arrays`` are
        used; the calculator is neither read nor recomputed.

        Parameters
        ----------
        name : str, optional
            File name (without extension). Defaults to "{symbol}_best".
        candidate : bool, optional
            Whether to export the whole candidate list instead of the current
            best cluster. Default False.
        dir : str | pathlib.Path, optional
            Output directory. Default "./Trajectories".

        Returns
        -------
        pathlib.Path
            Full absolute path of the exported extxyz file.
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
        """Import population data from a JSON file.

        The specified JSON file is read directly.
        ASE JSON (Atoms) entries in ``clusters`` are rebuilt as Atoms objects;
        the legacy hand-written dict format is also supported.
        If the population is not empty, confirm=True is required to overwrite
        the existing data.

        Parameters
        ----------
        filename : str | pathlib.Path
            Full path of the JSON file.
        confirm : bool, optional
            Whether to confirm overwriting the current population data.
            Default False.
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
                data: dict = read_json(f) # type: ignore
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
                # ASE JSON snapshot: energy/point group are in info, forces are in
                # arrays; adopt directly.
                self._clusters.append(c_data)
                continue

            try:
                # Compatible with the legacy hand-written dict format.
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


class Clusters(MultiAtoms):
    """A cluster container with unique chemical formulas.

    1. Chemical formulas must be unique.
    2. Clusters are ordered by the atom count of a specified element.
    """

    def __setitem__(self, index: int | slice, value: Atoms | Iterable[Atoms]) -> None:
        raise TypeError(
            "Clusters does not support item assignment; use add_cluster() or merge()"
        )

    def sort(self, element: str = "auto", reverse: bool = False) -> None:
        """Sort clusters in place by the count of a given element.

        Parameters
        ----------
        element : str, optional
            Element symbol to count. If ``"auto"``, use the element of the
            first atom in the first cluster. Default is ``"auto"``.
        reverse : bool, optional
            If True, sort in descending order. Default is False.

        Raises
        ------
        ValueError
            If the cluster list is empty or the element does not exist in any
            cluster.
        """
        if not self._clusters:
            raise ValueError("Cannot sort an empty cluster list")
        if element == "auto":
            element = self._clusters[0][0].symbol
        if not any(
            element in cluster.get_chemical_symbols()
            for cluster in self._clusters
        ):
            raise ValueError(f"Element {element} does not exist in any cluster")
        self._clusters.sort(
            key=lambda cluster: cluster.get_chemical_symbols().count(element),
            reverse=reverse,
        )

    def add_cluster(
        self,
        cluster: Atoms,
        replace: bool = False,
        sort: bool = True,
    ) -> int:
        """Add a cluster and enforce unique chemical formulas.

        Parameters
        ----------
        cluster : Atoms
            Cluster to add. The stored object is an ``Atoms.copy()``.
        replace : bool, optional
            If True, replace the existing cluster with the same chemical
            formula. Default False.
        sort : bool, optional
            If True, sort the clusters after adding. Default True.

        Returns
        -------
        int
            Index of the cluster with the same chemical formula after adding
            or replacing it.
        """
        formula = cluster.get_chemical_formula()
        symbols = self.symbols
        try:
            index = symbols.index(formula)
        except ValueError:
            super().add_cluster(cluster)
            if sort:
                self.sort()
            return self.symbols.index(formula)

        if not replace:
            print(f"Cluster formula should be unique in Clusters, formula {formula} already exists")
            return index
        self._clusters[index] = cluster.copy()
        if sort:
            self.sort()
        return self.symbols.index(formula)

    def merge(self, clusters: Self) -> None:
        """Merge all clusters from another container.

        Parameters
        ----------
        clusters : Self
            Clusters to add one by one.
        """
        for cluster in clusters:
            index = self.add_cluster(cluster, replace=False, sort=False)
            existing = self._clusters[index]
            energy_existing = existing.info.get('energy')
            energy_cluster = cluster.info.get('energy')
            if energy_cluster is not None and (
                energy_existing is None
                or float(energy_cluster) < float(energy_existing)
            ):
                self.add_cluster(cluster, replace=True)