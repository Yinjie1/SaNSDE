"""Built-in ASE calculators shipped with sansde."""

from __future__ import annotations

from typing import Any, ClassVar, TypeAlias

import numba
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

__all__ = ["GuptaCalculator", "ZBLCalculator", "zbl_energy", "zbl_force"]

RadialValue: TypeAlias = float | npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Shared pair helpers
# ---------------------------------------------------------------------------

_SMOOTH_ONSET_FRACTION = 0.66
"""Onset of the smooth cutoff relative to the cutoff, as in ASE LennardJones."""


@numba.njit
def _cutoff_switch(r: float, cutoff: float, onset: float) -> float:
    """Evaluate the quintic switch that falls from 1 at onset to 0 at cutoff."""
    t = (r - onset) / (cutoff - onset)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return 1.0 - t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


@numba.njit
def _cutoff_switch_derivative(r: float, cutoff: float, onset: float) -> float:
    """Evaluate d(switch)/dr, zero outside the transition region."""
    t = (r - onset) / (cutoff - onset)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return -30.0 * t**2 * (1.0 - t) ** 2 / (cutoff - onset)


@numba.njit
def _add_pair_force(
    forces: npt.NDArray[np.float64],
    index_i: int,
    index_j: int,
    force_x: float,
    force_y: float,
    force_z: float,
) -> None:
    """Add the force of one pair to both atoms in place."""
    forces[index_j, 0] += force_x
    forces[index_j, 1] += force_y
    forces[index_j, 2] += force_z
    forces[index_i, 0] -= force_x
    forces[index_i, 1] -= force_y
    forces[index_i, 2] -= force_z


# ---------------------------------------------------------------------------
# ZBL screened Coulomb repulsion
# ---------------------------------------------------------------------------

_PREFACTOR = 14.399645481316385
"""Coulomb prefactor alpha * hbar * c in eV * Angstrom."""


@numba.njit
def _screening(x: RadialValue) -> RadialValue:
    """Evaluate the ZBL screening function phi(x)."""
    return (
        0.18175 * np.exp(-3.19980 * x)
        + 0.50986 * np.exp(-0.94229 * x)
        + 0.28022 * np.exp(-0.40290 * x)
        + 0.02817 * np.exp(-0.20162 * x)
    )


@numba.njit
def _screening_derivative(x: RadialValue) -> RadialValue:
    """Evaluate dphi/dx of the ZBL screening function."""
    return (
        -3.19980 * 0.18175 * np.exp(-3.19980 * x)
        - 0.94229 * 0.50986 * np.exp(-0.94229 * x)
        - 0.40290 * 0.28022 * np.exp(-0.40290 * x)
        - 0.20162 * 0.02817 * np.exp(-0.20162 * x)
    )


@numba.njit
def _screening_length(z_i: RadialValue, z_j: RadialValue) -> RadialValue:
    """Return the Firsov screening length a_ij in Angstrom."""
    return 0.46850 / (z_i**0.23 + z_j**0.23)


@numba.njit
def zbl_energy(z_i: RadialValue, z_j: RadialValue, r: RadialValue) -> RadialValue:
    """Evaluate the Ziegler-Biersack-Littmark pair energy.

    The ZBL energy is the screened Coulomb repulsion between two nuclei,

    .. math::
        E(r) = \\frac{1}{4\\pi\\epsilon_0}
               \\frac{Z_i Z_j}{r} \\phi\\!\\left(\\frac{r}{a_{ij}}\\right),

    with the universal screening function ``phi`` and the Firsov screening
    length ``a_ij``. It is only relevant for internuclear distances far below
    chemical bond lengths.

    Args:
        z_i: Atomic number of the first atom.
        z_j: Atomic number of the second atom.
        r: Internuclear distance in Angstrom, scalar or array.

    Returns:
        Pair energy in eV, with the shape of ``r``.

    Examples:
        >>> zbl_energy(79, 79, 1.0)
        471.2526265793471
    """
    return _PREFACTOR * z_i * z_j / r * _screening(r / _screening_length(z_i, z_j))


@numba.njit
def zbl_force(z_i: RadialValue, z_j: RadialValue, r: RadialValue) -> RadialValue:
    """Evaluate the radial Ziegler-Biersack-Littmark force.

    The returned magnitude is ``-dE/dr``, so a positive value pushes the two
    atoms apart.

    Args:
        z_i: Atomic number of the first atom.
        z_j: Atomic number of the second atom.
        r: Internuclear distance in Angstrom, scalar or array.

    Returns:
        Radial force in eV/Angstrom, with the shape of ``r``.

    Examples:
        >>> zbl_force(79, 79, 1.0)
        2124.3717461872957
    """
    screening_length = _screening_length(z_i, z_j)
    x = r / screening_length
    return (
        -_PREFACTOR
        * z_i
        * z_j
        * (_screening_derivative(x) / (screening_length * r) - _screening(x) / r**2)
    )


@numba.njit
def _pair_energy_force(
    z_i: float,
    z_j: float,
    distance: float,
    cutoff: float,
    onset: float,
    smooth: bool,
) -> tuple[float, float]:
    """Return the pair energy and radial force, optionally switched."""
    pair_energy = zbl_energy(z_i, z_j, distance)
    radial = zbl_force(z_i, z_j, distance)
    if smooth:
        switch = _cutoff_switch(distance, cutoff, onset)
        switch_derivative = _cutoff_switch_derivative(distance, cutoff, onset)
        radial = radial * switch - pair_energy * switch_derivative
        pair_energy = pair_energy * switch
    return pair_energy, radial


@numba.njit
def _accumulate_pairs(
    vectors: npt.NDArray[np.float64],
    distances: npt.NDArray[np.float64],
    z_i: npt.NDArray[np.float64],
    z_j: npt.NDArray[np.float64],
    indices_i: npt.NDArray[np.int64],
    indices_j: npt.NDArray[np.int64],
    forces: npt.NDArray[np.float64],
    cutoff: float,
    onset: float,
    smooth: bool,
) -> float:
    """Sum the pair energies of listed pairs and add their forces in place."""
    energy = 0.0
    for pair in range(distances.shape[0]):
        distance = distances[pair]
        pair_energy, radial = _pair_energy_force(
            z_i[pair], z_j[pair], distance, cutoff, onset, smooth
        )
        energy += pair_energy
        scale = radial / distance
        _add_pair_force(
            forces,
            indices_i[pair],
            indices_j[pair],
            scale * vectors[pair, 0],
            scale * vectors[pair, 1],
            scale * vectors[pair, 2],
        )
    return energy


@numba.njit
def _accumulate_all_pairs(
    positions: npt.NDArray[np.float64],
    numbers: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    cutoff: float,
    onset: float,
    smooth: bool,
) -> float:
    """Sum all pair energies inside the cutoff and add their forces in place."""
    energy = 0.0
    cutoff_squared = cutoff * cutoff
    n_atoms = positions.shape[0]
    for index_i in range(n_atoms - 1):
        x_i = positions[index_i, 0]
        y_i = positions[index_i, 1]
        z_i = positions[index_i, 2]
        for index_j in range(index_i + 1, n_atoms):
            vector_x = positions[index_j, 0] - x_i
            vector_y = positions[index_j, 1] - y_i
            vector_z = positions[index_j, 2] - z_i
            squared_distance = (
                vector_x * vector_x + vector_y * vector_y + vector_z * vector_z
            )
            if squared_distance >= cutoff_squared:
                continue
            distance = np.sqrt(squared_distance)
            pair_energy, radial = _pair_energy_force(
                numbers[index_i],
                numbers[index_j],
                distance,
                cutoff,
                onset,
                smooth,
            )
            energy += pair_energy
            scale = radial / distance
            _add_pair_force(
                forces,
                index_i,
                index_j,
                scale * vector_x,
                scale * vector_y,
                scale * vector_z,
            )
    return energy


class ZBLCalculator(Calculator):
    """ASE calculator for the ZBL short-range nuclear repulsion.

    With ``cutoff`` set, the energy is the sum of the ZBL pair energies over
    all pairs inside the cutoff. Two truncation modes are available:

    * hard cutoff (``smooth=False``): plain ZBL pair energies, truncated at the
      cutoff,

      .. math::
          E = \\sum_{i<j} E_{ZBL}(r_{ij}).

      This mode reproduces ``hiphive.calculators.zbl.ZBLCalculator`` exactly:
      the energy jumps by ``E_ZBL(cutoff)`` at the cutoff, while the forces jump
      to zero.
    * smooth cutoff (``smooth=True``): every pair energy is multiplied by the
      quintic switch ``s(r)``,

      .. math::
          E = \\sum_{i<j} E_{ZBL}(r_{ij})\\, s(r_{ij}),
          \\qquad
          s(r) = 1 - t^3 (10 - 15 t + 6 t^2),
          \\quad
          t = \\frac{r - r_{on}}{r_c - r_{on}},

      with ``t`` clipped to ``[0, 1]`` and ``r_on = 0.66 * cutoff``. Both
      energy and forces are continuous at the cutoff.

    With ``cutoff=None`` all pairs are summed with the bare ZBL energy and
    ``smooth`` has no effect. This mode requires a non-periodic system, since a
    periodic lattice has no finite pair sum.

    Args:
        cutoff: Interaction cutoff in Angstrom. ``None`` sums all pairs.
        smooth: Multiply the pair energies by the smooth switch at the cutoff.
            Only used when ``cutoff`` is set. Default False.
        **kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`.

    Attributes:
        cutoff: Interaction cutoff in Angstrom, ``None`` when unused.
        smooth: Whether the smooth switch is requested. It is only applied
            when ``cutoff`` is set.
        smooth_onset: Onset radius ``0.66 * cutoff`` of the smooth switch, or
            ``None`` when no cutoff is set.

    Raises:
        ValueError: If ``cutoff`` is not positive, or if a periodic system is
            evaluated without a cutoff.

    Examples:
        >>> from ase import Atoms
        >>> from sansde.calc import ZBLCalculator
        >>> atoms = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> atoms.calc = ZBLCalculator(cutoff=1.5)
        >>> bool(atoms.get_potential_energy() > 0.0)
        True
        >>> atoms.get_forces().sum(axis=0)
        array([0., 0., 0.])
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        cutoff: float | None = None,
        smooth: bool = False,
        **kwargs: Any,
    ) -> None:
        """Store the cutoff settings and initialize the ASE calculator."""
        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        super().__init__(**kwargs)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.smooth = bool(smooth)
        self.smooth_onset = (
            None if self.cutoff is None else _SMOOTH_ONSET_FRACTION * self.cutoff
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate the ZBL energy and forces.

        Args:
            atoms: Structure to evaluate. Defaults to the attached atoms.
            properties: Requested properties. Default ``["energy"]``.
            system_changes: ASE change flags, unused here.
        """
        if properties is None:
            properties = ["energy"]
        Calculator.calculate(self, atoms, properties, system_changes)

        if "energy" not in properties and "forces" not in properties:
            return

        atoms = self.atoms
        positions = atoms.positions
        numbers = atoms.numbers.astype(np.float64)
        cutoff = np.inf if self.cutoff is None else self.cutoff
        onset = 0.0 if self.smooth_onset is None else self.smooth_onset
        smooth = self.smooth and self.cutoff is not None

        energy = 0.0
        forces = np.zeros_like(positions)

        if not atoms.pbc.any():
            energy = float(
                _accumulate_all_pairs(positions, numbers, forces, cutoff, onset, smooth)
            )
        else:
            if self.cutoff is None:
                raise ValueError(
                    "ZBLCalculator without a cutoff cannot sum a periodic cell"
                )
            indices_i, indices_j, offsets = neighbor_list(
                "ijS",
                atoms,
                self.cutoff,
                self_interaction=False,
            )
            # ASE returns every pair in both directions; keep one direction.
            keep = indices_i < indices_j
            indices_i = indices_i[keep]
            indices_j = indices_j[keep]
            offsets = offsets[keep]
            vectors = positions[indices_j] + offsets @ atoms.cell.array
            vectors -= positions[indices_i]
            distances = np.linalg.norm(vectors, axis=1)
            if len(distances) > 0:
                energy = float(
                    _accumulate_pairs(
                        vectors,
                        distances,
                        numbers[indices_i],
                        numbers[indices_j],
                        indices_i,
                        indices_j,
                        forces,
                        cutoff,
                        onset,
                        smooth,
                    )
                )

        self.results["energy"] = energy
        self.results["forces"] = forces



# ---------------------------------------------------------------------------
# Gupta potential (asap3 backend)
# ---------------------------------------------------------------------------


class GuptaCalculator(Calculator):
    """ASE calculator for the Gupta potential, backed by asap3.

    The energy and forces are computed by ``asap3.Gupta``. asap3 requires a
    valid unit cell even for isolated clusters, so non-periodic inputs are
    evaluated inside a cubic cell of ``cell`` Angstrom with ``pbc`` false.
    Periodic inputs are passed through unchanged.

    asap3 expects to be attached through ``atoms.calc``, so the underlying
    potential is bound to one persistent Atoms object and only its positions
    are updated between calls.

    Args:
        parameters: Element symbol or symbol pair mapped to
            ``[p, q, A, xi, r0]``, the parameter order of
            ``asap3.Internal.BuiltinPotentials.RGL``. Mixed pairs need an
            explicit ``(symbol_i, symbol_j)`` entry.
        cell: Edge length of the cubic cell used for non-periodic inputs in
            Angstrom. Default 60.0.
        **kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`.

    Attributes:
        cell: Edge length of the cubic cell used for non-periodic inputs.
        potential: The underlying ``asap3.Gupta`` instance.

    Raises:
        ImportError: If asap3 is not installed.
        ValueError: If ``cell`` is not positive.

    Examples:
        >>> from ase import Atoms
        >>> from sansde.calc import GuptaCalculator
        >>> atoms = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]])
        >>> atoms.calc = GuptaCalculator({"Au": [10.229, 4.036, 0.2061, 1.790, 2.884]})
        >>> bool(atoms.get_potential_energy() < 0.0)
        True
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        parameters: dict[object, list[float] | tuple[float, ...]],
        cell: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Create the asap3 potential and prepare the evaluation system."""
        try:
            from asap3 import Gupta
        except ImportError as e:
            raise ImportError("GuptaCalculator requires the asap3 package") from e
        if cell <= 0:
            raise ValueError(f"cell must be positive, got {cell}")
        super().__init__(**kwargs)
        self.cell = float(cell)
        self.potential = Gupta(parameters)
        self._system: Atoms | None = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate the Gupta energy and forces through asap3.

        Args:
            atoms: Structure to evaluate. Defaults to the attached atoms.
            properties: Requested properties. Default ``["energy"]``.
            system_changes: ASE change flags, unused here.
        """
        if properties is None:
            properties = ["energy"]
        Calculator.calculate(self, atoms, properties, system_changes)

        if "energy" not in properties and "forces" not in properties:
            return

        atoms = self.atoms
        rebuild = (
            self._system is None
            or len(self._system) != len(atoms)
            or not np.array_equal(self._system.numbers, atoms.numbers)
            or not np.array_equal(self._system.pbc, atoms.pbc)
        )
        if rebuild:
            self._system = atoms.copy()
            if not atoms.pbc.any():
                self._system.set_cell([self.cell, self.cell, self.cell])
            self._system.calc = self.potential
        else:
            self._system.positions = atoms.positions
            if atoms.pbc.any():
                self._system.set_cell(atoms.cell)

        self.results["energy"] = float(self._system.get_potential_energy())
        self.results["forces"] = np.asarray(
            self._system.get_forces(), dtype=np.float64
        )
