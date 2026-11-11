"""Built-in ASE calculators shipped with sansde."""

from __future__ import annotations

from typing import Any, ClassVar, TypeAlias

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

__all__ = ["ZBLCalculator", "zbl_energy", "zbl_force"]

RadialValue: TypeAlias = float | npt.NDArray[np.float64]

_PREFACTOR = 14.399645481316385
"""Coulomb prefactor alpha * hbar * c in eV * Angstrom."""

_SCREENING_TERMS: tuple[tuple[float, float], ...] = (
    (0.18175, 3.19980),
    (0.50986, 0.94229),
    (0.28022, 0.40290),
    (0.02817, 0.20162),
)
"""Coefficients and exponents of the universal ZBL screening function."""

_SMOOTH_ONSET_FRACTION = 0.66
"""Onset of the smooth cutoff relative to the cutoff, as in ASE LennardJones."""


def _screening(x: RadialValue) -> RadialValue:
    """Evaluate the ZBL screening function phi(x)."""
    return sum(
        coefficient * np.exp(-exponent * x)
        for coefficient, exponent in _SCREENING_TERMS
    )


def _screening_derivative(x: RadialValue) -> RadialValue:
    """Evaluate dphi/dx of the ZBL screening function."""
    return sum(
        -coefficient * exponent * np.exp(-exponent * x)
        for coefficient, exponent in _SCREENING_TERMS
    )


def _screening_length(z_i: RadialValue, z_j: RadialValue) -> RadialValue:
    """Return the Firsov screening length a_ij in Angstrom."""
    return 0.46850 / (np.asarray(z_i) ** 0.23 + np.asarray(z_j) ** 0.23)


def zbl_energy(z_i: RadialValue, z_j: RadialValue, r: RadialValue) -> RadialValue:
    """Evaluate the unshifted Ziegler-Biersack-Littmark pair energy.

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
        np.float64(471.25...)
    """
    return (
        _PREFACTOR
        * z_i
        * z_j
        / r
        * _screening(np.asarray(r) / _screening_length(z_i, z_j))
    )


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
        np.float64(2124.37...)
    """
    screening_length = _screening_length(z_i, z_j)
    x = np.asarray(r) / screening_length
    return (
        -_PREFACTOR
        * z_i
        * z_j
        * (_screening_derivative(x) / (screening_length * r) - _screening(x) / r**2)
    )


def _cutoff_switch(r: RadialValue, cutoff: float, onset: float) -> RadialValue:
    """Evaluate the quintic switch that falls from 1 at onset to 0 at cutoff."""
    t = np.clip((np.asarray(r) - onset) / (cutoff - onset), 0.0, 1.0)
    return 1.0 - t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def _cutoff_switch_derivative(
    r: RadialValue, cutoff: float, onset: float
) -> RadialValue:
    """Evaluate d(switch)/dr, zero outside the transition region."""
    t = np.clip((np.asarray(r) - onset) / (cutoff - onset), 0.0, 1.0)
    return -30.0 * t**2 * (1.0 - t) ** 2 / (cutoff - onset)


class ZBLCalculator(Calculator):
    """ASE calculator for the ZBL short-range nuclear repulsion.

    With ``cutoff`` set, the energy is the sum of the ZBL pair energies over
    all pairs inside the cutoff. Two truncation modes are available:

    * hard cutoff (``smooth=False``): plain ZBL pair energies, truncated when a
      pair leaves the neighbor list,

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

        if self.cutoff is None:
            if atoms.pbc.any():
                raise ValueError(
                    "ZBLCalculator without a cutoff cannot sum a periodic cell"
                )
            indices_i, indices_j = np.triu_indices(len(atoms), k=1)
            vectors = positions[indices_j] - positions[indices_i]
        else:
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
            cell = atoms.cell.array
            vectors = positions[indices_j] + offsets @ cell - positions[indices_i]

        distances = np.linalg.norm(vectors, axis=1)
        z_i = numbers[indices_i]
        z_j = numbers[indices_j]

        energy = 0.0
        forces = np.zeros_like(positions)
        if len(distances) > 0:
            pair_energy = zbl_energy(z_i, z_j, distances)
            radial = zbl_force(z_i, z_j, distances)
            if self.cutoff is not None and self.smooth:
                switch = _cutoff_switch(distances, self.cutoff, self.smooth_onset)
                switch_derivative = _cutoff_switch_derivative(
                    distances, self.cutoff, self.smooth_onset
                )
                radial = radial * switch - pair_energy * switch_derivative
                pair_energy = pair_energy * switch
            energy = float(np.sum(pair_energy))
            pair_forces = radial[:, None] * vectors / distances[:, None]
            np.add.at(forces, indices_j, pair_forces)
            np.add.at(forces, indices_i, -pair_forces)

        self.results["energy"] = energy
        self.results["forces"] = forces
