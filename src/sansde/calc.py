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


class ZBLCalculator(Calculator):
    """ASE calculator for the ZBL short-range nuclear repulsion.

    The energy is the sum of the ZBL pair energies over all pairs inside the
    neighbor list, shifted by the ZBL energy at the interaction range so that
    the total energy is continuous when a pair enters or leaves the list:

    .. math::
        E = \\sum_{i<j} \\left[ E_{ZBL}(r_{ij}) - E_{ZBL}(r_{c}) \\right],
        \\quad r_c = \\text{cutoff} + 2 \\cdot \\text{skin}.

    The unshifted ZBL energy is still finite at ``r_c``, so without the shift
    the calculator would add a step of tens of eV for heavy elements. Forces
    are analytic derivatives of the pair energies; the constant shift does not
    contribute to them. Periodic cells are handled through the neighbor list
    offsets, so the calculator also works for isolated clusters with a zero
    cell.

    Args:
        cutoff: Nominal interaction cutoff in Angstrom.
        skin: Extra neighbor search distance in Angstrom. ASE adds ``skin`` to
            every atomic radius, hence the pair range ``cutoff + 2 * skin``.
            Default 0.0.
        **kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`.

    Attributes:
        cutoff: Nominal interaction cutoff in Angstrom.
        skin: Extra neighbor search distance in Angstrom.
        interaction_range: Pair distance reach ``cutoff + 2 * skin``.

    Raises:
        ValueError: If ``cutoff`` is not positive or ``skin`` is negative.

    Examples:
        >>> from ase import Atoms
        >>> from sansde.calc import ZBLCalculator
        >>> atoms = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> atoms.calc = ZBLCalculator(cutoff=1.5, skin=0.0)
        >>> bool(atoms.get_potential_energy() > 0.0)
        True
        >>> atoms.get_forces().sum(axis=0)
        array([0., 0., 0.])
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        cutoff: float,
        skin: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Store the cutoff settings and initialize the ASE calculator."""
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.interaction_range = self.cutoff + 2.0 * self.skin

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate the shifted ZBL energy and forces.

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
        numbers = atoms.numbers.astype(np.float64)
        positions = atoms.positions
        cell = atoms.cell.array

        indices_i, indices_j, offsets = neighbor_list(
            "ijS",
            atoms,
            self.interaction_range,
            self_interaction=False,
        )
        # ASE returns every pair in both directions; keep one direction only.
        keep = indices_i < indices_j
        indices_i = indices_i[keep]
        indices_j = indices_j[keep]
        offsets = offsets[keep]
        vectors = positions[indices_j] + offsets @ cell - positions[indices_i]
        distances = np.linalg.norm(vectors, axis=1)
        z_i = numbers[indices_i]
        z_j = numbers[indices_j]

        energy = 0.0
        forces = np.zeros_like(positions)
        if len(distances) > 0:
            shift = zbl_energy(z_i, z_j, self.interaction_range)
            energy = float(np.sum(zbl_energy(z_i, z_j, distances) - shift))
            radial = zbl_force(z_i, z_j, distances)
            pair_forces = radial[:, None] * vectors / distances[:, None]
            np.add.at(forces, indices_j, pair_forces)
            np.add.at(forces, indices_i, -pair_forces)

        self.results["energy"] = energy
        self.results["forces"] = forces
