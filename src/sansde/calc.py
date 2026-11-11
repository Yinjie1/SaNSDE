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
# Gupta (RGL) many-body potential
# ---------------------------------------------------------------------------


def _gupta_cutoff_polynomial(
    amplitude: float, decay: float, width: float
) -> tuple[float, float, float]:
    """Return the cubic, quartic and quintic terms of the RGL cutoff.

    The polynomial ``c3 d^3 + c4 d^4 + c5 d^5`` with ``d = r - r_end`` matches
    the analytic pair function and its first two derivatives at
    ``r_start = r_end - width``, and vanishes with zero slope and curvature at
    ``r_end``.

    Args:
        amplitude: Value of the pair function at ``r_start``.
        decay: Decay rate of the pair function per Angstrom.
        width: Width of the cutoff region in Angstrom.

    Returns:
        Coefficients ``(c3, c4, c5)``.
    """
    factor = -amplitude / width**3
    scaled = decay * width
    cubic = (20.0 * factor - 8.0 * factor * scaled + factor * scaled**2) / 2.0
    quartic = (15.0 * factor - 7.0 * factor * scaled + factor * scaled**2) / width
    quintic = (
        12.0 * factor - 6.0 * factor * scaled + factor * scaled**2
    ) / (2.0 * width**2)
    return cubic, quartic, quintic


def _gupta_arrays(
    parameters: dict[object, tuple[float, float, float, float, float]],
    cutoff: float,
    cutoff_width: float,
) -> dict[str, object]:
    """Build the element mapping and per-pair parameter matrices.

    Args:
        parameters: Element symbol or symbol pair mapped to
            ``[p, q, A, xi, r0]``, the parameter order of
            ``asap3.Internal.BuiltinPotentials.RGL``.
        cutoff: Start of the cutoff region in units of the equilibrium
            nearest-neighbor distance.
        cutoff_width: Width of the cutoff region in units of the same distance.

    Returns:
        Mapping with the element symbols, the symbol-to-index lookup, the
        per-pair matrices and the cutoff radii.

    Raises:
        TypeError: If ``parameters`` is not a dictionary.
        ValueError: If a parameter entry is malformed, a pair is missing, or the
            parameters give an unreasonable nearest-neighbor distance.
    """
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary")

    symbols = sorted(
        {
            symbol
            for key in parameters
            for symbol in (key if isinstance(key, tuple) else (key,))
        }
    )
    index = {symbol: position for position, symbol in enumerate(symbols)}
    n_elements = len(symbols)
    p = np.zeros((n_elements, n_elements))
    q = np.zeros((n_elements, n_elements))
    a = np.zeros((n_elements, n_elements))
    xi = np.zeros((n_elements, n_elements))
    r0 = np.zeros((n_elements, n_elements))
    defined = np.zeros((n_elements, n_elements), dtype=bool)

    for key, values in parameters.items():
        if len(values) != 5:
            raise ValueError(f"parameters for {key} must hold [p, q, A, xi, r0]")
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(f"pair key must hold two symbols, got {key}")
            element_i, element_j = index[key[0]], index[key[1]]
        else:
            element_i = element_j = index[key]
        p[element_i, element_j] = float(values[0])
        q[element_i, element_j] = float(values[1])
        a[element_i, element_j] = float(values[2])
        xi[element_i, element_j] = float(values[3])
        r0[element_i, element_j] = float(values[4])
        defined[element_i, element_j] = True
        defined[element_j, element_i] = True

    if not defined.all():
        missing = [
            (symbols[i], symbols[j])
            for i in range(n_elements)
            for j in range(i, n_elements)
            if not defined[i, j]
        ]
        raise ValueError(f"missing Gupta parameters for {missing}")

    nearest = (np.log(np.sqrt(12.0) * a * p / (xi * q)) / (p - q) + 1.0) * r0
    if not np.all((nearest > 1.0) & (nearest < 10.0)):
        raise ValueError(
            "unreasonable Gupta parameters, nearest-neighbor distance "
            f"span [{nearest.min():.3f}, {nearest.max():.3f}]"
        )

    r_start = cutoff * float(nearest.max())
    width = cutoff_width * float(nearest.max())
    r_end = r_start + width

    p3 = np.zeros((n_elements, n_elements))
    p4 = np.zeros((n_elements, n_elements))
    p5 = np.zeros((n_elements, n_elements))
    q3 = np.zeros((n_elements, n_elements))
    q4 = np.zeros((n_elements, n_elements))
    q5 = np.zeros((n_elements, n_elements))
    for element_i in range(n_elements):
        for element_j in range(n_elements):
            scaled_start = r_start / r0[element_i, element_j] - 1.0
            p3[element_i, element_j], p4[element_i, element_j], p5[
                element_i, element_j
            ] = _gupta_cutoff_polynomial(
                a[element_i, element_j] * np.exp(-p[element_i, element_j] * scaled_start),
                p[element_i, element_j] / r0[element_i, element_j],
                width,
            )
            q3[element_i, element_j], q4[element_i, element_j], q5[
                element_i, element_j
            ] = _gupta_cutoff_polynomial(
                xi[element_i, element_j] * np.exp(-q[element_i, element_j] * scaled_start),
                q[element_i, element_j] / r0[element_i, element_j],
                width,
            )

    return {
        "symbols": symbols,
        "index": index,
        "p": p,
        "q": q,
        "a": a,
        "qsi2": xi**2,
        "r0": r0,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "r_start": r_start,
        "r_end": r_end,
    }


@numba.njit
def _gupta_accumulate(
    positions: npt.NDArray[np.float64],
    elements: npt.NDArray[np.int64],
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    qsi2: npt.NDArray[np.float64],
    r0: npt.NDArray[np.float64],
    p3: npt.NDArray[np.float64],
    p4: npt.NDArray[np.float64],
    p5: npt.NDArray[np.float64],
    q3: npt.NDArray[np.float64],
    q4: npt.NDArray[np.float64],
    q5: npt.NDArray[np.float64],
    r_start: float,
    r_end: float,
    forces: npt.NDArray[np.float64],
) -> float:
    """Sum the Gupta energy and add the forces in place.

    The first pass accumulates the repulsive sum and the density sum of every
    atom, the second pass turns them into forces.
    """
    n_atoms = positions.shape[0]
    sigma_p = np.zeros(n_atoms)
    sigma_q = np.zeros(n_atoms)
    cutoff_squared = r_end * r_end

    for index_i in range(n_atoms - 1):
        x_i = positions[index_i, 0]
        y_i = positions[index_i, 1]
        z_i = positions[index_i, 2]
        element_i = elements[index_i]
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
            element_j = elements[index_j]
            if distance < r_start:
                scaled = distance / r0[element_i, element_j] - 1.0
                pair = a[element_i, element_j] * np.exp(
                    -p[element_i, element_j] * scaled
                )
                density = qsi2[element_i, element_j] * np.exp(
                    -2.0 * q[element_i, element_j] * scaled
                )
            else:
                delta = distance - r_end
                delta2 = delta * delta
                delta3 = delta2 * delta
                pair = (
                    p3[element_i, element_j] * delta3
                    + p4[element_i, element_j] * delta3 * delta
                    + p5[element_i, element_j] * delta3 * delta2
                )
                amplitude = (
                    q3[element_i, element_j] * delta3
                    + q4[element_i, element_j] * delta3 * delta
                    + q5[element_i, element_j] * delta3 * delta2
                )
                density = amplitude * amplitude
            sigma_p[index_i] += pair
            sigma_p[index_j] += pair
            sigma_q[index_i] += density
            sigma_q[index_j] += density

    energy = 0.0
    for index_i in range(n_atoms):
        energy += sigma_p[index_i] - np.sqrt(sigma_q[index_i])

    for index_i in range(n_atoms - 1):
        x_i = positions[index_i, 0]
        y_i = positions[index_i, 1]
        z_i = positions[index_i, 2]
        element_i = elements[index_i]
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
            element_j = elements[index_j]
            if distance < r_start:
                scaled = distance / r0[element_i, element_j] - 1.0
                pair_slope = (
                    -2.0
                    * a[element_i, element_j]
                    * p[element_i, element_j]
                    * np.exp(-p[element_i, element_j] * scaled)
                    / r0[element_i, element_j]
                )
                density_slope = (
                    -qsi2[element_i, element_j]
                    * q[element_i, element_j]
                    * np.exp(-2.0 * q[element_i, element_j] * scaled)
                    / r0[element_i, element_j]
                )
            else:
                delta = distance - r_end
                delta2 = delta * delta
                delta3 = delta2 * delta
                pair_slope = 2.0 * (
                    5.0 * p5[element_i, element_j] * delta3 * delta
                    + 4.0 * p4[element_i, element_j] * delta3
                    + 3.0 * p3[element_i, element_j] * delta2
                )
                amplitude = (
                    q3[element_i, element_j] * delta3
                    + q4[element_i, element_j] * delta3 * delta
                    + q5[element_i, element_j] * delta3 * delta2
                )
                amplitude_slope = (
                    5.0 * q5[element_i, element_j] * delta3 * delta
                    + 4.0 * q4[element_i, element_j] * delta3
                    + 3.0 * q3[element_i, element_j] * delta2
                )
                density_slope = amplitude * amplitude_slope

            coefficient = (
                pair_slope
                - density_slope
                * (
                    1.0 / np.sqrt(sigma_q[index_i])
                    + 1.0 / np.sqrt(sigma_q[index_j])
                )
            ) / distance
            force_x = coefficient * vector_x
            force_y = coefficient * vector_y
            force_z = coefficient * vector_z
            forces[index_i, 0] += force_x
            forces[index_i, 1] += force_y
            forces[index_i, 2] += force_z
            forces[index_j, 0] -= force_x
            forces[index_j, 1] -= force_y
            forces[index_j, 2] -= force_z

    return energy


class GuptaCalculator(Calculator):
    """ASE calculator for the Gupta (RGL) many-body potential.

    The energy is the second-moment tight-binding form used by the RGL
    implementation of asap3,

    .. math::
        E = \\sum_i \\left( \\sum_{j \\neq i} V_{ij} - \\sqrt{\\rho_i} \\right),
        \\qquad
        V_{ij} = A_{ij} e^{-p_{ij} (r_{ij}/r_{0,ij} - 1)},
        \\quad
        \\rho_i = \\sum_{j \\neq i} \\xi_{ij}^2
        e^{-2 q_{ij} (r_{ij}/r_{0,ij} - 1)}.

    The pair functions are analytic below ``r_start`` and replaced by a quintic
    polynomial between ``r_start`` and ``r_end``. The polynomial matches the
    value, slope and curvature at ``r_start`` and vanishes with zero slope and
    curvature at ``r_end``, so energy and forces are continuous. The density
    cutoff polynomial acts on the density amplitude ``xi e^{-q (r/r0 - 1)}``
    and is squared, as in asap3.

    ``r_start`` and ``r_end`` follow the asap3 convention: the equilibrium
    nearest-neighbor distance of each element pair is

    .. math::
        a_1 = r_0 \\left[ 1 + \\frac{\\ln(\\sqrt{12} A p / (\\xi q))}{p - q}
        \\right],

    ``r_start = cutoff * a_1,max`` and ``r_end = r_start + cutoff_width *
    a_1,max``.

    Args:
        parameters: Element symbol or symbol pair mapped to
            ``[p, q, A, xi, r0]``, the parameter order of
            ``asap3.Internal.BuiltinPotentials.RGL``. Mixed pairs need an
            explicit ``(symbol_i, symbol_j)`` entry.
        cutoff: Start of the cutoff region in units of the equilibrium
            nearest-neighbor distance. Default 1.73, as in asap3.
        cutoff_width: Width of the cutoff region in the same units. Default
            0.15, as in asap3.
        **kwargs: Forwarded to :class:`ase.calculators.calculator.Calculator`.

    Attributes:
        symbols: Element symbols covered by the parameters.
        cutoff_start: Start radius of the cutoff region in Angstrom.
        cutoff_end: End radius of the cutoff region in Angstrom.

    Raises:
        ValueError: If the parameters are malformed or incomplete, if the
            structure holds an element without parameters, or if it is
            periodic.

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
        parameters: dict[object, tuple[float, float, float, float, float]],
        cutoff: float = 1.73,
        cutoff_width: float = 0.15,
        **kwargs: Any,
    ) -> None:
        """Store the Gupta parameters and initialize the ASE calculator."""
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if cutoff_width <= 0:
            raise ValueError(f"cutoff_width must be positive, got {cutoff_width}")
        super().__init__(**kwargs)
        data = _gupta_arrays(parameters, float(cutoff), float(cutoff_width))
        self.symbols: list[str] = data["symbols"]  # type: ignore[assignment]
        self.cutoff_start: float = data["r_start"]  # type: ignore[assignment]
        self.cutoff_end: float = data["r_end"]  # type: ignore[assignment]
        self._index: dict[str, int] = data["index"]  # type: ignore[assignment]
        self._p = data["p"]
        self._q = data["q"]
        self._a = data["a"]
        self._qsi2 = data["qsi2"]
        self._r0 = data["r0"]
        self._p3 = data["p3"]
        self._p4 = data["p4"]
        self._p5 = data["p5"]
        self._q3 = data["q3"]
        self._q4 = data["q4"]
        self._q5 = data["q5"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate the Gupta energy and forces.

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
        if atoms.pbc.any():
            raise ValueError("GuptaCalculator only supports non-periodic systems")

        try:
            elements = np.array(
                [self._index[symbol] for symbol in atoms.get_chemical_symbols()],
                dtype=np.int64,
            )
        except KeyError as e:
            raise ValueError(f"missing Gupta parameters for element {e}") from e

        forces = np.zeros_like(atoms.positions)
        energy = float(
            _gupta_accumulate(
                atoms.positions,
                elements,
                self._p,
                self._q,
                self._a,
                self._qsi2,
                self._r0,
                self._p3,
                self._p4,
                self._p5,
                self._q3,
                self._q4,
                self._q5,
                self.cutoff_start,
                self.cutoff_end,
                forces,
            )
        )
        self.results["energy"] = energy
        self.results["forces"] = forces
