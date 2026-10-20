#!/usr/bin/env python3
"""Paper-level observables for Schiff & Venumadhav (2025/2026).

This module keeps the published PMF normalization and angular conventions in
one place.  The dynamical mode equations live in :mod:`lmhd`; this layer turns
their transfer functions into the spectra and correlations plotted in the
paper.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

if __package__:
    from . import lmhd, lya_rt
else:
    import lmhd
    import lya_rt


IR_LENGTH_MPC = 1000.0
PLANCK_AS = 2.101e-9
PLANCK_NS = 0.9649
PLANCK_PIVOT_MPC = 0.05


def paper_cosmology() -> lmhd.Cosmology:
    """Planck-2018 TT,TE,EE+lowE+lensing+BAO background used by SV25.

    The manuscript cites that fit but does not print a parameter table.  These
    values therefore remain an explicit reconstruction assumption rather than
    author-supplied machine-readable metadata.
    """

    h = 0.6766
    omega_m = 0.3111
    omega_b = 0.02242 / h**2
    omega_r = 4.18e-5 / h**2
    return lmhd.Cosmology(
        h=h,
        omega_m=omega_m,
        omega_b=omega_b,
        omega_lambda=1.0 - omega_m - omega_r,
        omega_r=omega_r,
        hydrogen_mass_fraction=0.7546,
        tcmb0=2.725,
    )


@dataclass(frozen=True)
class MagneticSpectrum:
    """SV25 Eqs. (2), (66), and (67) for a red-tilted PMF."""

    b0_gauss: float
    epsilon: float
    ir_length_mpc: float = IR_LENGTH_MPC

    def __post_init__(self) -> None:
        if self.b0_gauss <= 0.0:
            raise ValueError("b0_gauss must be positive")
        if self.epsilon >= 0.0:
            raise ValueError("the SV25 mean-field construction requires epsilon < 0")
        if self.ir_length_mpc <= 0.0:
            raise ValueError("ir_length_mpc must be positive")

    @property
    def k_lambda_mpc(self) -> float:
        return 2.0 * math.pi / self.ir_length_mpc

    def power(self, k_mpc: np.ndarray | float) -> np.ndarray:
        """Dimensional ``P_B(k)`` in ``G^2 Mpc^3`` (paper Eq. 66)."""

        k = np.asarray(k_mpc, dtype=float)
        if np.any(k <= 0.0):
            raise ValueError("wavenumbers must be positive")
        return (
            math.pi**2
            * self.b0_gauss**2
            * abs(self.epsilon)
            * self.k_lambda_mpc ** (-self.epsilon)
            * k ** (self.epsilon - 3.0)
        )

    def dimensionless_power(self, k_mpc: np.ndarray | float) -> np.ndarray:
        """``Delta_B^2(k)`` in ``G^2`` (paper Eq. 67)."""

        k = np.asarray(k_mpc, dtype=float)
        if np.any(k <= 0.0):
            raise ValueError("wavenumbers must be positive")
        return (
            0.5
            * abs(self.epsilon)
            * self.b0_gauss**2
            * (k / self.k_lambda_mpc) ** self.epsilon
        )

    def fractional_mode_weight(self, k_mpc: np.ndarray | float) -> np.ndarray:
        """Return ``Delta_B^2/(2 B0^2)`` used in second-order integrals."""

        return 0.5 * self.dimensionless_power(k_mpc) / self.b0_gauss**2


@dataclass
class TransferGrid:
    """Complex mode transfers on a ``(k, mu, z)`` Gauss-Legendre grid."""

    k_mpc: np.ndarray
    mu: np.ndarray
    mu_weight: np.ndarray
    redshift: np.ndarray
    magnetosonic: Mapping[str, np.ndarray]
    alfven: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        self.k_mpc = np.asarray(self.k_mpc, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        self.mu_weight = np.asarray(self.mu_weight, dtype=float)
        self.redshift = np.asarray(self.redshift, dtype=float)
        expected = (self.k_mpc.size, self.mu.size, self.redshift.size)
        if self.mu.shape != self.mu_weight.shape:
            raise ValueError("mu nodes and weights must have matching shapes")
        if not np.isclose(np.sum(self.mu_weight), 2.0, rtol=0.0, atol=1.0e-13):
            raise ValueError("angular quadrature weights must integrate unity over theta")
        for polarization in (self.magnetosonic, self.alfven):
            for name, value in polarization.items():
                if np.asarray(value).shape != expected:
                    raise ValueError(f"transfer {name} has shape {np.asarray(value).shape}; expected {expected}")

    def angular_cross(
        self,
        left: str,
        right: str | None = None,
        *,
        polarization: str = "magnetosonic",
    ) -> np.ndarray:
        """Paper Eq. (13), returning an array with shape ``(k,z)``."""

        right = left if right is None else right
        if polarization not in ("magnetosonic", "alfven"):
            raise ValueError("polarization must be magnetosonic or alfven")
        fields = self.magnetosonic if polarization == "magnetosonic" else self.alfven
        product = np.asarray(fields[left]) * np.conjugate(np.asarray(fields[right]))
        return np.einsum("m,kmz->kz", self.mu_weight, product)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        payload: dict[str, np.ndarray] = {
            "k_mpc": self.k_mpc,
            "mu": self.mu,
            "mu_weight": self.mu_weight,
            "redshift": self.redshift,
        }
        payload.update({f"ms_{name}": np.asarray(value) for name, value in self.magnetosonic.items()})
        payload.update({f"a_{name}": np.asarray(value) for name, value in self.alfven.items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path | str, metadata: Mapping[str, object] | None = None) -> "TransferGrid":
        with np.load(Path(path)) as data:
            files = tuple(data.files)
            magnetosonic = {
                name[3:]: np.asarray(data[name]) for name in files if name.startswith("ms_")
            }
            alfven = {
                name[2:]: np.asarray(data[name]) for name in files if name.startswith("a_")
            }
            return cls(
                np.asarray(data["k_mpc"]),
                np.asarray(data["mu"]),
                np.asarray(data["mu_weight"]),
                np.asarray(data["redshift"]),
                magnetosonic,
                alfven,
                {} if metadata is None else metadata,
            )


def _interpolate_complex_result(
    result: lmhd.EvolutionResult,
    field: str,
    redshift: np.ndarray,
) -> np.ndarray:
    order = np.argsort(result.redshift)
    source_z = result.redshift[order]
    source = np.asarray(result.fields[field])[order]
    real = np.interp(redshift, source_z, np.real(source), left=np.nan, right=np.nan)
    imaginary = np.interp(
        redshift, source_z, np.imag(source), left=np.nan, right=np.nan
    )
    return real + 1j * imaginary


_TRANSFER_FIELDS = (
    "b_mode_over_initial",
    "delta_b",
    "v_b_x",
    "v_b_y",
    "v_b_z",
    "theta_b",
    "delta_xe",
    "delta_temperature",
    "delta_temperature_fraction",
    "pressure_fraction",
)


def _transfer_mode_task(task):
    (
        ik,
        im,
        polarization,
        k_value,
        angle,
        b0_gauss,
        cosmology,
        ionization,
        z_end,
        requested,
        closure,
        recombination_rates,
        max_dln_a,
        outputs,
        baryon_temperature,
        temperature_start_redshift,
        background_electron_fraction,
    ) = task
    mode = lmhd.ReferenceMode(
        k_mpc=float(k_value),
        theta=float(angle),
        b0_gauss=float(b0_gauss),
        perturbation_fraction=1.0,
        polarization=polarization,
    )
    mode_closure = closure if polarization == "magnetosonic" else "fixed"
    mode_temperature = (
        baryon_temperature if mode_closure == "three_level" else None
    )
    result = lmhd.evolve_sv_paper_mode(
        mode,
        cosmology,
        z_end,
        requested,
        ionization,
        initialization="physical",
        closure=mode_closure,
        recombination_rates=(
            recombination_rates if mode_closure == "three_level" else None
        ),
        max_dln_a=max_dln_a,
        include_gravity=True,
        baryon_temperature=mode_temperature,
        temperature_start_redshift=temperature_start_redshift,
        background_electron_fraction=background_electron_fraction,
    )
    payload = {}
    for field in _TRANSFER_FIELDS:
        if field in result.fields:
            payload[field] = _interpolate_complex_result(result, field, outputs)
        elif field in (
            "delta_xe",
            "delta_temperature",
            "delta_temperature_fraction",
            "pressure_fraction",
        ):
            payload[field] = np.zeros_like(outputs, dtype=np.complex128)
    return ik, im, polarization, payload


def build_transfer_grid(
    cosmology: lmhd.Cosmology,
    ionization,
    k_mpc: Sequence[float] | np.ndarray,
    redshift: Sequence[float] | np.ndarray,
    b0_gauss: float,
    *,
    angle_count: int = 10,
    closure: str = "saha",
    recombination_rates: Callable[[float, float], Sequence[float]] | None = None,
    max_dln_a: float = 2.0e-3,
    include_alfven: bool = True,
    workers: int = 1,
    angle_scheme: str = "gauss",
    fixed_theta: float | None = None,
    baryon_temperature: Callable[[float], float] | None = None,
    temperature_start_redshift: float = 1900.0,
    background_electron_fraction: Callable[[float], float] | None = None,
) -> TransferGrid:
    """Build the paper's complex transfer grid with physical mode starts.

    Values before a mode's horizon-entry/1-MeV initialization are ``NaN``.
    This is intentional: those regions are outside the model rather than
    pristine evolved states.  Spectrum calculations at recombination lie
    below every start redshift in the published wavenumber range.
    """

    wave = np.asarray(k_mpc, dtype=float)
    outputs = np.asarray(redshift, dtype=float)
    if (
        wave.ndim != 1
        or outputs.ndim != 1
        or wave.size == 0
        or outputs.size == 0
        or np.any(wave <= 0.0)
        or np.any(outputs < 0.0)
        or b0_gauss <= 0.0
    ):
        raise ValueError("transfer grids require positive k, nonnegative z, and B0")
    if closure not in ("fixed", "saha", "three_level"):
        raise ValueError("closure must be fixed, saha, or three_level")
    if closure == "three_level" and recombination_rates is None:
        raise ValueError("three_level transfer grids require recombination rates")
    if max_dln_a <= 0.0:
        raise ValueError("max_dln_a must be positive")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if angle_scheme not in ("gauss", "split", "uniform_theta", "fixed"):
        raise ValueError(
            "angle_scheme must be gauss, split, uniform_theta, or fixed"
        )
    if angle_scheme == "fixed":
        if fixed_theta is None or not 0.0 < fixed_theta < 0.5 * math.pi:
            raise ValueError("fixed angle grids require 0 < fixed_theta < pi/2")
    elif fixed_theta is not None:
        raise ValueError("fixed_theta is valid only with angle_scheme='fixed'")

    # Store redshifts in the caller's order.  The individual mode integrator
    # handles its own descending integration order.
    if angle_scheme == "gauss":
        theta, mu, weight = angular_nodes(angle_count)
    elif angle_scheme == "split":
        theta, mu, weight = split_angular_nodes(angle_count)
    elif angle_scheme == "fixed":
        theta = np.asarray([float(fixed_theta)])
        mu = np.cos(theta)
        weight = np.asarray([2.0])
    else:
        theta, mu, weight = uniform_theta_nodes(angle_count)
    shape = (wave.size, mu.size, outputs.size)
    magnetosonic = {
        field: np.full(shape, np.nan + 0.0j, dtype=np.complex128)
        for field in _TRANSFER_FIELDS
    }
    alfven = {
        field: np.full(shape, np.nan + 0.0j, dtype=np.complex128)
        for field in _TRANSFER_FIELDS
    }
    z_end = float(np.min(outputs))
    requested = outputs.tolist()

    tasks = []
    for ik, k_value in enumerate(wave):
        for im, angle in enumerate(theta):
            # The power and scalar cross-correlations are invariant under
            # B0 -> -B0.  ReferenceMode stores this symmetry-reduced domain,
            # while the Gauss-Legendre weights still cover mu in [-1, 1].
            physical_angle = min(float(angle), math.pi - float(angle))
            if angle_scheme == "uniform_theta":
                endpoint_epsilon = 1.0e-12
                physical_angle = min(
                    max(physical_angle, endpoint_epsilon),
                    0.5 * math.pi - endpoint_epsilon,
                )
            for polarization in ("magnetosonic", "alfven"):
                if polarization == "alfven" and not include_alfven:
                    continue
                tasks.append(
                    (
                        ik,
                        im,
                        polarization,
                        float(k_value),
                        physical_angle,
                        float(b0_gauss),
                        cosmology,
                        ionization,
                        z_end,
                        requested,
                        closure,
                        recombination_rates,
                        max_dln_a,
                        outputs,
                        baryon_temperature,
                        temperature_start_redshift,
                        background_electron_fraction,
                    )
                )

    if workers == 1:
        results = map(_transfer_mode_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(
            _transfer_mode_task,
            tasks,
            # Mode costs vary strongly with k.  Keep chunks short enough that
            # high-k tails do not strand most workers at the end of a grid.
            chunksize=max(1, len(tasks) // (workers * 32)),
        )
    try:
        for ik, im, polarization, payload in results:
            destination = magnetosonic if polarization == "magnetosonic" else alfven
            for field, values in payload.items():
                destination[field][ik, im] = values
    finally:
        if workers != 1:
            executor.shutdown()

    return TransferGrid(
        k_mpc=wave,
        mu=mu,
        mu_weight=weight,
        redshift=outputs,
        magnetosonic=magnetosonic,
        alfven=alfven,
        metadata={
            "b0_gauss": float(b0_gauss),
            "closure": closure,
            "angle_count": int(angle_count),
            "angle_scheme": angle_scheme,
            "fixed_theta": fixed_theta,
            "max_dln_a": float(max_dln_a),
            "physical_mode_initialization": True,
            "include_alfven": bool(include_alfven),
            "workers": int(workers),
            "include_baryon_temperature": baryon_temperature is not None,
            "custom_background_electron_fraction": (
                background_electron_fraction is not None
            ),
            "temperature_start_redshift": float(temperature_start_redshift),
        },
    )


def angular_nodes(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes for ``integral sin(theta) dtheta``."""

    if count < 2:
        raise ValueError("at least two angular nodes are required")
    mu, weight = np.polynomial.legendre.leggauss(count)
    return np.arccos(mu), mu, weight


def split_angular_nodes(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite quadrature resolving the narrow ``mu=0`` damping layer.

    Transfer powers are even under ``mu -> -mu``. The positive hemisphere is
    integrated over logarithmically split intervals and its weights are
    doubled. A single global Gauss rule would require hundreds of nodes to
    sample this endpoint layer in strongly damped high-k modes.
    """

    if order < 1:
        raise ValueError("split angular quadrature order must be positive")
    edges = np.asarray(
        (0.0, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3, 1.0)
    )
    base_node, base_weight = np.polynomial.legendre.leggauss(order)
    nodes = []
    weights = []
    for left, right in zip(edges[:-1], edges[1:]):
        half = 0.5 * (right - left)
        center = 0.5 * (right + left)
        nodes.extend(center + half * base_node)
        weights.extend(2.0 * half * base_weight)
    mu = np.asarray(nodes)
    weight = np.asarray(weights)
    return np.arccos(mu), mu, weight


def uniform_theta_nodes(intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-hemisphere trapezoid used to diagnose endpoint contamination.

    The weights include the reflected hemisphere. Unlike Gaussian rules, this
    rule assigns finite weight to the measure-zero ``mu=0`` endpoint.
    """

    if intervals < 2:
        raise ValueError("at least two uniform-theta intervals are required")
    theta = np.linspace(0.0, 0.5 * math.pi, intervals + 1)
    spacing = 0.5 * math.pi / intervals
    weight = 2.0 * spacing * np.sin(theta)
    weight[[0, -1]] *= 0.5
    weight *= 2.0 / float(np.sum(weight))
    return theta, np.cos(theta), weight


def density_power(grid: TransferGrid, spectrum: MagneticSpectrum) -> np.ndarray:
    """Dimensionless baryon-density power per log k (paper Eqs. 13--14)."""

    if not np.isclose(
        float(grid.metadata.get("b0_gauss", spectrum.b0_gauss)),
        spectrum.b0_gauss,
        rtol=1.0e-12,
    ):
        raise ValueError("transfer grid and magnetic spectrum use different B0")
    angular = np.real(grid.angular_cross("delta_b"))
    return spectrum.fractional_mode_weight(grid.k_mpc)[:, None] * angular


def scalar_field_power(
    grid: TransferGrid, spectrum: MagneticSpectrum, field: str
) -> np.ndarray:
    """Dimensionless power per log k for a magnetosonic scalar transfer."""

    if field not in grid.magnetosonic:
        raise KeyError(f"transfer grid does not contain {field}")
    if not np.isclose(
        float(grid.metadata.get("b0_gauss", spectrum.b0_gauss)),
        spectrum.b0_gauss,
        rtol=1.0e-12,
    ):
        raise ValueError("transfer grid and magnetic spectrum use different B0")
    angular = np.real(grid.angular_cross(field))
    return spectrum.fractional_mode_weight(grid.k_mpc)[:, None] * angular


def magnetic_power(grid: TransferGrid, spectrum: MagneticSpectrum) -> np.ndarray:
    """Evolved total PMF power, normalized so pristine transfers equal Eq. 67."""

    ms = np.real(grid.angular_cross("b_mode_over_initial", polarization="magnetosonic"))
    alfven = np.real(grid.angular_cross("b_mode_over_initial", polarization="alfven"))
    transfer = 0.25 * (ms + alfven)
    return spectrum.dimensionless_power(grid.k_mpc)[:, None] * transfer


def cross_correlation(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    left: str,
    right: str,
) -> np.ndarray:
    angular = grid.angular_cross(left, right)
    return spectrum.fractional_mode_weight(grid.k_mpc)[:, None] * angular


def integrate_log_wavenumber(k_mpc: Sequence[float], values: np.ndarray) -> np.ndarray:
    k = np.asarray(k_mpc, dtype=float)
    value = np.asarray(values)
    if value.shape[0] != k.size:
        raise ValueError("first value dimension must match k_mpc")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("k_mpc must be strictly increasing")
    return np.trapezoid(value, np.log(k), axis=0)


def clumping_factor(grid: TransferGrid, spectrum: MagneticSpectrum) -> np.ndarray:
    """Return ``<delta_b^2>`` as a function of output redshift."""

    return np.real(integrate_log_wavenumber(grid.k_mpc, density_power(grid, spectrum)))


@dataclass(frozen=True)
class VisibilityHistory:
    """Mean/variance optical-depth history and conformal visibility.

    ``optical_depth_variance`` contains the full unequal-time covariance:
    each Fourier transfer is integrated in redshift before its absolute
    square is averaged over angle and wavenumber.  Squaring an equal-time
    electron variance would not implement SV25 Eqs. (51)--(52).
    """

    redshift: np.ndarray
    mean_optical_depth: np.ndarray
    optical_depth_variance: np.ndarray
    visibility: np.ndarray
    homogeneous_visibility: np.ndarray
    peak_redshift: float
    homogeneous_peak_redshift: float

    @property
    def peak_shift(self) -> float:
        return self.peak_redshift - self.homogeneous_peak_redshift


@dataclass(frozen=True)
class MeanTemperatureShift:
    """Second-order homogeneous solution of SV25 Eq. (76)."""

    redshift: np.ndarray
    absolute_temperature_shift: np.ndarray
    fractional_temperature_shift: np.ndarray
    compton_second_order_source_k_s: np.ndarray
    compton_cross_source_k_s: np.ndarray
    drag_dissipation_source_k_s: np.ndarray
    adiabatic_density_source_k_s: np.ndarray
    adiabatic_temperature_source_k_s: np.ndarray


@dataclass(frozen=True)
class CoupledMeanThermalHistory:
    """Coupled revised-PRD second-order ionization/temperature history."""

    redshift: np.ndarray
    mean_ionization_shift: np.ndarray
    mean_ionization_state: np.ndarray
    fractional_temperature_shift: np.ndarray
    ionization_temperature_linear_s: np.ndarray
    ionization_temperature_quadratic_s: np.ndarray
    temperature_linear_s: np.ndarray
    temperature_ionization_linear_s: np.ndarray
    temperature_quadratic_source_s: np.ndarray
    delta_xe_density_cross: np.ndarray


def _quadratic_peak_coordinate(x: np.ndarray, y: np.ndarray) -> float:
    index = int(np.nanargmax(y))
    if index == 0 or index == x.size - 1:
        return float(x[index])
    coefficients = np.polyfit(x[index - 1 : index + 2], y[index - 1 : index + 2], 2)
    if coefficients[0] >= 0.0:
        return float(x[index])
    candidate = -coefficients[1] / (2.0 * coefficients[0])
    return float(np.clip(candidate, x[index - 1], x[index + 1]))


def visibility_history(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    mean_ionization_shift: Sequence[float] | np.ndarray | None = None,
    *,
    evaluation_redshift: Sequence[float] | np.ndarray | None = None,
    background_electron_fraction: Callable[[float], float] | None = None,
) -> VisibilityHistory:
    """Evaluate SV25 Eqs. (51)--(52) from first-order mode histories.

    The grid must be ordered from low to high redshift because optical depth
    accumulates from the observer toward last scattering.  Its first point is
    the lower boundary of the perturbative optical-depth integral; the mean
    homogeneous optical depth below that boundary is included explicitly,
    while its unprovided perturbation variance is zero by construction.
    """

    source_redshift = np.asarray(grid.redshift, dtype=float)
    if (
        source_redshift.size < 5
        or np.any(np.diff(source_redshift) <= 0.0)
        or np.any(source_redshift < 0.0)
    ):
        raise ValueError("visibility history requires increasing nonnegative redshifts")
    redshift = (
        source_redshift
        if evaluation_redshift is None
        else np.asarray(evaluation_redshift, dtype=float)
    )
    if (
        redshift.ndim != 1
        or redshift.size < 5
        or np.any(np.diff(redshift) <= 0.0)
        or redshift[0] < source_redshift[0]
        or redshift[-1] > source_redshift[-1]
    ):
        raise ValueError(
            "evaluation redshifts must increase within the transfer-grid support"
        )
    if not np.isclose(
        float(grid.metadata.get("b0_gauss", spectrum.b0_gauss)),
        spectrum.b0_gauss,
        rtol=1.0e-12,
    ):
        raise ValueError("transfer grid and magnetic spectrum use different B0")
    fields = grid.magnetosonic
    if "delta_xe" not in fields or "delta_b" not in fields:
        raise ValueError("visibility history requires delta_xe and delta_b transfers")

    source_shift = (
        np.zeros_like(source_redshift)
        if mean_ionization_shift is None
        else np.asarray(mean_ionization_shift, dtype=float)
    )
    if source_shift.shape != source_redshift.shape or not np.all(np.isfinite(source_shift)):
        raise ValueError("mean ionization shift must match the redshift grid")
    shift = np.interp(redshift, source_redshift, source_shift)
    xe = np.asarray([float(ionization(float(z))) for z in redshift])
    electrons = (
        xe
        if background_electron_fraction is None
        else np.asarray(
            [float(background_electron_fraction(float(z))) for z in redshift]
        )
    )
    if np.any(electrons + shift <= 0.0):
        raise ValueError("mean free-electron fraction must remain positive")

    optical_weight = np.asarray(
        [
            lmhd.CLIGHT
            * lmhd.SIGMA_T
            * cosmology.hydrogen_density(float(z))
            / ((1.0 + float(z)) * cosmology.hubble(float(z)))
            for z in redshift
        ]
    )
    def interpolate_transfer(name: str) -> np.ndarray:
        values = np.asarray(fields[name])
        if evaluation_redshift is None:
            return values
        flat = values.reshape(-1, source_redshift.size)
        interpolated = np.empty((flat.shape[0], redshift.size), dtype=np.complex128)
        for index, row in enumerate(flat):
            interpolated[index] = np.interp(
                redshift, source_redshift, np.real(row)
            ) + 1j * np.interp(redshift, source_redshift, np.imag(row))
        return interpolated.reshape(values.shape[:-1] + (redshift.size,))

    electron_transfer = (
        interpolate_transfer("delta_xe")
        + electrons[None, None, :] * interpolate_transfer("delta_b")
    )
    if not np.all(np.isfinite(electron_transfer)):
        raise ValueError("electron-number transfer contains non-finite values")
    mode_optical_depth = cumulative_trapezoid(
        optical_weight[None, None, :] * electron_transfer,
        redshift,
        axis=2,
        initial=0.0,
    )
    angular_variance = np.einsum(
        "m,kmz->kz",
        grid.mu_weight,
        np.abs(mode_optical_depth) ** 2,
    )
    variance = np.real(
        integrate_log_wavenumber(
            grid.k_mpc,
            spectrum.fractional_mode_weight(grid.k_mpc)[:, None]
            * angular_variance,
        )
    )

    # The transfer grid need not extend to z=0.  Include the homogeneous
    # foreground optical depth with a modest auxiliary grid; no perturbative
    # foreground is invented when the paper does not provide one.
    lower = float(redshift[0])
    foreground_z = np.linspace(0.0, lower, max(2, int(math.ceil(lower)) + 1))
    foreground_weight = np.asarray(
        [
            lmhd.CLIGHT
            * lmhd.SIGMA_T
            * cosmology.hydrogen_density(float(z))
            / ((1.0 + float(z)) * cosmology.hubble(float(z)))
            * float(
                ionization(float(z))
                if background_electron_fraction is None
                else background_electron_fraction(float(z))
            )
            for z in foreground_z
        ]
    )
    foreground_tau = float(np.trapezoid(foreground_weight, foreground_z))
    mean_integrand = optical_weight * (electrons + shift)
    homogeneous_integrand = optical_weight * electrons
    mean_tau = foreground_tau + cumulative_trapezoid(
        mean_integrand, redshift, initial=0.0
    )
    homogeneous_tau = foreground_tau + cumulative_trapezoid(
        homogeneous_integrand, redshift, initial=0.0
    )

    variance_derivative = np.gradient(variance, redshift, edge_order=2)
    # |dz/deta|=H in the paper's c=1 conformal-time convention.  The common
    # positive factor fixes the visibility normalization and its peak.
    hubble = np.asarray([cosmology.hubble(float(z)) for z in redshift])
    visibility = (
        hubble
        * (mean_integrand - 0.5 * variance_derivative)
        * np.exp(-mean_tau + 0.5 * variance)
    )
    homogeneous_visibility = (
        hubble * homogeneous_integrand * np.exp(-homogeneous_tau)
    )
    visibility_support = homogeneous_visibility > (
        1.0e-10 * float(np.max(homogeneous_visibility))
    )
    if (
        not np.all(np.isfinite(variance))
        or not np.all(np.isfinite(visibility))
        or np.any(variance < -1.0e-14)
        or np.any(visibility[visibility_support] < 0.0)
    ):
        raise RuntimeError("visibility calculation produced an unphysical history")
    return VisibilityHistory(
        redshift=redshift,
        mean_optical_depth=mean_tau,
        optical_depth_variance=np.maximum(variance, 0.0),
        visibility=visibility,
        homogeneous_visibility=homogeneous_visibility,
        peak_redshift=_quadratic_peak_coordinate(redshift, visibility),
        homogeneous_peak_redshift=_quadratic_peak_coordinate(
            redshift, homogeneous_visibility
        ),
    )


def mean_temperature_shift(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    baryon_temperature,
    mean_second_order_delta_xe: Sequence[float] | np.ndarray,
    *,
    alfven_grid: TransferGrid | None = None,
    background_electron_fraction: Callable[[float], float] | None = None,
) -> MeanTemperatureShift:
    """Integrate revised-PRD Eq. (A41c) for the mean temperature shift.

    The revised appendix sets unresolved nonlinear and ambipolar heating to
    zero.  Linear Thomson-drag kinetic-energy loss belongs to the photon
    spectral-distortion budget (Eqs. A45--A48), not the baryon heating term.
    ``drag_dissipation_source_k_s`` is therefore retained as a compatibility
    diagnostic and is identically zero here.
    """

    redshift = np.asarray(grid.redshift, dtype=float)
    second_xe = np.asarray(mean_second_order_delta_xe, dtype=float)
    if (
        redshift.size < 3
        or np.any(np.diff(redshift) <= 0.0)
        or second_xe.shape != redshift.shape
        or not np.all(np.isfinite(second_xe))
    ):
        raise ValueError("mean temperature shift requires an increasing matched history")
    required = (
        "delta_b",
        "delta_xe",
        "delta_temperature",
        "theta_b",
        "v_b_x",
        "v_b_y",
        "v_b_z",
    )
    if any(name not in grid.magnetosonic for name in required):
        raise ValueError("temperature shift grid lacks a required first-order field")
    fields = grid.magnetosonic
    covariance_xe2 = np.real(
        _array_cross_correlation(
            grid, spectrum, fields["delta_xe"], fields["delta_xe"]
        )
    )
    covariance_xe_temperature = np.real(
        _array_cross_correlation(
            grid,
            spectrum,
            fields["delta_xe"],
            fields["delta_temperature"],
        )
    )
    covariance_theta_density = np.real(
        _array_cross_correlation(
            grid, spectrum, fields["theta_b"], fields["delta_b"]
        )
    )
    covariance_theta_temperature = np.real(
        _array_cross_correlation(
            grid,
            spectrum,
            fields["theta_b"],
            fields["delta_temperature"],
        )
    )
    linear = np.empty_like(redshift)
    source_gamma2 = np.empty_like(redshift)
    source_gamma_temperature = np.empty_like(redshift)
    source_dissipation = np.empty_like(redshift)
    source_adiabatic_density = np.empty_like(redshift)
    source_adiabatic_temperature = np.empty_like(redshift)
    background_temperature = np.empty_like(redshift)
    for index, z in enumerate(redshift):
        xe = float(ionization(float(z)))
        electrons = (
            xe
            if background_electron_fraction is None
            else float(background_electron_fraction(float(z)))
        )
        tb = float(baryon_temperature(float(z)))
        background_temperature[index] = tb
        hubble = cosmology.hubble(float(z))
        gamma_c = lmhd.compton_temperature_rate(
            cosmology, float(z), electrons
        )
        denominator = 1.0 + cosmology.helium_per_hydrogen + electrons
        dgamma = (
            gamma_c
            * (1.0 + cosmology.helium_per_hydrogen)
            / max(electrons * denominator, np.finfo(float).tiny)
        )
        d2gamma = (
            -2.0
            * gamma_c
            * (1.0 + cosmology.helium_per_hydrogen)
            / max(electrons * denominator**2, np.finfo(float).tiny)
        )
        gamma2 = (
            dgamma * second_xe[index]
            + 0.5 * d2gamma * covariance_xe2[index]
        )
        tgamma = cosmology.tcmb0 * (1.0 + float(z))
        source_gamma2[index] = gamma2 * (tgamma - tb)
        source_gamma_temperature[index] = (
            -dgamma * covariance_xe_temperature[index]
        )
        coeff = lmhd.background_coefficients(
            cosmology,
            float(z),
            ionization,
            spectrum.b0_gauss,
            baryon_temperature=baryon_temperature,
            background_electron_fraction=background_electron_fraction,
        )
        source_dissipation[index] = 0.0
        source_adiabatic_density[index] = (
            (2.0 / 3.0)
            * covariance_theta_density[index]
            * tb
            / coeff["a"]
        )
        source_adiabatic_temperature[index] = (
            -(2.0 / 3.0)
            * covariance_theta_temperature[index]
            / coeff["a"]
        )
        linear[index] = -(2.0 * hubble + gamma_c)

    total_source = (
        source_gamma2
        + source_gamma_temperature
        + source_dissipation
        + source_adiabatic_density
        + source_adiabatic_temperature
    )

    def derivative(z: float, value: np.ndarray) -> np.ndarray:
        rate = np.interp(z, redshift, linear) * value[0] + np.interp(
            z, redshift, total_source
        )
        return np.asarray([-rate / (cosmology.hubble(z) * (1.0 + z))])

    descending = redshift[::-1]
    solution = solve_ivp(
        derivative,
        (float(descending[0]), float(descending[-1])),
        np.zeros(1),
        t_eval=descending,
        method="Radau",
        rtol=2.0e-7,
        atol=1.0e-10,
        max_step=max((redshift[-1] - redshift[0]) / 500.0, 0.25),
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError("mean second-order temperature integration failed")
    absolute = solution.y[0, ::-1]
    return MeanTemperatureShift(
        redshift=redshift,
        absolute_temperature_shift=absolute,
        fractional_temperature_shift=absolute / background_temperature,
        compton_second_order_source_k_s=source_gamma2,
        compton_cross_source_k_s=source_gamma_temperature,
        drag_dissipation_source_k_s=source_dissipation,
        adiabatic_density_source_k_s=source_adiabatic_density,
        adiabatic_temperature_source_k_s=source_adiabatic_temperature,
    )


def temperature_ionization_source_correction(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    baryon_temperature,
    mean_temperature_fraction: Sequence[float] | np.ndarray,
    *,
    relative_step: float = 2.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Leading local temperature correction to the quadratic electron rate.

    The correction contains the linear response to the homogeneous
    second-order temperature and every Hessian term with at least one
    first-order temperature leg.  It deliberately does not invent an
    unspecified temperature dependence for the nonlocal Ly-alpha operator.
    The second return value is the heated ``<delta_xe delta_b>`` observable.
    """

    redshift = np.asarray(grid.redshift, dtype=float)
    mean_fraction = np.asarray(mean_temperature_fraction, dtype=float)
    if mean_fraction.shape != redshift.shape:
        raise ValueError("mean temperature fraction must match the transfer grid")
    linear, quadratic, heated_cross = temperature_ionization_source_terms(
        grid,
        spectrum,
        cosmology,
        ionization,
        baryon_temperature,
        relative_step=relative_step,
    )
    return linear * mean_fraction + quadratic, heated_cross


def temperature_ionization_source_terms(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    baryon_temperature,
    *,
    relative_step: float = 2.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the A44 linear-temperature and quadratic ionization sources."""

    redshift = np.asarray(grid.redshift, dtype=float)
    fields = grid.magnetosonic
    required = (
        "delta_xe",
        "delta_b",
        "theta_b",
        "delta_temperature_fraction",
    )
    if any(name not in fields for name in required):
        raise ValueError("temperature ionization correction lacks required fields")
    a_hubble = np.asarray(
        [cosmology.hubble(float(z)) / (1.0 + float(z)) for z in redshift]
    )
    variables = (
        np.asarray(fields["delta_xe"]),
        np.asarray(fields["delta_b"]),
        np.asarray(fields["theta_b"]) / a_hubble[None, None, :],
        np.asarray(fields["delta_temperature_fraction"]),
    )
    covariance = np.empty((4, 4, redshift.size))
    for i in range(4):
        for j in range(i + 1):
            value = np.real(
                _array_cross_correlation(
                    grid, spectrum, variables[i], variables[j]
                )
            )
            covariance[i, j] = value
            covariance[j, i] = value
    linear = np.empty_like(redshift)
    quadratic = np.empty_like(redshift)
    for index, z in enumerate(redshift):
        gradient, hessian = local_peebles_temperature_rate_derivatives(
            cosmology,
            float(z),
            float(ionization(float(z))),
            baryon_temperature,
            relative_step=relative_step,
        )
        thermal_covariance = covariance[:, :, index].copy()
        thermal_covariance[:3, :3] = 0.0
        linear[index] = gradient[3]
        quadratic[index] = 0.5 * float(
            np.sum(hessian * thermal_covariance)
        )
    return linear, quadratic, covariance[0, 1]


def integrate_coupled_mean_thermal_history(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    baryon_temperature,
    snapshots: Sequence[NonlocalSecondOrderSnapshot],
    *,
    background_electron_fraction: Callable[[float], float] | None = None,
    relative_step: float = 2.0e-4,
) -> CoupledMeanThermalHistory:
    """Solve revised PRD Eqs. (61), (A41c), and (A42)--(A44)."""

    redshift = np.asarray(grid.redshift, dtype=float)
    ordered = sorted(snapshots, key=lambda row: row.redshift)
    snapshot_redshift = np.asarray([row.redshift for row in ordered])
    if not np.array_equal(redshift, snapshot_redshift):
        raise ValueError("thermal history requires snapshots on the transfer grid")
    fields = grid.magnetosonic
    required = (
        "delta_b",
        "delta_xe",
        "delta_temperature_fraction",
        "theta_b",
    )
    if any(name not in fields for name in required):
        raise ValueError("coupled thermal history lacks a first-order field")

    ion_temperature_linear, ion_temperature_quadratic, heated_cross = (
        temperature_ionization_source_terms(
            grid,
            spectrum,
            cosmology,
            ionization,
            baryon_temperature,
            relative_step=relative_step,
        )
    )
    covariance_xe2 = np.real(
        _array_cross_correlation(
            grid, spectrum, fields["delta_xe"], fields["delta_xe"]
        )
    )
    covariance_xe_temperature = np.real(
        _array_cross_correlation(
            grid,
            spectrum,
            fields["delta_xe"],
            fields["delta_temperature_fraction"],
        )
    )
    covariance_theta_density = np.real(
        _array_cross_correlation(
            grid, spectrum, fields["theta_b"], fields["delta_b"]
        )
    )
    covariance_theta_temperature = np.real(
        _array_cross_correlation(
            grid,
            spectrum,
            fields["theta_b"],
            fields["delta_temperature_fraction"],
        )
    )

    ion_linear = np.asarray([row.mean_rate_linear_s for row in ordered])
    ion_source = np.asarray([row.mean_rate_source_s for row in ordered])
    temperature_linear = np.empty_like(redshift)
    temperature_ionization_linear = np.empty_like(redshift)
    temperature_source = np.empty_like(redshift)
    for index, z in enumerate(redshift):
        xe = float(ionization(float(z)))
        electrons = (
            xe
            if background_electron_fraction is None
            else float(background_electron_fraction(float(z)))
        )
        tb = float(baryon_temperature(float(z)))
        tgamma = cosmology.tcmb0 * (1.0 + float(z))
        gamma_c = lmhd.compton_temperature_rate(
            cosmology, float(z), electrons
        )
        denominator = 1.0 + cosmology.helium_per_hydrogen + electrons
        dgamma = (
            gamma_c
            * (1.0 + cosmology.helium_per_hydrogen)
            / max(electrons * denominator, np.finfo(float).tiny)
        )
        temperature_linear[index] = -gamma_c * tgamma / tb
        temperature_ionization_linear[index] = dgamma * (tgamma / tb - 1.0)
        temperature_source[index] = (
            -dgamma
            * covariance_xe2[index]
            / denominator
            * (tgamma / tb - 1.0)
            - dgamma * covariance_xe_temperature[index]
            + (2.0 / 3.0)
            * (1.0 + float(z))
            * (
                covariance_theta_density[index]
                - covariance_theta_temperature[index]
            )
        )

    def derivative(z: float, value: np.ndarray) -> np.ndarray:
        ion_rate = (
            np.interp(z, redshift, ion_linear) * value[0]
            + np.interp(z, redshift, ion_source)
            + np.interp(z, redshift, ion_temperature_linear) * value[1]
            + np.interp(z, redshift, ion_temperature_quadratic)
        )
        temperature_rate = (
            np.interp(z, redshift, temperature_linear) * value[1]
            + np.interp(z, redshift, temperature_ionization_linear) * value[0]
            + np.interp(z, redshift, temperature_source)
        )
        return -np.asarray((ion_rate, temperature_rate)) / (
            cosmology.hubble(z) * (1.0 + z)
        )

    descending = redshift[::-1]
    solution = solve_ivp(
        derivative,
        (float(descending[0]), float(descending[-1])),
        np.zeros(2),
        t_eval=descending,
        method="Radau",
        rtol=2.0e-7,
        atol=np.asarray((1.0e-12, 1.0e-13)),
        max_step=max((redshift[-1] - redshift[0]) / 500.0, 0.25),
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError("coupled mean thermal integration failed")
    ion_state = solution.y[0, ::-1]
    temperature_fraction = solution.y[1, ::-1]
    return CoupledMeanThermalHistory(
        redshift=redshift,
        mean_ionization_shift=ion_state + heated_cross,
        mean_ionization_state=ion_state,
        fractional_temperature_shift=temperature_fraction,
        ionization_temperature_linear_s=ion_temperature_linear,
        ionization_temperature_quadratic_s=ion_temperature_quadratic,
        temperature_linear_s=temperature_linear,
        temperature_ionization_linear_s=temperature_ionization_linear,
        temperature_quadratic_source_s=temperature_source,
        delta_xe_density_cross=heated_cross,
    )


def saha_linear_delta_xe(xe: np.ndarray | float, delta_b: np.ndarray | float) -> np.ndarray:
    xe_array = np.asarray(xe, dtype=float)
    return -xe_array * (1.0 - xe_array) / (2.0 - xe_array) * np.asarray(delta_b)


def saha_mean_ionization_shift(
    xe: np.ndarray | float, density_variance: np.ndarray | float
) -> np.ndarray:
    """Paper Eq. (24), the observable mean shift ``Delta x_e``."""

    xe_array = np.asarray(xe, dtype=float)
    return (
        -xe_array
        * (1.0 - xe_array)
        / (2.0 - xe_array) ** 3
        * np.asarray(density_variance)
    )


def local_peebles_electron_rate(
    cosmology: lmhd.Cosmology,
    redshift: float,
    xe: float,
    density_contrast: float = 0.0,
    theta_over_aH: float = 0.0,
    temperature_fraction: float = 0.0,
    baryon_temperature=None,
) -> float:
    """Local 3LA ``dx_e/dt`` with density and expansion perturbations.

    This implements the separate-universe prescription described below SV25
    Eq. (49): ``n_H -> n_H(1+delta_b)`` and
    ``H -> H + Theta/(3a)``.  Matter and radiation temperatures are held at
    the homogeneous CMB temperature, as in the paper's main calculation.
    """

    z = float(redshift)
    electron_fraction = float(xe)
    delta = float(density_contrast)
    velocity = float(theta_over_aH)
    thermal = float(temperature_fraction)
    if z < 0.0 or not 0.0 <= electron_fraction <= 1.0:
        raise ValueError("local Peebles rate requires z >= 0 and 0 <= xe <= 1")
    if delta <= -1.0 or 1.0 + velocity / 3.0 <= 0.0:
        raise ValueError("local density and expansion rate must remain positive")

    background_temperature = (
        cosmology.tcmb0 * (1.0 + z)
        if baryon_temperature is None
        else float(baryon_temperature(z))
    )
    temperature = background_temperature * (1.0 + thermal)
    if temperature <= 0.0:
        raise ValueError("local matter temperature must remain positive")
    hydrogen = cosmology.hydrogen_density(z) * (1.0 + delta)
    hubble = cosmology.hubble(z) * (1.0 + velocity / 3.0)
    alpha_b = lmhd._case_b_recombination_rate(temperature)
    beta_b = alpha_b * (
        2.0
        * math.pi
        * lmhd.MELECTRON
        * lmhd.KBOLTZ
        * temperature
        / lmhd.HPLANCK**2
    ) ** 1.5 * math.exp(-lmhd.E_LEVEL2 / (lmhd.KBOLTZ * temperature))
    neutral = max(1.0 - electron_fraction, 0.0)
    escape = lmhd.LAMBDA_LYA**3 / (8.0 * math.pi * hubble)
    kl = escape * lmhd.LAMBDA_2S1S * hydrogen * neutral
    kb = escape * beta_b * hydrogen * neutral
    peebles = (1.0 + kl) / (1.0 + kl + kb)
    ground_photoionization = beta_b * math.exp(
        -lmhd.E_LYA / (lmhd.KBOLTZ * temperature)
    )
    return peebles * (
        -hydrogen * alpha_b * electron_fraction**2
        + ground_photoionization * neutral
    )


def local_peebles_rate_derivatives(
    cosmology: lmhd.Cosmology,
    redshift: float,
    xe: float,
    *,
    relative_step: float = 2.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of the local 3LA rate.

    Coordinates are ``(delta_xe, delta_b, Theta/(aH))``.  Centered mixed
    differences make the quadratic source explicit and independently
    refinable without introducing an automatic-differentiation dependency.
    """

    step = float(relative_step)
    if not 1.0e-7 <= step <= 1.0e-2:
        raise ValueError("relative_step must lie in [1e-7, 1e-2]")
    center = np.asarray((float(xe), 0.0, 0.0))
    electron_spacing = min(
        step * max(float(xe), 1.0e-2),
        0.2 * max(float(xe), 1.0e-12),
        0.2 * max(1.0 - float(xe), 1.0e-12),
    )
    spacing = np.asarray((electron_spacing, step, step))

    def evaluate(point: np.ndarray) -> float:
        return local_peebles_electron_rate(
            cosmology,
            redshift,
            point[0],
            point[1],
            point[2],
        )

    f0 = evaluate(center)
    gradient = np.empty(3)
    hessian = np.empty((3, 3))
    for i in range(3):
        offset = np.zeros(3)
        offset[i] = spacing[i]
        plus = evaluate(center + offset)
        minus = evaluate(center - offset)
        gradient[i] = (plus - minus) / (2.0 * spacing[i])
        hessian[i, i] = (plus - 2.0 * f0 + minus) / spacing[i] ** 2
        for j in range(i):
            second = np.zeros(3)
            second[j] = spacing[j]
            mixed = (
                evaluate(center + offset + second)
                - evaluate(center + offset - second)
                - evaluate(center - offset + second)
                + evaluate(center - offset - second)
            ) / (4.0 * spacing[i] * spacing[j])
            hessian[i, j] = mixed
            hessian[j, i] = mixed
    return gradient, hessian


def local_peebles_temperature_rate_derivatives(
    cosmology: lmhd.Cosmology,
    redshift: float,
    xe: float,
    baryon_temperature,
    *,
    relative_step: float = 2.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient/Hessian including the coordinate ``delta(T_b)/T_b``."""

    step = float(relative_step)
    if not 1.0e-7 <= step <= 1.0e-2:
        raise ValueError("relative_step must lie in [1e-7, 1e-2]")
    center = np.asarray((float(xe), 0.0, 0.0, 0.0))
    electron_spacing = min(
        step * max(float(xe), 1.0e-2),
        0.2 * max(float(xe), 1.0e-12),
        0.2 * max(1.0 - float(xe), 1.0e-12),
    )
    spacing = np.asarray((electron_spacing, step, step, step))

    def evaluate(point: np.ndarray) -> float:
        return local_peebles_electron_rate(
            cosmology,
            redshift,
            point[0],
            point[1],
            point[2],
            point[3],
            baryon_temperature,
        )

    f0 = evaluate(center)
    gradient = np.empty(4)
    hessian = np.empty((4, 4))
    for i in range(4):
        offset = np.zeros(4)
        offset[i] = spacing[i]
        plus = evaluate(center + offset)
        minus = evaluate(center - offset)
        gradient[i] = (plus - minus) / (2.0 * spacing[i])
        hessian[i, i] = (plus - 2.0 * f0 + minus) / spacing[i] ** 2
        for j in range(i):
            second = np.zeros(4)
            second[j] = spacing[j]
            mixed = (
                evaluate(center + offset + second)
                - evaluate(center + offset - second)
                - evaluate(center - offset + second)
                + evaluate(center - offset - second)
            ) / (4.0 * spacing[i] * spacing[j])
            hessian[i, j] = mixed
            hessian[j, i] = mixed
    return gradient, hessian


@dataclass(frozen=True)
class LocalPeeblesResponse:
    """Picklable first-order local-3LA response for transfer-grid workers."""

    cosmology: lmhd.Cosmology
    ionization: object
    relative_step: float = 2.0e-4

    def __call__(self, redshift: float, k_mpc: float) -> np.ndarray:
        del k_mpc
        gradient, _ = local_peebles_rate_derivatives(
            self.cosmology,
            redshift,
            float(self.ionization(redshift)),
            relative_step=self.relative_step,
        )
        # The LMHD closure consumes coefficients of
        # (delta_b, delta_xe, Theta/(aH)).
        return gradient[[1, 0, 2]]


@dataclass(frozen=True)
class LocalPeeblesResponseTable:
    """Interpolated local response shared by all wavenumbers."""

    redshift: np.ndarray
    rates: np.ndarray

    @classmethod
    def build(
        cls,
        cosmology: lmhd.Cosmology,
        ionization,
        *,
        zmin: float = 500.0,
        zmax: float = 2000.0,
        samples: int = 1501,
        relative_step: float = 2.0e-4,
    ) -> "LocalPeeblesResponseTable":
        if not zmax > zmin >= 0.0 or samples < 3:
            raise ValueError("local response table requires zmax > zmin and samples >= 3")
        redshift = np.linspace(zmin, zmax, samples)
        rates = np.empty((samples, 3))
        for index, z in enumerate(redshift):
            gradient, _ = local_peebles_rate_derivatives(
                cosmology,
                float(z),
                float(ionization(float(z))),
                relative_step=relative_step,
            )
            rates[index] = gradient[[1, 0, 2]]
        return cls(redshift, rates)

    def __post_init__(self) -> None:
        if (
            np.asarray(self.redshift).ndim != 1
            or np.asarray(self.rates).shape != (np.asarray(self.redshift).size, 3)
            or np.asarray(self.redshift).size < 3
            or np.any(np.diff(self.redshift) <= 0.0)
            or not np.all(np.isfinite(self.rates))
        ):
            raise ValueError("invalid local Peebles response table")

    def __call__(self, redshift: float, k_mpc: float) -> np.ndarray:
        del k_mpc
        z = float(redshift)
        if not self.redshift[0] <= z <= self.redshift[-1]:
            return np.zeros(3)
        return np.asarray(
            [np.interp(z, self.redshift, self.rates[:, index]) for index in range(3)]
        )


def _array_cross_correlation(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    product = np.asarray(left) * np.conjugate(np.asarray(right))
    expected = (grid.k_mpc.size, grid.mu.size, grid.redshift.size)
    if product.shape != expected:
        raise ValueError(f"cross-correlation arrays must have shape {expected}")
    angular = np.einsum("m,kmz->kz", grid.mu_weight, product)
    weighted = spectrum.fractional_mode_weight(grid.k_mpc)[:, None] * angular
    return integrate_log_wavenumber(grid.k_mpc, weighted)


def local_three_level_mean_ionization_shift(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    *,
    relative_step: float = 2.0e-4,
    rtol: float = 2.0e-7,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the local-3LA ``Delta x_e`` and its two SV25 Eq. (16) terms.

    The first-order fields provide the full time-dependent covariance matrix
    of ``(delta_xe, delta_b, Theta/(aH))``.  Taylor expanding the local
    Peebles rate to second order gives a scalar stiff ODE for
    ``<delta x_e^(2)>``.  The observable also includes
    ``<delta_xe^(1) delta_b^(1)>``.
    """

    fields = grid.magnetosonic
    required = ("delta_xe", "delta_b", "theta_b")
    if any(name not in fields for name in required):
        raise ValueError("local 3LA shift requires delta_xe, delta_b, and theta_b")
    if not np.isclose(
        float(grid.metadata.get("b0_gauss", spectrum.b0_gauss)),
        spectrum.b0_gauss,
        rtol=1.0e-12,
    ):
        raise ValueError("transfer grid and magnetic spectrum use different B0")

    redshift = np.asarray(grid.redshift, dtype=float)
    if redshift.size < 3 or np.any(redshift < 0.0):
        raise ValueError("quadratic ionization history requires at least three redshifts")
    a_hubble = np.asarray(
        [cosmology.hubble(float(z)) / (1.0 + float(z)) for z in redshift]
    )
    variables = (
        np.asarray(fields["delta_xe"]),
        np.asarray(fields["delta_b"]),
        np.asarray(fields["theta_b"]) / a_hubble[None, None, :],
    )
    covariance = np.empty((3, 3, redshift.size))
    for i in range(3):
        for j in range(i + 1):
            value = np.real(
                _array_cross_correlation(grid, spectrum, variables[i], variables[j])
            )
            covariance[i, j] = value
            covariance[j, i] = value

    order = np.argsort(redshift)
    ascending_z = redshift[order]
    ascending_covariance = covariance[:, :, order]
    zmin = float(ascending_z[0])
    zmax = float(ascending_z[-1])

    def covariance_at(z: float) -> np.ndarray:
        return np.asarray(
            [
                np.interp(z, ascending_z, ascending_covariance[i, j])
                for i in range(3)
                for j in range(3)
            ]
        ).reshape(3, 3)

    def derivative(z: float, value: np.ndarray) -> np.ndarray:
        xe = float(ionization(z))
        gradient, hessian = local_peebles_rate_derivatives(
            cosmology, z, xe, relative_step=relative_step
        )
        source = 0.5 * float(np.sum(hessian * covariance_at(z)))
        rate = gradient[0] * value[0] + source
        return np.asarray([-rate / (cosmology.hubble(z) * (1.0 + z))])

    descending_z = ascending_z[::-1]
    solution = solve_ivp(
        derivative,
        (zmax, zmin),
        np.zeros(1),
        t_eval=descending_z,
        method="Radau",
        rtol=rtol,
        atol=atol,
        max_step=max((zmax - zmin) / 400.0, 0.25),
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError("local second-order ionization integration failed")
    second_ascending = solution.y[0, ::-1]
    second_order = np.interp(redshift, ascending_z, second_ascending)
    cross_term = covariance[0, 1]
    return second_order + cross_term, second_order, cross_term


@dataclass(frozen=True)
class NonlocalSecondOrderSnapshot:
    """One-redshift reduction of the SV25 quadratic Ly-alpha problem."""

    redshift: float
    mean_rate_linear_s: float
    mean_rate_source_s: float
    delta_xe_density_cross: float
    source_line_average: float
    source_residual_max: float
    peebles_second_order: float
    a2_line_average: float
    c2_line_average: float
    covariance_delta_xe2: float
    covariance_feq_delta_xe: float
    covariance_fbar_delta_xe: float
    source_line_average_collision: float = 0.0
    source_line_average_redshift_first: float = 0.0
    source_line_average_redshift_second: float = 0.0
    source_line_average_advection_lensing: float = 0.0


def _angular_array_cross(
    weight: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    return float(np.real(np.sum(weight * left * np.conjugate(right))))


def nonlocal_second_order_snapshots(
    grid: TransferGrid,
    spectra: Sequence[MagneticSpectrum],
    cosmology: lmhd.Cosmology,
    ionization,
    redshift_index: int,
    *,
    xmax: float = 1000.0,
    bins_per_width: float = 2.0,
    lmax: int = 16,
) -> list[NonlocalSecondOrderSnapshot]:
    """Solve SV25 Eqs. (57)--(60) for several spectral tilts at one redshift.

    All first-order scalar and vector line bases are recomputed independently
    on a common Doppler-frequency grid.  The source includes the collision,
    first/second-order redshift, advection, and lensing terms listed in the
    paper's Appendix C.  The small Lyman-continuum correction is not included;
    the paper states that it is subdominant and does not provide the frequency
    dependence of its cross section or emission profile.  Basis construction
    is independent of spectral tilt, so all requested spectra share that
    expensive work.
    """

    requested_spectra = tuple(spectra)
    if not requested_spectra:
        raise ValueError("at least one magnetic spectrum is required")
    if not 0 <= redshift_index < grid.redshift.size:
        raise IndexError("redshift_index is outside the transfer grid")
    if not bool(grid.metadata.get("include_alfven", False)):
        raise ValueError("the nonlocal quadratic source requires Alfven transfers")
    grid_b0 = float(
        grid.metadata.get("b0_gauss", requested_spectra[0].b0_gauss)
    )
    if any(
        not np.isclose(grid_b0, spectrum.b0_gauss, rtol=1.0e-12)
        for spectrum in requested_spectra
    ):
        raise ValueError("transfer grid and magnetic spectra use different B0")
    for polarization in (grid.magnetosonic, grid.alfven):
        for field in ("delta_b", "delta_xe", "theta_b", "v_b_x", "v_b_y"):
            if field not in polarization:
                raise ValueError(f"quadratic source requires transfer field {field}")

    z = float(grid.redshift[redshift_index])
    background = lya_rt.lya_background(cosmology, z, ionization)
    homogeneous = lya_rt.solve_homogeneous_line(
        background,
        xmax=xmax,
        bins_per_width=bins_per_width,
        method="sparse",
    )
    zero_bases = {
        name: lya_rt.solve_scalar_basis(
            homogeneous,
            0.0,
            name,
            lmax=lmax,
            bins_per_width=bins_per_width,
            method="sparse",
        )
        for name in ("A", "C")
    }
    a2bar = float(zero_bases["A"].line_average)
    c2bar = float(zero_bases["C"].line_average)
    second_denominator = 1.0 + background.p_sc * c2bar
    second_escape = (1.0 - background.p_ab * c2bar) / second_denominator
    lya_second = 3.0 * lya_rt.A_LYA * second_escape
    p2 = (lya_second + lmhd.LAMBDA_2S1S) / (
        lya_second + lmhd.LAMBDA_2S1S + 4.0 * background.beta_b
    )

    source_integrands = []
    component_integrands: dict[str, list[np.ndarray]] = {
        "collision": [],
        "redshift_first": [],
        "redshift_second": [],
        "advection_lensing": [],
    }
    correlation_integrands: dict[str, list[float]] = {
        "xe_density": [],
        "xe2": [],
        "feq_xe": [],
        "fbar_xe": [],
    }
    mesh = homogeneous.x
    profile = lya_rt.voigt_profile(mesh, background.damping)
    homogeneous_derivative = np.interp(
        mesh, homogeneous.x, homogeneous.derivative
    )
    x1s = max(1.0 - background.xe, 1.0e-14)
    a_hubble = background.scale_factor * background.hubble
    rho_b = cosmology.baryon_density(z)

    for ik, k_mpc in enumerate(grid.k_mpc):
        scalar = {
            name: lya_rt.solve_scalar_basis(
                homogeneous,
                float(k_mpc),
                name,
                lmax=lmax,
                bins_per_width=bins_per_width,
                method="sparse",
            )
            for name in ("A", "B", "C")
        }
        vector = {
            polarization: lya_rt.solve_vector_basis(
                homogeneous,
                float(k_mpc),
                polarization,
                lmax=lmax,
                bins_per_width=bins_per_width,
                method="marching",
            )
            for polarization in ("+", "-")
        }
        first_order = {
            "background": background,
            "Abar": scalar["A"].line_average,
            "Bbar": scalar["B"].line_average,
            "Cbar": scalar["C"].line_average,
            "denominator": 1.0 + background.p_sc * scalar["C"].line_average,
        }
        equilibrium = lya_rt.equilibrium_source_coefficients(
            first_order, cosmology
        )

        ms = grid.magnetosonic
        d = np.asarray(ms["delta_b"])[ik, :, redshift_index]
        dx = np.asarray(ms["delta_xe"])[ik, :, redshift_index]
        u = np.asarray(ms["theta_b"])[ik, :, redshift_index] / a_hubble
        ground = -dx / x1s
        r = d + ground
        q = (
            equilibrium.alpha_bar * d
            + equilibrium.beta_bar * ground
            + equilibrium.gamma_bar * u
        )
        feq1 = (
            equilibrium.feq_density * d
            + equilibrium.feq_ground * ground
            + equilibrium.feq_velocity * u
        )
        fbar1 = (
            r * scalar["A"].line_average
            + u * scalar["B"].line_average
            + q * scalar["C"].line_average
        )
        weight = grid.mu_weight
        covariance = {
            "rr": _angular_array_cross(weight, r, r),
            "ru": _angular_array_cross(weight, r, u),
            "rq": _angular_array_cross(weight, r, q),
            "ur": _angular_array_cross(weight, u, r),
            "uu": _angular_array_cross(weight, u, u),
            "uq": _angular_array_cross(weight, u, q),
            "dd": _angular_array_cross(weight, d, d),
            "dx": _angular_array_cross(weight, d, dx),
            "du": _angular_array_cross(weight, d, u),
            "xx": _angular_array_cross(weight, dx, dx),
            "feqx": _angular_array_cross(weight, feq1, dx),
            "fbarx": _angular_array_cross(weight, fbar1, dx),
        }

        k_cgs = float(k_mpc) / lmhd.MPC_CM
        streaming = (
            lmhd.CLIGHT
            * k_cgs
            * background.doppler_width
            / a_hubble
        )
        derivatives = {
            name: np.gradient(
                basis.multipoles, mesh, axis=1, edge_order=2
            )
            for name, basis in scalar.items()
        }
        vector_derivatives = {
            name: basis.equation_derivatives for name, basis in vector.items()
        }
        op_a = (
            -homogeneous_derivative
            + derivatives["A"][0]
            - streaming * scalar["A"].multipoles[1] / 3.0
        )
        op_b = (
            homogeneous_derivative / 3.0
            + derivatives["B"][0]
            - streaming * scalar["B"].multipoles[1] / 3.0
        )
        op_c = (
            derivatives["C"][0]
            - streaming * scalar["C"].multipoles[1] / 3.0
        )
        collision = (
            covariance["rr"] * op_a
            + covariance["ru"] * op_b
            + covariance["rq"] * op_c
        )
        redshift_scalar = (
            covariance["ur"]
            * (0.4 * derivatives["A"][2] - derivatives["A"][0])
            + covariance["uu"]
            * (0.4 * derivatives["B"][2] - derivatives["B"][0])
            + covariance["uq"]
            * (0.4 * derivatives["C"][2] - derivatives["C"][0])
        ) / 3.0

        vector_cross = 0.0
        for fields in (grid.magnetosonic, grid.alfven):
            phi_x = (
                1j
                * k_cgs
                * np.asarray(fields["v_b_x"])[ik, :, redshift_index]
                / a_hubble
            )
            phi_y = (
                1j
                * k_cgs
                * np.asarray(fields["v_b_y"])[ik, :, redshift_index]
                / a_hubble
            )
            vector_cross += _angular_array_cross(
                weight, phi_x - 1j * phi_y, phi_x + 1j * phi_y
            )
        redshift_vector = -vector_cross * (
            vector_derivatives["+"][2] + vector_derivatives["-"][2]
        ) / 5.0

        redshift_second = (
            16.0
            * math.pi
            * lmhd.GRAVITY
            * rho_b
            * background.scale_factor**2
            / (3.0 * lmhd.CLIGHT**2 * k_cgs**2)
            * covariance["du"]
            * homogeneous_derivative
        )
        gravity_bracket = (
            covariance["dd"]
            * (scalar["A"].multipoles[1] + equilibrium.alpha_bar * scalar["C"].multipoles[1])
            - covariance["dx"]
            / x1s
            * (scalar["A"].multipoles[1] + equilibrium.beta_bar * scalar["C"].multipoles[1])
            + covariance["du"]
            * (scalar["B"].multipoles[1] + equilibrium.gamma_bar * scalar["C"].multipoles[1])
        )
        advection_plus_lensing = (
            -8.0
            * math.pi
            * lmhd.GRAVITY
            * rho_b
            * background.scale_factor
            * background.doppler_width
            / (3.0 * lmhd.CLIGHT * background.hubble * k_cgs)
            * gravity_bracket
        )
        # PRD Eq. (52) writes +(dnu/dt)^(1) dnu(delta f^(1))/(H nu)
        # after moving this known term to the right-hand side.  Its displayed
        # expression therefore enters S directly.  The minus sign on the
        # corresponding arXiv-v2 left-hand side was a manuscript typo.
        redshift_first = redshift_scalar + redshift_vector
        component_integrands["collision"].append(collision)
        component_integrands["redshift_first"].append(redshift_first)
        component_integrands["redshift_second"].append(redshift_second)
        component_integrands["advection_lensing"].append(
            advection_plus_lensing
        )
        source_integrands.append(
            collision
            + redshift_first
            + redshift_second
            + advection_plus_lensing
        )
        correlation_integrands["xe_density"].append(covariance["dx"])
        correlation_integrands["xe2"].append(covariance["xx"])
        correlation_integrands["feq_xe"].append(covariance["feqx"])
        correlation_integrands["fbar_xe"].append(covariance["fbarx"])

    log_wave = np.log(grid.k_mpc)
    n_h = cosmology.hydrogen_density(z)
    xe = background.xe
    alpha = background.alpha_b
    beta = background.beta_b
    feq = background.f_equilibrium
    linear = (
        # Eq. (60) in the v2 manuscript prints x_e^2/x_e here.  The
        # preceding perturbed population equations require x_e^2/x_1s:
        # this term must cancel 4 f_eq beta_B in equilibrium and recover
        # the (negative) local Peebles relaxation eigenvalue as k -> 0.
        (1.0 - p2) * n_h * xe**2 * alpha / x1s
        - 2.0 * p2 * n_h * xe * alpha
        - 4.0 * feq * beta
        - 3.0
        * (1.0 - p2)
        * lya_rt.A_LYA
        * a2bar
        / second_denominator
    )
    snapshots = []
    for spectrum in requested_spectra:
        spectral_weight = spectrum.fractional_mode_weight(grid.k_mpc)
        source = np.trapezoid(
            spectral_weight[:, None] * np.asarray(source_integrands),
            log_wave,
            axis=0,
        )
        correlations = {
            name: float(
                np.trapezoid(spectral_weight * np.asarray(values), log_wave)
            )
            for name, values in correlation_integrands.items()
        }
        source_solution = lya_rt.solve_second_order_source(
            homogeneous,
            source,
            source_grid=mesh,
            bins_per_width=bins_per_width,
            method="sparse",
        )
        component_line_averages = {}
        for name, values in component_integrands.items():
            component_source = np.trapezoid(
                spectral_weight[:, None] * np.asarray(values),
                log_wave,
                axis=0,
            )
            component_line_averages[name] = lya_rt.solve_second_order_source(
                homogeneous,
                component_source,
                source_grid=mesh,
                bins_per_width=bins_per_width,
                method="sparse",
            ).line_average

        source_rate = (
            3.0
            * (1.0 - p2)
            * x1s
            * lya_rt.A_LYA
            * (
                source_solution.line_average / second_denominator
                - correlations["fbar_xe"] / x1s
                - a2bar
                / second_denominator
                * correlations["xe_density"]
                / x1s
            )
            - 2.0 * p2 * n_h * xe * alpha * correlations["xe_density"]
            # Eliminating the quasi-steady n=2 population gives -P^(2)
            # n_H alpha_B <delta_xe^2>.  Eq. (60) in the v2 manuscript prints
            # a plus sign, inconsistent with its immediately preceding Eq. (59)
            # and with the k -> 0 Taylor expansion of the local Peebles rate.
            - p2 * n_h * alpha * correlations["xe2"]
            + (
                (1.0 - p2) * (3.0 * lya_rt.A_LYA + lmhd.LAMBDA_2S1S)
                - 4.0 * p2 * beta
            )
            * correlations["feq_xe"]
        )
        snapshots.append(
            NonlocalSecondOrderSnapshot(
                redshift=z,
                mean_rate_linear_s=float(linear),
                mean_rate_source_s=float(source_rate),
                delta_xe_density_cross=correlations["xe_density"],
                source_line_average=float(source_solution.line_average),
                source_residual_max=float(source_solution.residual_max),
                peebles_second_order=float(p2),
                a2_line_average=a2bar,
                c2_line_average=c2bar,
                covariance_delta_xe2=correlations["xe2"],
                covariance_feq_delta_xe=correlations["feq_xe"],
                covariance_fbar_delta_xe=correlations["fbar_xe"],
                source_line_average_collision=float(
                    component_line_averages["collision"]
                ),
                source_line_average_redshift_first=float(
                    component_line_averages["redshift_first"]
                ),
                source_line_average_redshift_second=float(
                    component_line_averages["redshift_second"]
                ),
                source_line_average_advection_lensing=float(
                    component_line_averages["advection_lensing"]
                ),
            )
        )
    return snapshots


def nonlocal_second_order_snapshot(
    grid: TransferGrid,
    spectrum: MagneticSpectrum,
    cosmology: lmhd.Cosmology,
    ionization,
    redshift_index: int,
    *,
    xmax: float = 1000.0,
    bins_per_width: float = 2.0,
    lmax: int = 16,
) -> NonlocalSecondOrderSnapshot:
    """Single-spectrum wrapper for :func:`nonlocal_second_order_snapshots`."""

    return nonlocal_second_order_snapshots(
        grid,
        (spectrum,),
        cosmology,
        ionization,
        redshift_index,
        xmax=xmax,
        bins_per_width=bins_per_width,
        lmax=lmax,
    )[0]


def integrate_nonlocal_mean_ionization_shift(
    snapshots: Sequence[NonlocalSecondOrderSnapshot],
    cosmology: lmhd.Cosmology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate SV25 Eq. (60) from a sequence of source snapshots."""

    if len(snapshots) < 3:
        raise ValueError("nonlocal mean history requires at least three snapshots")
    ordered = sorted(snapshots, key=lambda row: row.redshift)
    redshift = np.asarray([row.redshift for row in ordered])
    if np.any(np.diff(redshift) <= 0.0):
        raise ValueError("snapshot redshifts must be unique")
    linear = np.asarray([row.mean_rate_linear_s for row in ordered])
    source = np.asarray([row.mean_rate_source_s for row in ordered])
    cross = np.asarray([row.delta_xe_density_cross for row in ordered])

    def derivative(z: float, value: np.ndarray) -> np.ndarray:
        rate = np.interp(z, redshift, linear) * value[0] + np.interp(
            z, redshift, source
        )
        return np.asarray([-rate / (cosmology.hubble(z) * (1.0 + z))])

    descending = redshift[::-1]
    solution = solve_ivp(
        derivative,
        (float(descending[0]), float(descending[-1])),
        np.zeros(1),
        t_eval=descending,
        method="Radau",
        rtol=2.0e-7,
        atol=1.0e-12,
        max_step=max((redshift[-1] - redshift[0]) / 300.0, 0.25),
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError("nonlocal second-order mean integration failed")
    second = solution.y[0, ::-1]
    return redshift, second + cross, second


def conservation_wavenumber(
    cosmology: lmhd.Cosmology,
    ionization,
    b0_gauss: float,
    redshift: np.ndarray | float,
) -> np.ndarray:
    """Paper Eq. (15), returned in comoving ``Mpc^-1``."""

    values = []
    for z in np.atleast_1d(np.asarray(redshift, dtype=float)):
        coeff = lmhd.background_coefficients(cosmology, float(z), ionization, b0_gauss)
        values.append(
            coeff["a"]
            * math.sqrt(coeff["alpha_drag"] * coeff["hubble"] / coeff["va2_fsr"])
            * lmhd.MPC_CM
        )
    result = np.asarray(values)
    return result[0] if np.ndim(redshift) == 0 else result


def acoustic_diffusion_wavenumber(
    cosmology: lmhd.Cosmology,
    ionization,
    redshift: Sequence[float] | np.ndarray,
    *,
    integration_points: int = 12000,
) -> np.ndarray:
    """Paper Eq. (7) acoustic diffusion wavenumber in ``Mpc^-1``.

    The manuscript uses ``c=1``. Restoring c gives
    ``k_D^-2 = integral c*lambda_comoving*dz/(6H) * (...)``.
    Integration begins at the paper's 1 MeV cutoff.
    """

    output = np.asarray(redshift, dtype=float)
    if output.size == 0 or np.any(output < 0.0):
        raise ValueError("diffusion redshifts must be nonnegative and nonempty")
    if integration_points < 128:
        raise ValueError("diffusion integration requires at least 128 points")
    zmax = lmhd.neutrino_decoupling_redshift(cosmology)
    if np.any(output >= zmax):
        raise ValueError("diffusion redshifts must lie below the 1 MeV cutoff")
    zmin = float(np.min(output))
    grid = np.geomspace(1.0 + zmin, 1.0 + zmax, integration_points) - 1.0
    integrand = np.empty_like(grid)
    for index, z in enumerate(grid):
        coeff = lmhd.background_coefficients(cosmology, float(z), ionization, 1.0e-12)
        ratio = coeff["R"]
        transport = (
            ratio * ratio + 16.0 * (1.0 + ratio) / 15.0
        ) / (1.0 + ratio) ** 2
        integrand[index] = (
            lmhd.CLIGHT
            * coeff["lambda_comoving"]
            * transport
            / (6.0 * coeff["hubble"])
        )
    reverse_integral = -cumulative_trapezoid(
        integrand[::-1], grid[::-1], initial=0.0
    )[::-1]
    inverse_square = np.interp(output, grid, reverse_integral)
    return lmhd.MPC_CM / np.sqrt(np.maximum(inverse_square, np.finfo(float).tiny))


def one_efold_damping_wavelength(
    k_mpc: Sequence[float] | np.ndarray,
    transfer: np.ndarray,
) -> np.ndarray:
    """Largest wavelength whose transfer amplitude is at most ``exp(-1)``.

    The first crossing is interpolated in log k and log amplitude. Columns
    without a resolved crossing return ``NaN`` rather than an extrapolation.
    """

    wave = np.asarray(k_mpc, dtype=float)
    amplitude = np.abs(np.asarray(transfer))
    if wave.ndim != 1 or np.any(wave <= 0.0) or np.any(np.diff(wave) <= 0.0):
        raise ValueError("damping extraction requires increasing positive k")
    if amplitude.shape[0] != wave.size:
        raise ValueError("transfer's first dimension must match k")
    flattened = amplitude.reshape(wave.size, -1)
    result = np.full(flattened.shape[1], np.nan)
    threshold = math.exp(-1.0)
    for column in range(flattened.shape[1]):
        values = flattened[:, column]
        finite = np.isfinite(values)
        crossings = np.nonzero(
            finite[1:]
            & finite[:-1]
            & (values[:-1] > threshold)
            & (values[1:] <= threshold)
        )[0]
        if crossings.size == 0:
            continue
        left = int(crossings[0])
        x0, x1 = np.log(wave[left : left + 2])
        y0, y1 = np.log(np.maximum(values[left : left + 2], 1.0e-300))
        fraction = (math.log(threshold) - y0) / (y1 - y0)
        crossing_k = math.exp(x0 + fraction * (x1 - x0))
        result[column] = 2.0 * math.pi / crossing_k
    return result.reshape(amplitude.shape[1:])


@dataclass(frozen=True)
class HomogeneousRecombination:
    """Homogeneous Sobolev and Fokker--Planck recombination histories."""

    redshift: np.ndarray
    saha_xe: np.ndarray
    sobolev_xe: np.ndarray
    fokker_planck_xe: np.ndarray
    response_redshift: np.ndarray
    sobolev_escape_probability: np.ndarray
    fokker_planck_escape_probability: np.ndarray
    iterations: int


def _integrate_escape_history(
    cosmology: lmhd.Cosmology,
    response_redshift: np.ndarray,
    escape_probability: np.ndarray,
    output_redshift: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    zmin = float(np.min(output_redshift))
    zmax = float(np.max(output_redshift))
    initial = lmhd.saha_ionization(cosmology, zmax)

    def derivative(redshift: float, value: np.ndarray) -> np.ndarray:
        z = float(np.clip(redshift, zmin, zmax))
        xe = float(np.clip(value[0], 1.0e-14, 1.0))
        temperature = cosmology.tcmb0 * (1.0 + z)
        hydrogen = cosmology.hydrogen_density(z)
        hubble = cosmology.hubble(z)
        alpha_b = lmhd._case_b_recombination_rate(temperature)
        beta_total = alpha_b * (
            2.0
            * math.pi
            * lmhd.MELECTRON
            * lmhd.KBOLTZ
            * temperature
            / lmhd.HPLANCK**2
        ) ** 1.5 * math.exp(
            -lmhd.E_LEVEL2 / (lmhd.KBOLTZ * temperature)
        )
        probability = float(
            np.interp(z, response_redshift, escape_probability)
        )
        effective_escape = 3.0 * lya_rt.A_LYA * probability + lmhd.LAMBDA_2S1S
        peebles = effective_escape / (effective_escape + beta_total)
        neutral = max(1.0 - xe, 0.0)
        net = peebles * (
            hydrogen * alpha_b * xe * xe
            - beta_total
            * neutral
            * math.exp(-lmhd.E_LYA / (lmhd.KBOLTZ * temperature))
        )
        return np.asarray([net / (hubble * (1.0 + z))])

    descending = np.sort(output_redshift)[::-1]
    solution = solve_ivp(
        derivative,
        (zmax, zmin),
        np.asarray([initial]),
        method="Radau",
        t_eval=descending,
        rtol=rtol,
        atol=atol,
        first_step=min(1.0, max((zmax - zmin) / 1000.0, 1.0e-3)),
        max_step=max((zmax - zmin) / 500.0, 0.25),
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError("homogeneous escape-history integration failed")
    return np.interp(output_redshift, solution.t[::-1], solution.y[0, ::-1])


def homogeneous_recombination_histories(
    cosmology: lmhd.Cosmology,
    *,
    zmin: float = 600.0,
    zmax: float = 1800.0,
    output_count: int = 601,
    response_count: int = 81,
    xmax: float = 1000.0,
    bins_per_width: float = 2.0,
    iterations: int = 2,
    rtol: float = 2.0e-9,
    atol: float = 1.0e-12,
) -> HomogeneousRecombination:
    """Solve paper Figure 16's homogeneous recombination comparison.

    The line escape probability is recomputed from the Fokker--Planck BVP
    along the current ionization history.  Two fixed-point iterations are
    sufficient because the correction to the Sobolev history is small.
    """

    if not (zmax > zmin >= 0.0):
        raise ValueError("homogeneous history requires zmax > zmin >= 0")
    if output_count < 3 or response_count < 3 or iterations < 1:
        raise ValueError("history grids and iteration count are too small")
    outputs = np.linspace(zmin, zmax, output_count)
    response_z = np.linspace(zmin, zmax, response_count)
    sobolev = lmhd.PeeblesIonization(
        cosmology,
        zmax=max(3500.0, zmax),
        zmin=min(500.0, zmin),
        samples=max(6000, output_count * 8),
        rtol=min(rtol, 2.0e-10),
        atol=atol,
    )
    sobolev_xe = np.asarray([sobolev(float(z)) for z in outputs])
    current_z = outputs.copy()
    current_xe = sobolev_xe.copy()
    fp_probability = np.empty_like(response_z)
    sobolev_probability = np.empty_like(response_z)

    for _ in range(iterations):
        def ionization(redshift: float) -> float:
            return float(np.interp(redshift, current_z, current_xe))

        for index, redshift in enumerate(response_z):
            background = lya_rt.lya_background(
                cosmology, float(redshift), ionization
            )
            normalized_background = replace(
                background, f_equilibrium=1.0, f_blackbody=0.0
            )
            line = lya_rt.solve_homogeneous_line(
                normalized_background,
                xmax=xmax,
                bins_per_width=bins_per_width,
                method="sparse",
            )
            fp_probability[index] = 1.0 - line.line_average
            sobolev_probability[index] = (
                -math.expm1(-background.tau_s) / background.tau_s
            )
        if np.any(fp_probability <= 0.0) or not np.all(np.isfinite(fp_probability)):
            raise FloatingPointError("nonphysical Fokker--Planck escape probability")
        current_xe = _integrate_escape_history(
            cosmology,
            response_z,
            fp_probability,
            outputs,
            rtol=rtol,
            atol=atol,
        )

    return HomogeneousRecombination(
        redshift=outputs,
        saha_xe=np.asarray(
            [lmhd.saha_ionization(cosmology, float(z)) for z in outputs]
        ),
        sobolev_xe=sobolev_xe,
        fokker_planck_xe=current_xe,
        response_redshift=response_z,
        sobolev_escape_probability=sobolev_probability.copy(),
        fokker_planck_escape_probability=fp_probability.copy(),
        iterations=iterations,
    )


def primordial_density_to_magnetic_ratio(
    k_mpc: np.ndarray | float,
    epsilon: float,
    theta: float,
    *,
    scalar_amplitude: float = PLANCK_AS,
    scalar_index: float = PLANCK_NS,
    pivot_mpc: float = PLANCK_PIVOT_MPC,
    ir_length_mpc: float = IR_LENGTH_MPC,
) -> np.ndarray:
    """Paper Eq. (71), independent of the mean-field amplitude."""

    k = np.asarray(k_mpc, dtype=float)
    k_lambda = 2.0 * math.pi / ir_length_mpc
    return (
        24.0
        / 5.0
        * math.sin(theta)
        * np.sqrt(
            0.5
            * scalar_amplitude
            * k ** (scalar_index - 1.0 - epsilon)
            / (pivot_mpc ** (scalar_index - 1.0) * k_lambda ** (-epsilon))
        )
    )


@dataclass(frozen=True)
class DensitySeedTransfer:
    """TCR transfer ratio used in the left panel of paper Figure 12."""

    k_mpc: np.ndarray
    z_free_streaming: np.ndarray
    ratio_gauss: np.ndarray


def density_seeded_magnetic_transfer(
    cosmology: lmhd.Cosmology,
    ionization,
    k_mpc: Sequence[float] | np.ndarray,
    b0_gauss: float,
    theta: float,
    *,
    max_dln_a: float = 2.0e-3,
) -> DensitySeedTransfer:
    """Numerically evaluate the transfer ratio in paper Eqs. (68)--(69).

    The two columns are propagated by the same TCR evolution operator.  One
    starts with ``b_y/B0=1``; the other starts with ``delta_r=1`` and
    ``b_y=0``.  Multiplication by ``B0`` converts the latter's normalized
    magnetic response to the dimensional transfer used in the manuscript.
    """

    wave = np.asarray(k_mpc, dtype=float)
    if wave.ndim != 1 or wave.size == 0 or np.any(wave <= 0.0):
        raise ValueError("k_mpc must be a non-empty positive one-dimensional grid")
    if b0_gauss <= 0.0 or not 0.0 < theta < 0.5 * math.pi:
        raise ValueError("b0_gauss and theta must define a physical mean field")
    cosine = math.cos(theta)
    sine = math.sin(theta)
    ratio = np.empty_like(wave)
    transition = np.empty_like(wave)
    for index, k_value in enumerate(wave):
        mode = lmhd.ReferenceMode(
            k_mpc=float(k_value),
            theta=theta,
            b0_gauss=b0_gauss,
            perturbation_fraction=1.0,
        )
        start = lmhd.sv_paper_initial_redshift(cosmology, mode.k_mpc)
        zfs = lmhd.free_streaming_redshift(
            mode, cosmology, ionization, 100.0, z_start=start
        )
        if zfs is None:
            raise ValueError(f"k={k_value:g} Mpc^-1 does not reach free streaming")
        states = np.zeros((6, 2), dtype=float)
        states[1, 0] = 1.0 / cosine
        states[1, 1] = -3.0 * math.tan(theta)
        states[2, 1] = -3.0
        matrix = lambda z, mode=mode: lmhd._sv_matrix(
            z,
            mode,
            cosmology,
            ionization,
            "tcr",
            False,
            False,
        )
        evolved = lmhd._advance_matrix(states, start, zfs, matrix, max_dln_a)
        by = cosine * evolved[1] - sine * evolved[2]
        if abs(by[0]) <= np.finfo(float).tiny:
            raise FloatingPointError("magnetic seed transfer vanished at free streaming")
        ratio[index] = b0_gauss * abs(by[1] / by[0])
        transition[index] = zfs
    return DensitySeedTransfer(wave, transition, ratio)


def lorentz_variance_integrand(
    q_over_k: np.ndarray | float,
    theta: np.ndarray | float,
    spectral_index: float,
) -> np.ndarray:
    """Dimensionless integrand of paper Eq. (65)."""

    ratio = np.asarray(q_over_k, dtype=float)
    angle = np.asarray(theta, dtype=float)
    ratio, angle = np.broadcast_arrays(ratio, angle)
    denominator = 1.0 + ratio**2 - 2.0 * ratio * np.cos(angle)
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    return (
        ratio ** (spectral_index + 3.0)
        * denominator ** (0.5 * spectral_index)
        * np.sin(angle)
        * (
            ratio * np.sin(angle) ** 2 / denominator
            + 2.0 * np.cos(angle)
        )
    )
