#!/usr/bin/env python3
"""Independent Float64 linear-MHD references for early primordial fields.

The implementation follows the equations and Fourier convention in
Schiff & Venumadhav, arXiv:2506.16517v2.  It also provides a continuous
baryon/photon P1 system for a like-for-like comparison with Vespa's evolved
photon moments.  All dimensional calculations use cgs units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, matrix_balance
from scipy.sparse.linalg import expm_multiply


CLIGHT = 2.99792458e10
GRAVITY = 6.67430e-8
SIGMA_T = 6.6524587321e-25
ARAD = 7.5657e-15
KBOLTZ = 1.380649e-16
HPLANCK = 6.62607015e-27
MELECTRON = 9.1093837139e-28
MHYDROGEN = 1.6735575e-24
MPC_CM = 3.0856775814913673e24
EV_ERG = 1.602176634e-12
LAMBDA_LYA = 1215.67e-8
LAMBDA_2S1S = 8.22458
E_ION = 13.59843449 * EV_ERG
E_LEVEL2 = E_ION / 4.0
E_LYA = E_ION - E_LEVEL2
CHI_HEI_OVER_K = 285335.0
CHI_HEII_OVER_K = 631515.0
REFERENCE_REVISION = "sv25-v2-eta-prime-c-lambda-over-15"


@dataclass(frozen=True)
class Cosmology:
    """Flat-background parameters used by the reference calculations."""

    h: float = 0.674
    omega_m: float = 0.315
    omega_b: float = 0.05366655
    omega_lambda: float = 0.684908645845
    omega_r: float = 9.13541547429e-5
    hydrogen_mass_fraction: float = 0.7546
    tcmb0: float = 2.725

    def __post_init__(self) -> None:
        if not (0 < self.hydrogen_mass_fraction < 1):
            raise ValueError("hydrogen_mass_fraction must be in (0, 1)")
        if min(self.omega_m, self.omega_b, self.omega_r) <= 0:
            raise ValueError("matter, baryon, and radiation densities must be positive")
        if self.omega_b > self.omega_m:
            raise ValueError("omega_b cannot exceed omega_m")

    @property
    def h0(self) -> float:
        return 100.0 * self.h * 1.0e5 / MPC_CM

    @property
    def omega_k(self) -> float:
        return 1.0 - self.omega_m - self.omega_lambda - self.omega_r

    @property
    def helium_per_hydrogen(self) -> float:
        xh = self.hydrogen_mass_fraction
        return (1.0 - xh) / (4.0 * xh)

    def hubble(self, z: float) -> float:
        zp1 = 1.0 + float(z)
        return self.h0 * math.sqrt(
            self.omega_r * zp1**4
            + self.omega_m * zp1**3
            + self.omega_k * zp1**2
            + self.omega_lambda
        )

    def critical_density0(self) -> float:
        return 3.0 * self.h0**2 / (8.0 * math.pi * GRAVITY)

    def baryon_density(self, z: float) -> float:
        return self.omega_b * self.critical_density0() * (1.0 + z) ** 3

    def dark_matter_density(self, z: float) -> float:
        return (self.omega_m - self.omega_b) * self.critical_density0() * (
            1.0 + z
        ) ** 3

    def photon_energy_density(self, z: float) -> float:
        return ARAD * (self.tcmb0 * (1.0 + z)) ** 4

    def hydrogen_density(self, z: float) -> float:
        return (
            self.hydrogen_mass_fraction
            * self.baryon_density(z)
            / MHYDROGEN
        )


class FullyIonized:
    """Hydrogen ionization history with x_e=1."""

    def __call__(self, z: float) -> float:
        del z
        return 1.0


def saha_ionization(cosmology: Cosmology, z: float) -> float:
    """Hydrogen Saha ionization fraction, excluding helium electrons."""

    temperature = cosmology.tcmb0 * (1.0 + z)
    phase = (2.0 * math.pi * MELECTRON * KBOLTZ * temperature) ** 1.5 / (
        HPLANCK**3
    )
    ratio = phase * math.exp(-E_ION / (KBOLTZ * temperature)) / max(
        cosmology.hydrogen_density(z), np.finfo(float).tiny
    )
    if ratio <= 0:
        return 0.0
    # Stable positive root of x^2/(1-x)=ratio.
    return min(1.0, 2.0 * ratio / (math.sqrt(ratio * ratio + 4.0 * ratio) + ratio))


def helium_saha_electrons(
    cosmology: Cosmology, z: float, hydrogen_xe: float, iterations: int = 8
) -> float:
    """Return total n_e/n_H with helium in CMB Saha equilibrium.

    A fixed two-electron helium term is valid only before helium recombination;
    using it near hydrogen recombination substantially overestimates drag.
    """

    temperature = cosmology.tcmb0 * (1.0 + z)
    quantum_density = (
        2.0 * math.pi * MELECTRON * KBOLTZ * temperature / HPLANCK**2
    ) ** 1.5
    saha1 = 4.0 * quantum_density * math.exp(-CHI_HEI_OVER_K / temperature)
    saha2 = quantum_density * math.exp(-CHI_HEII_OVER_K / temperature)
    xe = max(float(hydrogen_xe), 0.0)
    n_h = cosmology.hydrogen_density(z)
    for _ in range(iterations):
        n_e = max(xe * n_h, np.finfo(float).tiny)
        ratio1 = saha1 / n_e
        ratio2 = saha2 / n_e
        denominator = 1.0 + ratio1 + ratio1 * ratio2
        xe = hydrogen_xe + cosmology.helium_per_hydrogen * (
            ratio1 + 2.0 * ratio1 * ratio2
        ) / denominator
    return float(xe)


def _case_b_recombination_rate(temperature: float) -> float:
    t4 = max(temperature / 1.0e4, 1.0e-12)
    return 4.309e-13 * t4 ** (-0.6166) / (1.0 + 0.6703 * t4**0.5300)


def case_b_recombination_log_slope(temperature: float) -> float:
    """Return ``d ln(alpha_B) / d ln(T)`` for the RECFAST fit."""

    t4 = max(float(temperature) / 1.0e4, 1.0e-12)
    correction = 0.6703 * t4**0.5300
    return -0.6166 - 0.5300 * correction / (1.0 + correction)


def recfast_v2_gaussian_factor(redshift: float) -> float:
    logzp1 = math.log1p(float(redshift))
    return (
        1.0
        - 0.14 * math.exp(-((logzp1 - 7.28) / 0.18) ** 2)
        + 0.079 * math.exp(-((logzp1 - 6.73) / 0.33) ** 2)
    )


def _excited_photoionization_rate(temperature: float) -> float:
    quantum_density = (
        2.0 * math.pi * MELECTRON * KBOLTZ * temperature / HPLANCK**2
    ) ** 1.5
    return (
        _case_b_recombination_rate(temperature)
        * quantum_density
        * math.exp(-E_LEVEL2 / (KBOLTZ * temperature))
    )


def _recfast_v2_electron_rhs(
    cosmology: Cosmology,
    redshift: float,
    hydrogen_density: float,
    xe: float,
    baryon_temperature: float,
) -> float:
    """Hydrogen `dx_e/dt` with matter/radiation temperatures separated."""

    neutral_density = max(hydrogen_density * (1.0 - xe), 0.0)
    alpha_b = _case_b_recombination_rate(baryon_temperature)
    radiation_temperature = cosmology.tcmb0 * (1.0 + redshift)
    beta_b = _excited_photoionization_rate(radiation_temperature)
    escape_k = (
        recfast_v2_gaussian_factor(redshift)
        * LAMBDA_LYA**3
        / (8.0 * math.pi * cosmology.hubble(redshift))
    )
    kl = escape_k * LAMBDA_2S1S * neutral_density
    kb = escape_k * beta_b * neutral_density
    fudge = 1.125
    peebles = fudge * (1.0 + kl) / (1.0 + kl + fudge * kb)
    ground_photoionization = beta_b * math.exp(
        -E_LYA / (KBOLTZ * radiation_temperature)
    )
    return peebles * (
        -alpha_b * hydrogen_density * xe * xe
        + ground_photoionization * (1.0 - xe)
    )


def recfast_v2_local_response(
    cosmology: Cosmology,
    redshift: float,
    xe: float,
    baryon_temperature: float,
    relative_step: float = 1.0e-5,
) -> np.ndarray:
    """Return local `(delta_b, delta_xe, Theta/aH, delta_T/T)` rates.

    The derivatives are evaluated independently from the RECFAST-v2 scalar
    equation. The local chemistry has no velocity-gradient closure, so that
    component is exactly zero; line transport supplies its full response.
    """

    n_h = cosmology.hydrogen_density(redshift)
    step = float(relative_step)
    if not 0.0 < step < 0.1:
        raise ValueError("relative_step must be in (0, 0.1)")
    density_rate = (
        _recfast_v2_electron_rhs(
            cosmology, redshift, n_h * (1.0 + step), xe, baryon_temperature
        )
        - _recfast_v2_electron_rhs(
            cosmology, redshift, n_h * (1.0 - step), xe, baryon_temperature
        )
    ) / (2.0 * step)
    xe_step = step * max(xe, 1.0e-3)
    ionization_rate = (
        _recfast_v2_electron_rhs(
            cosmology, redshift, n_h, xe + xe_step, baryon_temperature
        )
        - _recfast_v2_electron_rhs(
            cosmology, redshift, n_h, xe - xe_step, baryon_temperature
        )
    ) / (2.0 * xe_step)
    temperature_rate = (
        _recfast_v2_electron_rhs(
            cosmology, redshift, n_h, xe,
            baryon_temperature * (1.0 + step),
        )
        - _recfast_v2_electron_rhs(
            cosmology, redshift, n_h, xe,
            baryon_temperature * (1.0 - step),
        )
    ) / (2.0 * step)
    return np.asarray(
        (density_rate, ionization_rate, 0.0, temperature_rate), dtype=float
    )


def _hydrogen_atomic_cooling(
    redshift: float,
    hydrogen_density: float,
    xe: float,
    temperature: float,
    radiation_temperature: float,
    electrons_per_hydrogen: Optional[float] = None,
) -> float:
    """Abel--Anninos hydrogen plus Compton cooling in erg cm^-3 s^-1."""

    neutral = max(hydrogen_density * (1.0 - xe), 0.0)
    ionized = max(hydrogen_density * xe, 0.0)
    electrons = hydrogen_density * max(
        xe if electrons_per_hydrogen is None else electrons_per_hydrogen,
        0.0,
    )
    t = max(float(temperature), 1.0)
    excitation = 7.5e-19 * math.exp(-min(math.log(1.0e30), 118348.0 / t)) / (
        1.0 + math.sqrt(t / 1.0e5)
    )
    tev = t / 11605.0
    log_tev = math.log(tev)
    collisional_rate = math.exp(
        -32.71396786375
        + log_tev * (
            13.53655609057
            + log_tev * (
                -5.739328757388
                + log_tev * (
                    1.563154982022
                    + log_tev * (
                        -0.2877056004391
                        + log_tev * (
                            0.03482559773736999
                            + log_tev * (
                                -0.00263197617559
                                + log_tev * (
                                    0.0001119543953861
                                    - log_tev * 2.039149852002e-6
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    collisional_ionization = 2.18e-11 * collisional_rate
    lambda_h = 2.0 * 157807.0 / t
    recombination = (
        3.435e-30
        * t
        * lambda_h**1.970
        / (1.0 + (lambda_h / 2.25) ** 0.376) ** 3.720
    )
    bremsstrahlung = (
        1.43e-27
        * math.sqrt(t)
        * (1.1 + 0.34 * math.exp(-((5.5 - math.log10(t)) ** 2) / 3.0))
    )
    atomic = (
        (excitation + collisional_ionization) * neutral * electrons
        + (recombination + bremsstrahlung) * ionized * electrons
    )
    compton = (
        5.65e-36
        * (1.0 + redshift) ** 4
        * (t - radiation_temperature)
        * electrons
    )
    return atomic + compton


def recfast_v2_temperature_response(
    cosmology: Cosmology,
    redshift: float,
    xe: float,
    baryon_temperature: float,
    recombination_rates: Sequence[float],
    relative_step: float = 1.0e-5,
    electrons_per_hydrogen: Optional[float] = None,
) -> np.ndarray:
    """Linearized local temperature RHS for the runtime energy convention.

    The returned derivatives multiply ``(delta_b, delta_xe, theta_b,
    delta_T)``. The ionization response may include a nonlocal correction;
    changing electron number then changes temperature at fixed internal energy.
    H2 cooling is omitted because the matched tests have ``f_H2 < 4e-12``.
    """

    rates = np.asarray(tuple(recombination_rates), dtype=float)
    if rates.shape != (4,):
        raise ValueError("temperature response needs four recombination rates")
    step = float(relative_step)
    if not 0.0 < step < 0.1:
        raise ValueError("relative_step must be in (0, 0.1)")
    hubble = cosmology.hubble(redshift)
    rho_background = cosmology.baryon_density(redshift)
    xh = cosmology.hydrogen_mass_fraction
    helium_mass_fraction = 1.0 - xh
    radiation_temperature = cosmology.tcmb0 * (1.0 + redshift)
    gamma_minus_one = 2.0 / 3.0
    energy_factor = KBOLTZ / (gamma_minus_one * MHYDROGEN)
    xdot_background = _recfast_v2_electron_rhs(
        cosmology,
        redshift,
        cosmology.hydrogen_density(redshift),
        xe,
        baryon_temperature,
    )
    helium_electrons = max(
        0.0,
        (xe if electrons_per_hydrogen is None else electrons_per_hydrogen) - xe,
    )

    def rhs(delta: float, delta_x: float, theta: float, delta_t: float) -> float:
        rho = rho_background * (1.0 + delta)
        x_local = xe + delta_x
        temperature = baryon_temperature + delta_t
        hydrogen_density = xh * rho / MHYDROGEN
        particles_per_mass_h = (
            xh * (1.0 + x_local + helium_electrons)
            + helium_mass_fraction / 4.0
        )
        specific_energy = energy_factor * particles_per_mass_h * temperature
        xdot = (
            xdot_background
            + rates[0] * delta
            + rates[1] * delta_x
            + rates[2] * theta / hubble
            + rates[3] * delta_t / baryon_temperature
        )
        cooling = _hydrogen_atomic_cooling(
            redshift,
            hydrogen_density,
            x_local,
            temperature,
            radiation_temperature,
            x_local + helium_electrons,
        )
        specific_energy_dot = (
            -2.0 * hubble * specific_energy
            - gamma_minus_one * specific_energy * theta
            - cooling / rho
        )
        return (
            specific_energy_dot / energy_factor - temperature * xh * xdot
        ) / particles_per_mass_h

    def derivative(index: int, increment: float) -> float:
        plus = [0.0, 0.0, 0.0, 0.0]
        minus = [0.0, 0.0, 0.0, 0.0]
        plus[index] = increment
        minus[index] = -increment
        return (rhs(*plus) - rhs(*minus)) / (2.0 * increment)

    return np.asarray(
        (
            derivative(0, step),
            derivative(1, step * max(xe, 1.0e-3)),
            derivative(2, step * hubble),
            derivative(3, step * baryon_temperature),
        ),
        dtype=float,
    )


class PeeblesIonization:
    """Independent homogeneous three-level-atom hydrogen history.

    This is a coefficient model for MHD verification, not the campaign's
    nonlocal Lyman-alpha oracle.  The latter must pass separate hierarchy
    convergence tests.
    """

    def __init__(
        self,
        cosmology: Cosmology,
        zmax: float = 3500.0,
        zmin: float = 100.0,
        samples: int = 12000,
        rtol: float = 2.0e-10,
        atol: float = 1.0e-12,
        recfast_v2: bool = False,
    ) -> None:
        if not (zmax > zmin >= 0):
            raise ValueError("Peebles history requires zmax > zmin >= 0")
        self.cosmology = cosmology
        self.zmax = float(zmax)
        self.zmin = float(zmin)
        initial = saha_ionization(cosmology, zmax)

        def derivative(z: float, value: np.ndarray) -> np.ndarray:
            xe = float(np.clip(value[0], 1.0e-12, 1.0))
            temperature = cosmology.tcmb0 * (1.0 + z)
            nh = cosmology.hydrogen_density(z)
            hubble = cosmology.hubble(z)
            alpha_b = _case_b_recombination_rate(temperature)
            beta_b = alpha_b * (
                2.0 * math.pi * MELECTRON * KBOLTZ * temperature / HPLANCK**2
            ) ** 1.5 * math.exp(-E_LEVEL2 / (KBOLTZ * temperature))
            neutral = max(1.0 - xe, 0.0)
            if recfast_v2:
                logzp1 = math.log1p(z)
                gaussian = (
                    1.0
                    - 0.14 * math.exp(-((logzp1 - 7.28) / 0.18) ** 2)
                    + 0.079 * math.exp(-((logzp1 - 6.73) / 0.33) ** 2)
                )
                fudge = 1.125
            else:
                gaussian = 1.0
                fudge = 1.0
            k_escape = (
                gaussian * LAMBDA_LYA**3 / (8.0 * math.pi * hubble)
            )
            kl = k_escape * LAMBDA_2S1S * nh * neutral
            kb = k_escape * beta_b * nh * neutral
            c_factor = fudge * (1.0 + kl) / (1.0 + kl + fudge * kb)
            net = c_factor * (
                nh * alpha_b * xe * xe
                - beta_b * neutral * math.exp(-E_LYA / (KBOLTZ * temperature))
            )
            return np.asarray([net / (hubble * (1.0 + z))])

        redshifts = np.linspace(zmax, zmin, samples)
        solution = solve_ivp(
            derivative,
            (zmax, zmin),
            np.asarray([initial]),
            method="Radau",
            t_eval=redshifts,
            rtol=rtol,
            atol=atol,
            first_step=1.0,
            max_step=5.0,
        )
        if not solution.success:
            raise RuntimeError("Peebles history failed: " + solution.message)
        self._z_ascending = solution.t[::-1].copy()
        self._xe_ascending = np.clip(solution.y[0, ::-1], 0.0, 1.0).copy()

    def __call__(self, z: float) -> float:
        z = float(z)
        if z >= self.zmax:
            return saha_ionization(self.cosmology, z)
        if z <= self.zmin:
            return float(self._xe_ascending[0])
        return float(np.interp(z, self._z_ascending, self._xe_ascending))

    def table(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._z_ascending.copy(), self._xe_ascending.copy()


def compton_temperature_rate(
    cosmology: Cosmology, z: float, hydrogen_xe: float
) -> float:
    """Return the baryon Compton thermalization rate in s^-1."""

    xe = max(float(hydrogen_xe), 0.0)
    denominator = 1.0 + cosmology.helium_per_hydrogen + xe
    return (
        8.0
        * cosmology.photon_energy_density(z)
        * SIGMA_T
        * xe
        / (3.0 * MELECTRON * CLIGHT * denominator)
    )


class ComptonTemperature:
    """Independent homogeneous baryon-temperature history.

    This integrates Schiff--Venumadhav Eq. (65) without perturbative or
    dissipative heating. The supplied ionization history is hydrogen
    ``x_e=n_e/n_H``; helium contributes to the heat capacity.
    """

    def __init__(
        self,
        cosmology: Cosmology,
        ionization: Callable[[float], float],
        zmax: float = 3500.0,
        zmin: float = 100.0,
        samples: int = 12000,
        rtol: float = 2.0e-10,
        atol: float = 1.0e-8,
    ) -> None:
        if not (zmax > zmin >= 0):
            raise ValueError("temperature history requires zmax > zmin >= 0")
        self.cosmology = cosmology
        self.ionization = ionization
        self.zmax = float(zmax)
        self.zmin = float(zmin)

        def derivative(z: float, value: np.ndarray) -> np.ndarray:
            tb = max(float(value[0]), 1.0)
            hubble = cosmology.hubble(z)
            tgamma = cosmology.tcmb0 * (1.0 + z)
            gamma_c = compton_temperature_rate(cosmology, z, ionization(z))
            return np.asarray(
                [(2.0 * hubble * tb - gamma_c * (tgamma - tb)) /
                 (hubble * (1.0 + z))]
            )

        redshifts = np.linspace(zmax, zmin, samples)
        initial = cosmology.tcmb0 * (1.0 + zmax)
        solution = solve_ivp(
            derivative,
            (zmax, zmin),
            np.asarray([initial]),
            method="Radau",
            t_eval=redshifts,
            rtol=rtol,
            atol=atol,
            first_step=1.0,
            max_step=5.0,
        )
        if not solution.success:
            raise RuntimeError("baryon-temperature history failed: " + solution.message)
        self._z_ascending = solution.t[::-1].copy()
        self._temperature_ascending = solution.y[0, ::-1].copy()

    def __call__(self, z: float) -> float:
        z = float(z)
        if z >= self.zmax:
            return self.cosmology.tcmb0 * (1.0 + z)
        if z <= self.zmin:
            return float(self._temperature_ascending[0])
        return float(
            np.interp(z, self._z_ascending, self._temperature_ascending)
        )

    def table(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._z_ascending.copy(), self._temperature_ascending.copy()


class TabulatedThermochemistry:
    """Interpolated ``z, x_HII, T_b`` history from a CSV or TSV fixture."""

    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        rows: list[tuple[float, float, float]] = []
        for raw_line in source.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                fields = line.split("\t")
            elif "," in line:
                fields = line.split(",")
            else:
                fields = line.split()
            if len(fields) < 3:
                continue
            try:
                row = tuple(float(value.strip()) for value in fields[:3])
            except ValueError:
                continue
            rows.append(row)
        rows.sort()
        if len(rows) < 2:
            raise ValueError(f"invalid thermochemistry history: {source}")
        values = np.asarray(rows, dtype=float)
        if not (
            np.all(np.isfinite(values))
            and np.all(np.diff(values[:, 0]) > 0.0)
            # Some RECFAST fixtures include helium electrons and therefore
            # legitimately exceed one electron per hydrogen nucleus at high z.
            and np.all((values[:, 1] >= 0.0) & (values[:, 1] <= 2.0))
            and np.all(values[:, 2] > 0.0)
        ):
            raise ValueError(f"invalid thermochemistry values: {source}")
        self.path = source
        self.redshift = values[:, 0]
        self.x_hii = values[:, 1]
        self.baryon_temperature = values[:, 2]

    def _interpolate(self, values: np.ndarray, redshift: float) -> float:
        z = float(redshift)
        if not self.redshift[0] <= z <= self.redshift[-1]:
            raise ValueError(
                f"redshift {z:g} is outside thermochemistry history {self.path}"
            )
        return float(np.interp(z, self.redshift, values))

    def ionization(self, redshift: float) -> float:
        return self._interpolate(self.x_hii, redshift)

    def clamped_ionization(self, redshift: float) -> float:
        """Return ionization with endpoint values outside the tabulated range."""

        return float(
            np.interp(float(redshift), self.redshift, self.x_hii)
        )

    def temperature(self, redshift: float) -> float:
        return self._interpolate(self.baryon_temperature, redshift)

    def __call__(self, redshift: float) -> float:
        return self.ionization(redshift)


@dataclass(frozen=True)
class ReferenceMode:
    """One Fourier mode in the k=z-axis, B0=y-z-plane convention."""

    k_mpc: float
    theta: float
    b0_gauss: float
    perturbation_fraction: float = 1.0e-4
    polarization: str = "magnetosonic"

    def __post_init__(self) -> None:
        if self.k_mpc <= 0 or self.b0_gauss <= 0:
            raise ValueError("k_mpc and b0_gauss must be positive")
        if not (0 < self.theta < math.pi / 2):
            raise ValueError("theta must be strictly between 0 and pi/2")
        if self.perturbation_fraction <= 0:
            raise ValueError("perturbation_fraction must be positive")
        if self.polarization not in ("magnetosonic", "alfven"):
            raise ValueError("polarization must be magnetosonic or alfven")

    @property
    def k_comoving_cgs(self) -> float:
        return self.k_mpc / MPC_CM


@dataclass
class EvolutionResult:
    model: str
    redshift: np.ndarray
    fields: Dict[str, np.ndarray]
    metadata: Dict[str, object]

    def field(self, name: str) -> np.ndarray:
        return self.fields[name]


def background_coefficients(
    cosmology: Cosmology,
    z: float,
    ionization: Callable[[float], float],
    b0_gauss: float,
    include_helium_electrons: bool = False,
    baryon_temperature: Optional[Callable[[float], float]] = None,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> Dict[str, float]:
    """Return dimensional coefficients shared by both linear models."""

    a = 1.0 / (1.0 + z)
    temperature = (
        cosmology.tcmb0 / a
        if baryon_temperature is None
        else max(float(baryon_temperature(z)), 1.0)
    )
    rho_b = cosmology.baryon_density(z)
    photon_energy = cosmology.photon_energy_density(z)
    inertia_ratio = 3.0 * rho_b * CLIGHT**2 / (4.0 * photon_energy)
    xe = max(float(ionization(z)), np.finfo(float).eps)
    if background_electron_fraction is not None:
        electrons_per_h = max(
            float(background_electron_fraction(z)), np.finfo(float).eps
        )
    else:
        electrons_per_h = (
            helium_saha_electrons(cosmology, z, xe)
            if include_helium_electrons else xe
        )
    ne = cosmology.hydrogen_density(z) * electrons_per_h
    lambda_physical = 1.0 / (ne * SIGMA_T)
    alpha_drag = ne * SIGMA_T * CLIGHT / inertia_ratio
    b_physical = b0_gauss / (a * a)
    va2_fsr = b_physical**2 / (4.0 * math.pi * rho_b)
    va2_tcr = (
        b_physical**2
        * CLIGHT**2
        / (4.0 * math.pi * (4.0 / 3.0) * photon_energy * (1.0 + inertia_ratio))
    )
    particles_per_h = (
        1.0 + cosmology.helium_per_hydrogen + electrons_per_h
    )
    cs2_baryon = (
        KBOLTZ
        * temperature
        * cosmology.hydrogen_mass_fraction
        * particles_per_h
        / MHYDROGEN
    )
    return {
        "a": a,
        "temperature": temperature,
        "hubble": cosmology.hubble(z),
        "rho_b": rho_b,
        "rho_dm": cosmology.dark_matter_density(z),
        "rho_gamma_energy": photon_energy,
        "R": inertia_ratio,
        "xe": xe,
        "electrons_per_h": electrons_per_h,
        "ne": ne,
        "lambda_physical": lambda_physical,
        "lambda_comoving": lambda_physical / a,
        "alpha_drag": alpha_drag,
        "nu_scatter": inertia_ratio * alpha_drag,
        "va2_tcr": va2_tcr,
        "va2_fsr": va2_fsr,
        "cs2_tcr": CLIGHT**2 / (3.0 * (1.0 + inertia_ratio)),
        "cs2_baryon": cs2_baryon,
        # SV25 Table I writes the transverse/longitudinal coefficients as
        # 3*eta' and 4*eta'. From eta=(4/15)rho_gamma*c*lambda_phys and
        # rho_gamma+p_gamma=(4/3)rho_gamma, eta'=eta/[3(rho+p)].
        "eta_prime": CLIGHT * lambda_physical / 15.0,
        "particles_per_h": particles_per_h,
        "compton_temperature_rate": compton_temperature_rate(cosmology, z, xe),
    }


def wavenumber_for_q(
    cosmology: Cosmology,
    z: float,
    q: float,
    ionization: Callable[[float], float],
) -> float:
    """Comoving k [Mpc^-1] whose k*lambda_comoving equals q at z."""

    if q <= 0:
        raise ValueError("q must be positive")
    coeff = background_coefficients(cosmology, z, ionization, 1.0e-12)
    return q * MPC_CM / coeff["lambda_comoving"]


def neutrino_decoupling_redshift(
    cosmology: Cosmology, temperature_ev: float = 1.0e6
) -> float:
    """Return the redshift corresponding to the paper's neutrino cutoff.

    Schiff & Venumadhav initialize modes no earlier than ``T = 1 MeV``.  The
    accepted manuscript treats the photon temperature as ``T_gamma=T0/a`` in
    this interval, which is the convention used here as well.
    """

    if temperature_ev <= 0:
        raise ValueError("temperature_ev must be positive")
    temperature = temperature_ev * EV_ERG / KBOLTZ
    return temperature / cosmology.tcmb0 - 1.0


def horizon_entry_redshift(
    cosmology: Cosmology,
    k_mpc: float,
    *,
    maximum_redshift: float = 1.0e16,
) -> float:
    """Solve ``c k = a H`` for a comoving wavenumber in ``Mpc^-1``."""

    if k_mpc <= 0:
        raise ValueError("k_mpc must be positive")
    if maximum_redshift <= 0:
        raise ValueError("maximum_redshift must be positive")
    target = CLIGHT * k_mpc / MPC_CM

    def residual(logzp1: float) -> float:
        zp1 = math.exp(logzp1)
        return math.log(cosmology.hubble(zp1 - 1.0) / zp1) - math.log(target)

    lower = 0.0
    upper = math.log1p(maximum_redshift)
    if residual(lower) >= 0.0:
        return 0.0
    if residual(upper) <= 0.0:
        raise ValueError(
            f"horizon entry for k={k_mpc:g} Mpc^-1 exceeds "
            f"z={maximum_redshift:g}"
        )
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return math.expm1(0.5 * (lower + upper))


def sv_paper_initial_redshift(cosmology: Cosmology, k_mpc: float) -> float:
    """Paper IC epoch: horizon entry or 1 MeV, whichever occurs later."""

    return min(
        horizon_entry_redshift(cosmology, k_mpc),
        neutrino_decoupling_redshift(cosmology),
    )


def _angular_transport(q2: float) -> Tuple[float, float, float]:
    if q2 < 1.0e-3:
        q4 = q2 * q2
        q6 = q4 * q2
        a0 = 1.0 - q2 / 3.0 + q4 / 5.0 - q6 / 7.0
        along = 1.0 - 3.0 * q2 / 5.0 + 3.0 * q4 / 7.0 - q6 / 3.0
    else:
        q = math.sqrt(q2)
        a0 = math.atan(q) / q
        along = 3.0 * (1.0 - a0) / q2
    atrans = 0.5 * (3.0 * a0 - along)
    return a0, along, atrans


def _advance_matrix(
    state: np.ndarray,
    z0: float,
    z1: float,
    matrix: Callable[[float], np.ndarray],
    max_dln_a: float,
    method: str = "cf4",
) -> np.ndarray:
    if z1 > z0 * (1.0 + 1.0e-14):
        raise ValueError("evolution redshifts must be non-increasing")
    x0 = -math.log1p(z0)
    x1 = -math.log1p(z1)
    span = x1 - x0
    if span <= 1.0e-15:
        return state.copy()
    nsteps = max(1, int(math.ceil(span / max_dln_a)))
    dx = span / nsteps
    value = state.copy()
    if value.size == 7 and method == "cf4":
        # CF4 has a negative coefficient in each exponential. That is not
        # L-stable when a stiff recombination block activates between its two
        # samples. Exponential midpoint keeps a positive dissipative step.
        method = "midpoint2"

    def exponential_action(generator: np.ndarray, vector: np.ndarray) -> np.ndarray:
        # Photon scattering and gravity can differ by many orders of magnitude.
        # A diagonal similarity balance keeps weak off-diagonal transfer blocks
        # in the useful Float64 range without altering the represented system.
        balanced, transform = matrix_balance(
            generator, permute=False, scale=True, separate=False
        )
        scaled = np.linalg.solve(transform, vector)
        result = transform.dot(expm(balanced).dot(scaled))
        if np.all(np.isfinite(result)) or generator.shape != (7, 7):
            return result
        return np.asarray(
            expm_multiply(
                generator,
                vector,
                traceA=float(np.real(np.trace(generator))),
            )
        )

    for step in range(nsteps):
        xmid = x0 + (step + 0.5) * dx
        if method == "midpoint2":
            zmid = math.exp(-xmid) - 1.0
            propagator = matrix(zmid) * dx
        elif method in ("cf4", "magnus4"):
            offset = math.sqrt(3.0) * dx / 6.0
            zleft = math.exp(-(xmid - offset)) - 1.0
            zright = math.exp(-(xmid + offset)) - 1.0
            left = matrix(zleft)
            right = matrix(zright)
            if method == "cf4":
                low = (3.0 - 2.0 * math.sqrt(3.0)) / 12.0
                high = (3.0 + 2.0 * math.sqrt(3.0)) / 12.0
                value = exponential_action(
                    dx * (high * left + low * right), value
                )
                value = exponential_action(
                    dx * (low * left + high * right), value
                )
                zmid = math.exp(-xmid) - 1.0
                propagator = None
            else:
                commutator = left.dot(right) - right.dot(left)
                propagator = (
                    0.5 * dx * (left + right)
                    - math.sqrt(3.0) * dx * dx * commutator / 12.0
                )
            zmid = math.exp(-xmid) - 1.0
        else:
            raise ValueError("matrix evolution method must be midpoint2, cf4, or magnus4")
        if propagator is not None:
            value = exponential_action(propagator, value)
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(
                "non-finite matrix evolution at z={:.9g}".format(zmid)
            )
    return value


def _requested_redshifts(
    z_start: float, z_end: float, outputs: Iterable[float]
) -> np.ndarray:
    if not (z_start > z_end >= 0):
        raise ValueError("requires z_start > z_end >= 0")
    selected = {float(z_start), float(z_end)}
    selected.update(float(z) for z in outputs if z_end <= float(z) <= z_start)
    return np.asarray(sorted(selected, reverse=True), dtype=float)


def _find_free_streaming_redshift(
    mode: ReferenceMode,
    cosmology: Cosmology,
    ionization: Callable[[float], float],
    z_start: float,
    z_end: float,
    include_helium_electrons: bool,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> Optional[float]:
    def log_q(z: float) -> float:
        coeff = background_coefficients(
            cosmology,
            z,
            ionization,
            mode.b0_gauss,
            include_helium_electrons,
            background_electron_fraction=background_electron_fraction,
        )
        q = mode.k_comoving_cgs * coeff["lambda_comoving"]
        return math.log(q)

    grid = np.geomspace(1.0 + z_end, 1.0 + z_start, 4096)[::-1] - 1.0
    values = np.asarray([log_q(float(z)) for z in grid])
    if values[0] >= 0:
        return z_start
    crossings = np.nonzero((values[:-1] < 0) & (values[1:] >= 0))[0]
    if crossings.size == 0:
        return None
    lo = float(grid[crossings[0] + 1])
    hi = float(grid[crossings[0]])
    # z decreases from hi to lo while q increases. Bisect in ln(1+z).
    xlo = math.log1p(lo)
    xhi = math.log1p(hi)
    for _ in range(80):
        xmid = 0.5 * (xlo + xhi)
        zmid = math.exp(xmid) - 1.0
        if log_q(zmid) >= 0:
            xlo = xmid
        else:
            xhi = xmid
    return math.exp(0.5 * (xlo + xhi)) - 1.0


def free_streaming_redshift(
    mode: ReferenceMode,
    cosmology: Cosmology,
    ionization: Callable[[float], float],
    z_end: float,
    *,
    z_start: Optional[float] = None,
    include_helium_electrons: bool = False,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> Optional[float]:
    """Return the paper's ``k lambda_mfp = 1`` transition redshift.

    When ``z_start`` is omitted, the mode is followed from the physical SV25
    initialization epoch.  Exposing this quantity avoids reproducing the
    transition search in spectrum-level analysis code.
    """

    start = (
        sv_paper_initial_redshift(cosmology, mode.k_mpc)
        if z_start is None
        else float(z_start)
    )
    if start <= z_end:
        raise ValueError("free-streaming search requires z_start > z_end")
    return _find_free_streaming_redshift(
        mode,
        cosmology,
        ionization,
        start,
        float(z_end),
        include_helium_electrons,
        background_electron_fraction,
    )


def _sv_matrix(
    z: float,
    mode: ReferenceMode,
    cosmology: Cosmology,
    ionization: Callable[[float], float],
    regime: str,
    include_gravity: bool,
    include_helium_electrons: bool,
    baryon_temperature: Optional[Callable[[float], float]] = None,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    coeff = background_coefficients(
        cosmology,
        z,
        ionization,
        mode.b0_gauss,
        include_helium_electrons,
        baryon_temperature,
        background_electron_fraction,
    )
    hubble = coeff["hubble"]
    kphysical = mode.k_comoving_cgs / coeff["a"]
    sine = math.sin(mode.theta)
    cosine = math.cos(mode.theta)
    if regime == "tcr":
        diffusion = coeff["eta_prime"] * kphysical**2
        gamma = np.asarray(
            [
                (hubble + 2.0 * hubble * coeff["R"] + 3.0 * diffusion)
                / (1.0 + coeff["R"]),
                (hubble + 2.0 * hubble * coeff["R"] + 3.0 * diffusion)
                / (1.0 + coeff["R"]),
                (hubble + 2.0 * hubble * coeff["R"] + 4.0 * diffusion)
                / (1.0 + coeff["R"]),
            ]
        )
        va2 = coeff["va2_tcr"]
        cs2 = coeff["cs2_tcr"]
        gravity = 0.0
    elif regime == "fsr":
        gamma = np.full(3, 2.0 * hubble + coeff["alpha_drag"])
        va2 = coeff["va2_fsr"]
        cs2 = coeff["cs2_baryon"]
        gravity = 4.0 * math.pi * GRAVITY * coeff["rho_b"] if include_gravity else 0.0
    else:
        raise ValueError("regime must be tcr or fsr")

    omega = kphysical**2 * va2
    stiffness = np.zeros((3, 3), dtype=float)
    stiffness[0, 0] = omega * cosine**2
    stiffness[1, 1] = omega * cosine**2
    stiffness[1, 2] = -omega * sine * cosine
    stiffness[2, 1] = -omega * sine * cosine
    stiffness[2, 2] = kphysical**2 * (cs2 + va2 * sine**2) - gravity
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, 3:] = np.eye(3) / hubble
    matrix[3:, :3] = -stiffness / hubble
    matrix[3:, 3:] = -np.diag(gamma) / hubble
    return matrix


def _sv_fields(mode: ReferenceMode, state: np.ndarray, z: float) -> Dict[str, complex]:
    cosine = math.cos(mode.theta)
    sine = math.sin(mode.theta)
    a = 1.0 / (1.0 + z)
    bx = cosine * state[0]
    by = cosine * state[1] - sine * state[2]
    initial = mode.perturbation_fraction
    selected = by if mode.polarization == "magnetosonic" else bx
    velocity = -1j * a * state[3:] / mode.k_comoving_cgs
    return {
        "b_x_over_B0": complex(bx),
        "b_y_over_B0": complex(by),
        "b_mode_over_initial": complex(selected / initial),
        "delta_b": complex(-state[2]),
        "v_b_x": complex(velocity[0]),
        "v_b_y": complex(velocity[1]),
        "v_b_z": complex(velocity[2]),
        "theta_b": complex(a * state[5]),
    }


def evolve_sv_tcr_fsr(
    mode: ReferenceMode,
    cosmology: Cosmology,
    z_start: float,
    z_end: float,
    output_redshifts: Sequence[float],
    ionization: Optional[Callable[[float], float]] = None,
    max_dln_a: float = 1.0e-3,
    include_gravity: bool = True,
    include_helium_electrons: bool = False,
) -> EvolutionResult:
    """Evolve the paper's abrupt-transition displacement model."""

    ionization = FullyIonized() if ionization is None else ionization
    targets = _requested_redshifts(z_start, z_end, output_redshifts)
    cosine = math.cos(mode.theta)
    state = np.zeros(6, dtype=float)
    component = 1 if mode.polarization == "magnetosonic" else 0
    state[component] = mode.perturbation_fraction / cosine
    zfs = _find_free_streaming_redshift(
        mode,
        cosmology,
        ionization,
        z_start,
        z_end,
        include_helium_electrons,
    )
    regime = "fsr" if zfs is not None and zfs >= z_start * (1.0 - 1.0e-12) else "tcr"
    current = z_start
    records = []
    transitioned = regime == "fsr"

    def matrix_for(which: str) -> Callable[[float], np.ndarray]:
        return lambda z: _sv_matrix(
            z,
            mode,
            cosmology,
            ionization,
            which,
            include_gravity,
            include_helium_electrons,
        )

    for target in targets:
        if (
            not transitioned
            and zfs is not None
            and current >= zfs > target
        ):
            state = _advance_matrix(state, current, zfs, matrix_for("tcr"), max_dln_a)
            # Preserve both magnetic perturbations, erase fluid density and
            # velocities as in the paper's instantaneous FSR prescription.
            bx = cosine * state[0]
            by = cosine * state[1] - math.sin(mode.theta) * state[2]
            state[:] = 0.0
            state[0] = bx / cosine
            state[1] = by / cosine
            current = zfs
            regime = "fsr"
            transitioned = True
        if target < current:
            state = _advance_matrix(state, current, target, matrix_for(regime), max_dln_a)
            current = target
        records.append(_sv_fields(mode, state, target))

    names = tuple(records[0].keys())
    fields = {name: np.asarray([row[name] for row in records]) for name in names}
    return EvolutionResult(
        model="sv_tcr_fsr",
        redshift=targets,
        fields=fields,
        metadata={
            "k_mpc": mode.k_mpc,
            "theta": mode.theta,
            "b0_gauss": mode.b0_gauss,
            "perturbation_fraction": mode.perturbation_fraction,
            "polarization": mode.polarization,
            "z_free_streaming": zfs,
            "max_dln_a": max_dln_a,
            "include_gravity": include_gravity,
            "include_helium_electrons": include_helium_electrons,
        },
    )


def _sv_closed_fsr_matrix(
    z: float,
    mode: ReferenceMode,
    cosmology: Cosmology,
    ionization: Callable[[float], float],
    closure: str,
    recombination_rates: Optional[Callable[[float, float], Sequence[float]]],
    include_gravity: bool,
    include_helium_electrons: bool,
    baryon_temperature: Optional[Callable[[float], float]] = None,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """Paper FSR matrix with either algebraic Saha or evolved 3LA ``delta_xe``."""

    base = _sv_matrix(
        z,
        mode,
        cosmology,
        ionization,
        "fsr",
        include_gravity,
        include_helium_electrons,
        baryon_temperature,
        background_electron_fraction,
    )
    if closure not in ("fixed", "saha", "three_level"):
        raise ValueError("closure must be fixed, saha, or three_level")
    coeff = background_coefficients(
        cosmology,
        z,
        ionization,
        mode.b0_gauss,
        include_helium_electrons,
        baryon_temperature,
        background_electron_fraction,
    )
    pressure_xe = (
        (mode.k_comoving_cgs / coeff["a"]) ** 2
        * cosmology.hydrogen_mass_fraction
        * KBOLTZ
        * coeff["temperature"]
        / MHYDROGEN
    )
    if closure == "fixed":
        return base
    if closure == "saha":
        xe = float(np.clip(coeff["xe"], 0.0, 1.0))
        delta_xe_per_delta_b = -xe * (1.0 - xe) / max(2.0 - xe, 1.0)
        # delta_b=-X_z, hence delta_xe=-response*X_z.
        base[5, 2] += -pressure_xe * delta_xe_per_delta_b / coeff["hubble"]
        return base
    if recombination_rates is None:
        raise ValueError("three_level closure requires recombination_rates")

    rates = np.asarray(recombination_rates(z, mode.k_mpc), dtype=float)
    if rates.shape not in ((3,), (4,)) or not np.all(np.isfinite(rates)):
        raise ValueError("recombination_rates must return three or four finite rates")
    include_temperature = baryon_temperature is not None
    state_size = 8 if include_temperature else 7
    matrix = np.zeros((state_size, state_size), dtype=float)
    matrix[:6, :6] = base
    hubble = coeff["hubble"]
    matrix[5, 6] = pressure_xe / hubble
    # d(delta_xe)/dt = r_delta*delta_b + r_xe*delta_xe
    #                    + r_theta*Theta/(aH), with Theta=a*dX_z/dt.
    matrix[6, 2] = -rates[0] / hubble
    matrix[6, 5] = rates[2] / (hubble * hubble)
    matrix[6, 6] = rates[1] / hubble
    if include_temperature:
        temperature = coeff["temperature"]
        temperature_index = 7
        if rates.size == 4:
            matrix[6, temperature_index] = rates[3] / (
                hubble * temperature
            )
        kphysical = mode.k_comoving_cgs / coeff["a"]
        matrix[5, temperature_index] = (
            kphysical**2
            * coeff["cs2_baryon"]
            / (hubble * temperature)
        )
        gamma_c = compton_temperature_rate(
            cosmology, z, coeff["electrons_per_h"]
        )
        tgamma = cosmology.tcmb0 * (1.0 + z)
        denominator = (
            1.0
            + cosmology.helium_per_hydrogen
            + coeff["electrons_per_h"]
        )
        dgamma_dxe = (
            gamma_c
            * (1.0 + cosmology.helium_per_hydrogen)
            / max(
                coeff["electrons_per_h"] * denominator,
                np.finfo(float).tiny,
            )
        )
        # SV25 Eq. (76): state[5]=dX_z/dt=Theta/a.
        matrix[temperature_index, 5] = -(2.0 / 3.0) * temperature / hubble
        matrix[temperature_index, 6] = (
            dgamma_dxe * (tgamma - temperature) / hubble
        )
        matrix[temperature_index, temperature_index] = -(
            2.0 * hubble + gamma_c
        ) / hubble
    return matrix


def _sv_closed_fields(
    mode: ReferenceMode,
    state: np.ndarray,
    z: float,
    cosmology: Cosmology,
    ionization: Callable[[float], float],
    closure: str,
    baryon_temperature: Optional[Callable[[float], float]] = None,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> Dict[str, complex]:
    fields = _sv_fields(mode, state[:6], z)
    if closure == "three_level" and state.size >= 7:
        delta_xe = complex(state[6])
    elif closure == "saha":
        xe = float(np.clip(ionization(z), 0.0, 1.0))
        delta_xe = (
            -xe * (1.0 - xe) / max(2.0 - xe, 1.0) * fields["delta_b"]
        )
    else:
        delta_xe = 0.0j
    fields["delta_xe"] = delta_xe
    if state.size == 8:
        if baryon_temperature is None:
            raise ValueError("temperature state requires a background history")
        temperature = float(baryon_temperature(z))
        fields["delta_temperature"] = complex(state[7])
        fields["delta_temperature_fraction"] = complex(
            state[7] / temperature
        )
        electrons_per_h = (
            float(background_electron_fraction(z))
            if background_electron_fraction is not None
            else float(ionization(z))
        )
        particles_per_h = (
            1.0
            + cosmology.helium_per_hydrogen
            + electrons_per_h
        )
        fields["pressure_fraction"] = (
            fields["delta_b"]
            + fields["delta_temperature_fraction"]
            + delta_xe / particles_per_h
        )
    return fields


def evolve_sv_paper_mode(
    mode: ReferenceMode,
    cosmology: Cosmology,
    z_end: float,
    output_redshifts: Sequence[float],
    ionization: Optional[Callable[[float], float]] = None,
    *,
    initialization: str = "physical",
    start_redshift: Optional[float] = None,
    closure: str = "fixed",
    recombination_rates: Optional[Callable[[float, float], Sequence[float]]] = None,
    max_dln_a: float = 1.0e-3,
    include_gravity: bool = True,
    include_helium_electrons: bool = False,
    initial_state: Optional[Sequence[float]] = None,
    baryon_temperature: Optional[Callable[[float], float]] = None,
    temperature_start_redshift: float = 1900.0,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> EvolutionResult:
    """Evolve one mode with the initialization choices used by SV25.

    ``physical`` starts at horizon entry or 1 MeV and executes the paper's
    abrupt TCR/FSR reset. ``free_streaming`` starts with a pristine magnetic
    mode at its own FSR crossing. ``redshift`` does the same at an explicitly
    supplied redshift, reproducing the paper's delayed ``z=4500`` comparison.
    """

    ionization = FullyIonized() if ionization is None else ionization
    if baryon_temperature is not None and closure != "three_level":
        raise ValueError("temperature perturbations require three_level closure")
    if baryon_temperature is not None and temperature_start_redshift <= z_end:
        raise ValueError("temperature start must lie above z_end")
    if initialization not in ("physical", "free_streaming", "redshift"):
        raise ValueError(
            "initialization must be physical, free_streaming, or redshift"
        )
    physical_start = sv_paper_initial_redshift(cosmology, mode.k_mpc)
    zfs = _find_free_streaming_redshift(
        mode,
        cosmology,
        ionization,
        physical_start,
        z_end,
        include_helium_electrons,
        background_electron_fraction,
    )
    if initialization == "physical":
        z_start = physical_start
        regime = "tcr"
    elif initialization == "free_streaming":
        if zfs is None:
            raise ValueError("mode does not enter free streaming before z_end")
        z_start = zfs
        regime = "fsr"
    else:
        if start_redshift is None:
            raise ValueError("redshift initialization requires start_redshift")
        z_start = float(start_redshift)
        regime = "fsr"
    if not z_start > z_end:
        raise ValueError("paper mode start must be above z_end")

    cosine = math.cos(mode.theta)
    if initial_state is None:
        state = np.zeros(6, dtype=float)
        component = 1 if mode.polarization == "magnetosonic" else 0
        state[component] = mode.perturbation_fraction / cosine
    else:
        state = np.asarray(initial_state, dtype=float).copy()
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain six finite displacement variables")
    transitioned = regime == "fsr"
    if transitioned and closure == "three_level":
        state = np.pad(state, (0, 1))
    temperature_active = False
    if (
        transitioned
        and baryon_temperature is not None
        and z_start <= temperature_start_redshift
    ):
        state = np.pad(state, (0, 1))
        temperature_active = True
    current = z_start
    targets = _requested_redshifts(z_start, z_end, output_redshifts)
    records: list[Dict[str, complex]] = []

    def tcr_matrix(z: float) -> np.ndarray:
        return _sv_matrix(
            z,
            mode,
            cosmology,
            ionization,
            "tcr",
            include_gravity,
            include_helium_electrons,
            background_electron_fraction=background_electron_fraction,
        )

    def fsr_matrix(z: float) -> np.ndarray:
        return _sv_closed_fsr_matrix(
            z,
            mode,
            cosmology,
            ionization,
            closure,
            recombination_rates,
            include_gravity,
            include_helium_electrons,
            baryon_temperature if temperature_active else None,
            background_electron_fraction,
        )

    for target in targets:
        if (
            not transitioned
            and zfs is not None
            and current >= zfs > target
        ):
            state = _advance_matrix(state, current, zfs, tcr_matrix, max_dln_a)
            bx = cosine * state[0]
            by = cosine * state[1] - math.sin(mode.theta) * state[2]
            state = np.zeros(7 if closure == "three_level" else 6, dtype=float)
            state[0] = bx / cosine
            state[1] = by / cosine
            current = zfs
            transitioned = True
            regime = "fsr"
            if (
                baryon_temperature is not None
                and zfs <= temperature_start_redshift
            ):
                state = np.pad(state, (0, 1))
                temperature_active = True
        if (
            transitioned
            and baryon_temperature is not None
            and not temperature_active
            and current > temperature_start_redshift >= target
        ):
            state = _advance_matrix(
                state,
                current,
                temperature_start_redshift,
                fsr_matrix,
                max_dln_a,
            )
            state = np.pad(state, (0, 1))
            current = temperature_start_redshift
            temperature_active = True
        if target < current:
            matrix = fsr_matrix if transitioned else tcr_matrix
            state = _advance_matrix(state, current, target, matrix, max_dln_a)
            current = target
        records.append(
            _sv_closed_fields(
                mode, state, target, cosmology, ionization,
                closure if transitioned else "fixed",
                baryon_temperature if temperature_active else None,
                background_electron_fraction,
            )
        )

    names = set().union(*(record.keys() for record in records))
    fields = {
        name: np.asarray([record.get(name, 0.0j) for record in records])
        for name in names
    }
    return EvolutionResult(
        model="sv_paper_" + closure,
        redshift=targets,
        fields=fields,
        metadata={
            "k_mpc": mode.k_mpc,
            "theta": mode.theta,
            "b0_gauss": mode.b0_gauss,
            "perturbation_fraction": mode.perturbation_fraction,
            "polarization": mode.polarization,
            "physical_start_redshift": physical_start,
            "z_free_streaming": zfs,
            "initialization": initialization,
            "custom_initial_state": initial_state is not None,
            "start_redshift": z_start,
            "closure": closure,
            "max_dln_a": max_dln_a,
            "include_gravity": include_gravity,
            "include_helium_electrons": include_helium_electrons,
            "custom_background_electron_fraction": (
                background_electron_fraction is not None
            ),
            "include_baryon_temperature": baryon_temperature is not None,
            "temperature_start_redshift": temperature_start_redshift,
        },
    )


def _p1_matrix(
    z: float,
    mode: ReferenceMode,
    cosmology: Cosmology,
    ionization: Callable[[float], float],
    include_gravity: bool,
    include_dark_matter: bool,
    include_photon_stress: bool,
    include_helium_electrons: bool,
    recombination_rates: Optional[Callable[[float], Sequence[float]]],
    baryon_temperature: Optional[Callable[[float], float]] = None,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    coeff = background_coefficients(
        cosmology, z, ionization, mode.b0_gauss, include_helium_electrons,
        baryon_temperature, background_electron_fraction,
    )
    dm_index = 10 if include_dark_matter else None
    xe_index = (
        10 + (2 if include_dark_matter else 0)
        if recombination_rates is not None else None
    )
    temperature_index = (
        10 + (2 if include_dark_matter else 0) +
        (1 if recombination_rates is not None else 0)
        if baryon_temperature is not None else None
    )
    size = 10 + (2 if include_dark_matter else 0) + (
        1 if recombination_rates is not None else 0
    ) + (1 if baryon_temperature is not None else 0)
    matrix = np.zeros((size, size), dtype=np.complex128)
    a = coeff["a"]
    hubble = coeff["hubble"]
    kphysical = mode.k_comoving_cgs / a
    ik = 1j * kphysical
    sine = math.sin(mode.theta)
    cosine = math.cos(mode.theta)
    gamma = coeff["alpha_drag"]
    nu = coeff["nu_scatter"]
    q2 = (mode.k_comoving_cgs * coeff["lambda_comoving"]) ** 2
    if include_photon_stress:
        _, _, atrans = _angular_transport(q2)
        dtrans = q2 / 5.0 - 8.0 * q2 * q2 / 175.0 if q2 < 1.0e-3 else max(1.0 / atrans - 1.0, 0.0)
        stress_t = nu * dtrans
        stress_l = stress_t * (4.0 * atrans / 3.0 + 1.0 - atrans)
    else:
        stress_t = 0.0
        stress_l = 0.0

    # State: bx/B0, by/B0, delta_b, vb[x,y,z], delta_gamma, vg[x,y,z],
    # and optionally delta_dm, vdm_z.
    matrix[0, 3] = ik * cosine
    matrix[1, 4] = ik * cosine
    matrix[1, 5] = -ik * sine
    matrix[2, 5] = -ik

    for vb, vg, stress in ((3, 7, stress_t), (4, 8, stress_t), (5, 9, stress_l)):
        matrix[vb, vb] = -hubble - gamma
        matrix[vb, vg] = gamma
        matrix[vg, vb] = nu
        matrix[vg, vg] = -nu - stress

    va2 = coeff["va2_fsr"]
    matrix[3, 0] += ik * va2 * cosine
    matrix[4, 1] += ik * va2 * cosine
    matrix[5, 1] += -ik * va2 * sine
    matrix[5, 2] += -ik * coeff["cs2_baryon"]
    matrix[6, 9] = -(4.0 / 3.0) * ik
    matrix[9, 6] += -0.25 * ik * CLIGHT**2

    if include_gravity:
        inverse_k = 1.0 / kphysical
        matrix[5, 2] += 1j * 4.0 * math.pi * GRAVITY * coeff["rho_b"] * inverse_k
        if include_dark_matter:
            assert dm_index is not None
            matrix[5, dm_index] += 1j * 4.0 * math.pi * GRAVITY * coeff["rho_dm"] * inverse_k
            matrix[dm_index, dm_index + 1] = -ik
            matrix[dm_index + 1, dm_index + 1] = -hubble
            matrix[dm_index + 1, 2] = 1j * 4.0 * math.pi * GRAVITY * coeff["rho_b"] * inverse_k
            matrix[dm_index + 1, dm_index] = 1j * 4.0 * math.pi * GRAVITY * coeff["rho_dm"] * inverse_k
    elif include_dark_matter:
        assert dm_index is not None
        matrix[dm_index, dm_index + 1] = -ik
        matrix[dm_index + 1, dm_index + 1] = -hubble

    rates = None
    if recombination_rates is not None:
        assert xe_index is not None
        rates = tuple(float(rate) for rate in recombination_rates(z))
        if len(rates) not in (3, 4):
            raise ValueError("recombination response must have three or four rates")
        density_rate, ionization_rate, velocity_rate = rates[:3]
        matrix[xe_index, 2] = density_rate
        matrix[xe_index, xe_index] = ionization_rate
        # Theta/(aH) = i*k_physical*v_b,z/H.
        matrix[xe_index, 5] = velocity_rate * ik / hubble
        if len(rates) == 4:
            if temperature_index is None:
                raise ValueError("temperature recombination rate needs baryon temperature")
            matrix[xe_index, temperature_index] = rates[3] / coeff["temperature"]

    if temperature_index is not None:
        tb = coeff["temperature"]
        thermal_recombination_rates = (
            recfast_v2_local_response(cosmology, z, coeff["xe"], tb)
            if rates is None else rates
        )
        thermal = recfast_v2_temperature_response(
            cosmology, z, coeff["xe"], tb, thermal_recombination_rates,
            electrons_per_hydrogen=coeff["electrons_per_h"],
        )
        matrix[temperature_index, 2] = thermal[0]
        matrix[temperature_index, 5] = thermal[2] * ik
        matrix[temperature_index, temperature_index] = thermal[3]
        if xe_index is not None:
            matrix[temperature_index, xe_index] = thermal[1]
            matrix[5, xe_index] += (
                -ik * coeff["cs2_baryon"] / coeff["particles_per_h"]
            )
        matrix[5, temperature_index] += -ik * coeff["cs2_baryon"] / tb

    # Convert physical-time derivatives to d/d ln(a).
    return matrix / hubble


def _p1_fields(
    mode: ReferenceMode,
    state: np.ndarray,
    include_dark_matter: bool,
    include_ionization: bool,
    include_temperature: bool,
    background_temperature: Optional[float],
    background_electrons_per_hydrogen: Optional[float],
    helium_per_hydrogen: float,
) -> Dict[str, complex]:
    selected = state[1] if mode.polarization == "magnetosonic" else state[0]
    result = {
        "b_x_over_B0": state[0],
        "b_y_over_B0": state[1],
        "b_mode_over_initial": selected / mode.perturbation_fraction,
        "delta_b": state[2],
        "v_b_x": state[3],
        "v_b_y": state[4],
        "v_b_z": state[5],
        "delta_gamma": state[6],
        "v_gamma_x": state[7],
        "v_gamma_y": state[8],
        "v_gamma_z": state[9],
        "slip_x": state[7] - state[3],
        "slip_y": state[8] - state[4],
        "slip_z": state[9] - state[5],
    }
    if include_dark_matter:
        result["delta_dm"] = state[10]
        result["v_dm_z"] = state[11]
    if include_ionization:
        result["delta_xe"] = state[12 if include_dark_matter else 10]
    if include_temperature:
        index = 10 + (2 if include_dark_matter else 0) + (1 if include_ionization else 0)
        result["delta_temperature"] = state[index]
        assert background_temperature is not None
        result["delta_temperature_fraction"] = state[index] / background_temperature
        assert background_electrons_per_hydrogen is not None
        particles_per_h = (
            1.0 + helium_per_hydrogen + background_electrons_per_hydrogen
        )
        delta_xe = result.get("delta_xe", 0.0j)
        internal_energy_fraction = (
            result["delta_temperature_fraction"] + delta_xe / particles_per_h
        )
        result["specific_internal_energy_fraction"] = internal_energy_fraction
        result["pressure_fraction"] = state[2] + internal_energy_fraction
    return result


def evolve_photon_p1(
    mode: ReferenceMode,
    cosmology: Cosmology,
    z_start: float,
    z_end: float,
    output_redshifts: Sequence[float],
    ionization: Optional[Callable[[float], float]] = None,
    max_dln_a: float = 1.0e-3,
    include_gravity: bool = True,
    include_dark_matter: bool = False,
    include_photon_stress: bool = True,
    include_helium_electrons: bool = False,
    recombination_rates: Optional[Callable[[float], Sequence[float]]] = None,
    baryon_temperature: Optional[Callable[[float], float]] = None,
    initial_state: Optional[np.ndarray] = None,
    background_electron_fraction: Optional[Callable[[float], float]] = None,
) -> EvolutionResult:
    """Evolve a continuous baryon/photon P1 mean-field mode."""

    ionization = FullyIonized() if ionization is None else ionization
    targets = _requested_redshifts(z_start, z_end, output_redshifts)
    state_size = (
        10 + (2 if include_dark_matter else 0)
        + (1 if recombination_rates is not None else 0)
        + (1 if baryon_temperature is not None else 0)
    )
    if initial_state is None:
        state = np.zeros(state_size, dtype=np.complex128)
        state[1 if mode.polarization == "magnetosonic" else 0] = (
            mode.perturbation_fraction
        )
    else:
        supplied = np.asarray(initial_state, dtype=np.complex128)
        if supplied.shape != (state_size,):
            raise ValueError(
                f"photon-P1 initial state has shape {supplied.shape}; "
                f"expected {(state_size,)}"
            )
        if not np.all(np.isfinite(supplied.real)) or not np.all(
            np.isfinite(supplied.imag)
        ):
            raise ValueError("photon-P1 initial state must be finite")
        state = supplied.copy()
    current = z_start

    def matrix(z: float) -> np.ndarray:
        return _p1_matrix(
            z,
            mode,
            cosmology,
            ionization,
            include_gravity,
            include_dark_matter,
            include_photon_stress,
            include_helium_electrons,
            recombination_rates,
            baryon_temperature,
            background_electron_fraction,
        )

    records = []
    for target in targets:
        if target < current:
            state = _advance_matrix(state, current, target, matrix, max_dln_a)
            current = target
        records.append(
            _p1_fields(
                mode,
                state,
                include_dark_matter,
                recombination_rates is not None,
                baryon_temperature is not None,
                None if baryon_temperature is None else baryon_temperature(target),
                None if baryon_temperature is None else (
                    background_electron_fraction(target)
                    if background_electron_fraction is not None else helium_saha_electrons(
                        cosmology, target, ionization(target)
                    ) if include_helium_electrons else ionization(target)
                ),
                cosmology.helium_per_hydrogen,
            )
        )
    fields = {
        name: np.asarray([row[name] for row in records])
        for name in records[0].keys()
    }
    return EvolutionResult(
        model="photon_p1",
        redshift=targets,
        fields=fields,
        metadata={
            "k_mpc": mode.k_mpc,
            "theta": mode.theta,
            "b0_gauss": mode.b0_gauss,
            "perturbation_fraction": mode.perturbation_fraction,
            "polarization": mode.polarization,
            "max_dln_a": max_dln_a,
            "include_gravity": include_gravity,
            "include_dark_matter": include_dark_matter,
            "include_photon_stress": include_photon_stress,
            "include_helium_electrons": include_helium_electrons,
            "include_recombination_perturbation": recombination_rates is not None,
            "include_baryon_temperature": baryon_temperature is not None,
            "custom_initial_state": initial_state is not None,
            "custom_background_electron_fraction": (
                background_electron_fraction is not None
            ),
        },
    )


def relative_complex_error(reference: complex, candidate: complex) -> Tuple[float, float]:
    """Return relative amplitude error and wrapped phase error."""

    reference = complex(reference)
    candidate = complex(candidate)
    scale = abs(reference)
    amplitude = abs(abs(candidate) - scale) / max(scale, np.finfo(float).tiny)
    phase = math.atan2(
        math.sin(np.angle(candidate) - np.angle(reference)),
        math.cos(np.angle(candidate) - np.angle(reference)),
    )
    return float(amplitude), float(abs(phase))


def result_rows(result: EvolutionResult) -> Sequence[Mapping[str, object]]:
    """Flatten an evolution result into serializable scalar rows."""

    rows = []
    for index, redshift in enumerate(result.redshift):
        row: Dict[str, object] = {"model": result.model, "z": float(redshift)}
        for name, values in result.fields.items():
            value = complex(values[index])
            row[name + "_real"] = value.real
            row[name + "_imag"] = value.imag
            row[name + "_abs"] = abs(value)
            row[name + "_phase"] = float(np.angle(value))
        rows.append(row)
    return rows
