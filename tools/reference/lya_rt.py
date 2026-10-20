#!/usr/bin/env python3
"""Independent quasi-steady Lyman-alpha transfer hierarchy.

This implements Eqs. (40)--(48) of Schiff & Venumadhav, arXiv:2506.16517v2,
using Doppler-frequency coordinates and the Fokker-Planck redistribution
operator.  It is an offline oracle/table generator, not a per-cell runtime
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import wofz

if __package__:
    from . import lmhd
else:
    import lmhd


A_LYA = 6.265e8
A_3S_2P = 6.317e6
A_3D_2P = 6.469e7
LAMBDA_LYA = 1215.67e-8
NU_LYA = lmhd.CLIGHT / LAMBDA_LYA


def voigt_profile(x: np.ndarray, damping: float) -> np.ndarray:
    """Normalized Voigt profile in Doppler-width coordinates."""

    return np.real(wofz(np.asarray(x) + 1j * damping)) / math.sqrt(math.pi)


def true_absorption_probability(temperature: float) -> float:
    """Dominant two-photon destruction branch from 3s and 3d.

    This is Venumadhav & Hirata (2015), Eq. (29). The quoted `~1e-4` at
    z=1100 is rounded; evaluating the displayed formula is important because
    the C-basis plateau scales approximately as `1/p_ab`.
    """

    activation = 5.0 * lmhd.HPLANCK * NU_LYA / 27.0
    branching = (A_3S_2P + 5.0 * A_3D_2P) / (3.0 * A_LYA)
    probability = branching * math.exp(-activation / (lmhd.KBOLTZ * temperature))
    return float(np.clip(probability, 1.0e-12, 0.1))


@dataclass(frozen=True)
class LyaBackground:
    redshift: float
    xe: float
    tau_s: float
    p_ab: float
    p_sc: float
    doppler_width: float
    damping: float
    hubble: float
    scale_factor: float
    f_equilibrium: float
    f_blackbody: float
    alpha_b: float
    beta_b: float


def lya_background(
    cosmology: lmhd.Cosmology,
    redshift: float,
    ionization,
) -> LyaBackground:
    temperature = cosmology.tcmb0 * (1.0 + redshift)
    xe = float(np.clip(ionization(redshift), 0.0, 1.0 - 1.0e-14))
    n1s = cosmology.hydrogen_density(redshift) * (1.0 - xe)
    hubble = cosmology.hubble(redshift)
    tau_s = 3.0 * A_LYA * LAMBDA_LYA**3 * n1s / (8.0 * math.pi * hubble)
    p_ab = true_absorption_probability(temperature)
    p_sc = 1.0 - p_ab
    doppler = math.sqrt(
        2.0 * lmhd.KBOLTZ * temperature / (lmhd.MHYDROGEN * lmhd.CLIGHT**2)
    )
    gamma_2p = A_LYA / p_sc
    damping = gamma_2p / (4.0 * math.pi * NU_LYA * doppler)
    alpha_b = lmhd._case_b_recombination_rate(temperature)
    # The paper defines beta_B per one of the four thermally mixed n=2
    # sublevels; its rate equation carries the corresponding factor of four.
    beta_b = 0.25 * alpha_b * (
        2.0 * math.pi * lmhd.MELECTRON * lmhd.KBOLTZ * temperature
        / lmhd.HPLANCK**2
    ) ** 1.5 * math.exp(-lmhd.E_LEVEL2 / (lmhd.KBOLTZ * temperature))
    x1s = 1.0 - xe
    escape_rate = 3.0 * A_LYA / tau_s + lmhd.LAMBDA_2S1S
    x2 = 4.0 * (
        cosmology.hydrogen_density(redshift) * xe * xe * alpha_b
        + escape_rate * x1s
        * math.exp(-lmhd.E_LYA / (lmhd.KBOLTZ * temperature))
    ) / (escape_rate + 4.0 * beta_b)
    f_equilibrium = x2 / (4.0 * x1s)
    photon_exponent = lmhd.HPLANCK * NU_LYA / (lmhd.KBOLTZ * temperature)
    f_blackbody = 1.0 / math.expm1(photon_exponent)
    return LyaBackground(
        redshift=redshift,
        xe=xe,
        tau_s=tau_s,
        p_ab=p_ab,
        p_sc=p_sc,
        doppler_width=doppler,
        damping=damping,
        hubble=hubble,
        scale_factor=1.0 / (1.0 + redshift),
        f_equilibrium=f_equilibrium,
        f_blackbody=f_blackbody,
        alpha_b=alpha_b,
        beta_b=beta_b,
    )


@dataclass
class HomogeneousLine:
    x: np.ndarray
    profile: np.ndarray
    occupation: np.ndarray
    derivative: np.ndarray
    line_average: float
    background: LyaBackground
    residual_max: float


def _mesh(xmax: float, bins_per_width: float) -> np.ndarray:
    if xmax <= 0 or bins_per_width <= 0:
        raise ValueError("frequency range and bins per width must be positive")
    count = max(81, int(math.ceil(2.0 * xmax * bins_per_width)) + 1)
    return np.linspace(-xmax, xmax, count)


def _solve_linear_bvp_trapezoid(
    mesh: np.ndarray,
    matrices: np.ndarray,
    sources: np.ndarray,
    boundary_rows: list[tuple[list[tuple[int, int, float]], float]],
) -> tuple[np.ndarray, float]:
    """Solve `y' = A(x)y+s(x)` with second-order trapezoidal defects.

    Unknowns are node-major in the sparse system. `boundary_rows` contains one
    constraint per dependent variable as `(node, variable, coefficient)` terms.
    The hierarchy is linear, so this avoids the dense Newton systems built by a
    generic collocation solver while leaving the frequency grid explicit.
    """

    variables, variables2, nodes = matrices.shape
    if variables != variables2 or sources.shape != (variables, nodes):
        raise ValueError("linear BVP matrix/source dimensions are inconsistent")
    if mesh.size != nodes or len(boundary_rows) != variables:
        raise ValueError("linear BVP mesh or boundary dimensions are inconsistent")
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    right_hand_side = np.zeros(variables * nodes)
    identity = np.eye(variables)
    for node in range(nodes - 1):
        spacing = mesh[node + 1] - mesh[node]
        row_base = variables * node
        row_ids = np.arange(row_base, row_base + variables)
        left = -identity - 0.5 * spacing * matrices[:, :, node]
        right = identity - 0.5 * spacing * matrices[:, :, node + 1]
        for column_node, block in ((node, left), (node + 1, right)):
            block_rows, block_columns = np.nonzero(block)
            rows.append(row_ids[block_rows])
            columns.append(variables * column_node + block_columns)
            values.append(block[block_rows, block_columns])
        right_hand_side[row_ids] = 0.5 * spacing * (
            sources[:, node] + sources[:, node + 1]
        )
    boundary_base = variables * (nodes - 1)
    for offset, (terms, value) in enumerate(boundary_rows):
        row = boundary_base + offset
        for node, variable, coefficient in terms:
            node %= nodes
            rows.append(np.asarray([row]))
            columns.append(np.asarray([variables * node + variable]))
            values.append(np.asarray([coefficient]))
        right_hand_side[row] = value
    operator = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(variables * nodes, variables * nodes),
    ).tocsr()
    solution = spsolve(operator, right_hand_side)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("sparse Lyman-alpha BVP produced a non-finite solution")
    residual = operator @ solution - right_hand_side
    return solution.reshape(nodes, variables).T, float(np.max(np.abs(residual)))


def solve_homogeneous_line(
    background: LyaBackground,
    *,
    xmax: float = 1000.0,
    bins_per_width: float = 2.0,
    tolerance: float = 2.0e-5,
    max_nodes: int = 200000,
    method: str = "bvp",
) -> HomogeneousLine:
    """Solve the homogeneous line with blue Dirichlet/red Neumann BCs.

    Occupations retain their physical phase-space-density normalization.
    """

    mesh = _mesh(xmax, bins_per_width)
    tau = background.tau_s
    tau_sc = tau * background.p_sc
    if tau_sc <= 0:
        raise ValueError("neutral optical depth must be positive")
    if method not in ("bvp", "sparse"):
        raise ValueError("homogeneous solver method must be 'bvp' or 'sparse'")

    if method == "sparse":
        profile = np.maximum(voigt_profile(mesh, background.damping), 1.0e-300)
        matrices = np.zeros((4, 4, mesh.size))
        sources = np.zeros((4, mesh.size))
        factor = 2.0 / tau_sc
        matrices[0, 1] = 1.0 / profile
        matrices[1, 0] = factor * tau * background.p_ab * profile
        matrices[1, 1] = -factor / profile
        matrices[1, 3] = factor * tau * background.p_sc * profile
        matrices[2, 0] = profile
        sources[1] = -factor * tau * profile * background.f_equilibrium
        boundary_rows = [
            ([(0, 1, 1.0)], 0.0),
            ([(0, 2, 1.0)], 0.0),
            ([(-1, 0, 1.0)], background.f_blackbody),
            ([(-1, 3, 1.0), (-1, 2, -1.0)], 0.0),
        ]
        solution, residual = _solve_linear_bvp_trapezoid(
            mesh, matrices, sources, boundary_rows
        )
        return HomogeneousLine(
            x=mesh,
            profile=profile,
            occupation=solution[0],
            derivative=solution[1] / profile,
            line_average=float(solution[3, -1]),
            background=background,
            residual_max=residual,
        )

    def rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        occupation, flux, _, average = y
        source = tau * profile * (
            -background.f_equilibrium + background.p_sc * average
        )
        return np.vstack(
            (
                flux / profile,
                (2.0 / tau_sc)
                * (
                    source
                    - flux / profile
                    + tau * background.p_ab * profile * occupation
                ),
                profile * occupation,
                np.zeros_like(x),
            )
        )

    def rhs_jacobian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        jacobian = np.zeros((4, 4, x.size))
        jacobian[0, 1] = 1.0 / profile
        jacobian[1, 0] = 2.0 * tau * background.p_ab * profile / tau_sc
        jacobian[1, 1] = -2.0 / (tau_sc * profile)
        jacobian[1, 3] = 2.0 * tau * background.p_sc * profile / tau_sc
        jacobian[2, 0] = profile
        return jacobian

    # Domain ordering is red -> blue. Integral(red)=0, flux(red)=0,
    # occupation(blue)=f_blackbody, and the constant average equals Integral(blue).
    def boundary(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                red[1],
                red[2],
                blue[0] - background.f_blackbody,
                blue[3] - blue[2],
            )
        )

    def boundary_jacobian(red: np.ndarray, blue: np.ndarray):
        del red, blue
        dred = np.zeros((4, 4))
        dblue = np.zeros((4, 4))
        dred[0, 1] = 1.0
        dred[1, 2] = 1.0
        dblue[2, 0] = 1.0
        dblue[3, 2] = -1.0
        dblue[3, 3] = 1.0
        return dred, dblue

    guess = np.zeros((4, mesh.size))
    transition = 0.5 * (1.0 - np.tanh(mesh / max(1.0, xmax / 20.0)))
    guess[0] = (
        background.f_blackbody
        + transition * (background.f_equilibrium - background.f_blackbody)
    )
    guess[2] = np.cumsum(voigt_profile(mesh, background.damping) * guess[0])
    guess[2] *= (mesh[1] - mesh[0])
    guess[3] = guess[2, -1]
    solution = solve_bvp(
        rhs,
        boundary,
        mesh,
        guess,
        tol=tolerance,
        max_nodes=max_nodes,
        fun_jac=rhs_jacobian,
        bc_jac=boundary_jacobian,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError("homogeneous Lyman-alpha solve failed: " + solution.message)
    x = solution.x
    y = solution.y
    profile = voigt_profile(x, background.damping)
    derivative = y[1] / np.maximum(profile, 1.0e-300)
    residual = boundary(y[:, 0], y[:, -1])
    return HomogeneousLine(
        x=x,
        profile=profile,
        occupation=y[0],
        derivative=derivative,
        line_average=float(y[3, -1]),
        background=background,
        residual_max=float(np.max(np.abs(residual))),
    )


@dataclass
class HierarchyBasis:
    name: str
    x: np.ndarray
    multipoles: np.ndarray
    line_average: float
    k_mpc: float
    lmax: int
    residual_max: float


@dataclass
class VectorHierarchyBasis:
    polarization: str
    x: np.ndarray
    multipoles: np.ndarray
    derivative_x: np.ndarray
    derivatives: np.ndarray
    equation_derivatives: np.ndarray
    k_mpc: float
    lmax: int
    residual_max: float


def vector_second_order_source(
    plus: VectorHierarchyBasis,
    minus: VectorHierarchyBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SV25 vector source using derivatives from the hierarchy ODE.

    Re-differentiating sampled multipoles adds a second discretization error in
    the sharply varying line core. The solved hierarchy already supplies the
    physical frequency derivative at every node, including its Doppler source,
    so use that quantity directly for Eq. (59)'s D2 contribution.
    """

    if plus.polarization != "+" or minus.polarization != "-":
        raise ValueError("expected D+ and D- vector hierarchies")
    if plus.x.shape != minus.x.shape or not np.array_equal(plus.x, minus.x):
        raise ValueError("vector hierarchies must use an identical frequency grid")
    if min(plus.equation_derivatives.shape[0], minus.equation_derivatives.shape[0]) <= 2:
        raise ValueError("vector hierarchy does not contain the ell=2 derivative")
    source = -(
        plus.equation_derivatives[2] + minus.equation_derivatives[2]
    ) / 5.0
    return plus.x, source


@dataclass(frozen=True)
class RecombinationResponse:
    """Linear 3LA rate coefficients for one `(z,k)` hierarchy solution.

    `density_rate` and `ionization_rate` multiply dimensionless `delta_b` and
    absolute `delta_xe`. `velocity_divergence_rate` multiplies
    `Theta/(a H)`, and `temperature_rate` multiplies `delta(T_b)/T_b`.
    All four outputs have units s^-1.
    """

    peebles_factor: float
    density_rate: float
    ionization_rate: float
    velocity_divergence_rate: float
    temperature_rate: float
    A_effective: float
    B_effective: float
    escape_response: float


@dataclass(frozen=True)
class EquilibriumSourceCoefficients:
    """Coefficients of the first-order line source.

    `delta_f_eq = feq_density*delta_b + feq_ground*delta_x1s/x1s
    + feq_velocity*Theta/(aH)`. The barred coefficients describe
    `delta_f_eq - p_sc*delta_fbar` in the same variable order.
    """

    feq_density: float
    feq_ground: float
    feq_velocity: float
    alpha_bar: float
    beta_bar: float
    gamma_bar: float


@dataclass
class SecondOrderMonopole:
    x: np.ndarray
    occupation: np.ndarray
    line_average: float
    residual_max: float


@dataclass(frozen=True)
class ResponseTable:
    redshift: np.ndarray
    k_mpc: np.ndarray
    local_rates: np.ndarray
    delta_rates: np.ndarray

    @classmethod
    def load(cls, path: Path | str) -> "ResponseTable":
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                return cls(
                    redshift=np.asarray(data["redshift"], dtype=float),
                    k_mpc=np.asarray(data["k_mpc"], dtype=float),
                    local_rates=np.asarray(data["local_rates"], dtype=float),
                    delta_rates=np.asarray(data["delta_rates"], dtype=float),
                )
        if path.suffix not in (".tsv", ".txt"):
            raise ValueError(f"unsupported response-table format: {path.suffix}")

        records = np.atleast_1d(
            np.genfromtxt(path, delimiter="\t", names=True, dtype=float)
        )
        names = set(records.dtype.names or ())
        required = {"redshift", "k_mpc"}
        components = ("density", "ionization", "velocity_divergence", "temperature")
        selected = []
        for component in components:
            local_name = f"local_{component}_rate_s"
            delta_name = f"delta_{component}_rate_s"
            if (local_name in names) != (delta_name in names):
                raise ValueError(f"incomplete {component} response columns")
            if local_name in names:
                selected.append((local_name, delta_name))
        if not required.issubset(names) or len(selected) not in (3, 4):
            raise ValueError("invalid response-table columns")

        redshift = np.unique(np.asarray(records["redshift"], dtype=float))
        k_mpc = np.unique(np.asarray(records["k_mpc"], dtype=float))
        nz, nk = redshift.size, k_mpc.size
        if records.size != nz * nk:
            raise ValueError("response table is not a complete redshift-k grid")

        order = np.lexsort((records["k_mpc"], records["redshift"]))
        ordered_z = np.asarray(records["redshift"][order], dtype=float)
        ordered_k = np.asarray(records["k_mpc"][order], dtype=float)
        if not np.array_equal(ordered_z, np.repeat(redshift, nk)) or not np.array_equal(
            ordered_k, np.tile(k_mpc, nz)
        ):
            raise ValueError("response table contains duplicate or missing grid points")

        local_grid = np.stack(
            [np.asarray(records[name][order], dtype=float) for name, _ in selected], axis=-1
        ).reshape(nz, nk, len(selected))
        if not np.all(local_grid == local_grid[:, :1, :]):
            raise ValueError("local response rates vary across k at fixed redshift")
        delta_rates = np.stack(
            [np.asarray(records[name][order], dtype=float) for _, name in selected], axis=-1
        ).reshape(nz, nk, len(selected))
        return cls(redshift, k_mpc, local_grid[:, 0, :], delta_rates)

    def __post_init__(self) -> None:
        nz = self.redshift.size
        nk = self.k_mpc.size
        if (
            self.redshift.ndim != 1
            or self.k_mpc.ndim != 1
            or nz < 2
            or nk < 2
            or self.local_rates.shape not in ((nz, 3), (nz, 4))
            or self.delta_rates.shape != (nz, nk, self.local_rates.shape[1])
            or not np.all(np.diff(self.redshift) > 0)
            or not np.all(np.diff(self.k_mpc) > 0)
            or not np.all(np.isfinite(self.local_rates))
            or not np.all(np.isfinite(self.delta_rates))
        ):
            raise ValueError("invalid nonlocal recombination response table")

    @staticmethod
    def _signed_geometric(left: float, right: float, fraction: float) -> float:
        if left == 0.0 or right == 0.0 or math.copysign(1.0, left) != math.copysign(1.0, right):
            return left + fraction * (right - left)
        return math.copysign(
            math.exp(
                math.log(abs(left))
                + fraction * (math.log(abs(right)) - math.log(abs(left)))
            ),
            left,
        )

    @staticmethod
    def _coordinate(grid: np.ndarray, value: float) -> tuple[int, float]:
        index = int(np.searchsorted(grid, value, side="right") - 1)
        index = min(max(index, 0), grid.size - 2)
        fraction = (value - grid[index]) / (grid[index + 1] - grid[index])
        return index, float(np.clip(fraction, 0.0, 1.0))

    def rates(
        self, redshift: float, k_mpc: float, *, differential_only: bool = False
    ) -> np.ndarray:
        """Interpolate `(density, ionization, Theta/aH[, delta_T/T])` rates.

        Outside the tabulated redshift support the nonlocal model is inactive.
        Below the first k node its differential response follows the required
        k^2 local-limit scaling; above the table it is held at the converged
        short-wavelength limit, matching the Metal runtime.
        """

        if not self.redshift[0] <= redshift <= self.redshift[-1]:
            return np.zeros(self.local_rates.shape[1])
        zgrid = np.log1p(self.redshift)
        iz, fz = self._coordinate(zgrid, math.log1p(redshift))
        local = np.asarray(
            [
                self._signed_geometric(
                    self.local_rates[iz, component],
                    self.local_rates[iz + 1, component],
                    fz,
                )
                for component in range(self.local_rates.shape[1])
            ]
        )
        ksafe = max(float(k_mpc), np.finfo(float).tiny)
        low_scale = 1.0
        if ksafe <= self.k_mpc[0]:
            ik = 0
            fk = 0.0
            low_scale = (ksafe / self.k_mpc[0]) ** 2
        elif ksafe >= self.k_mpc[-1]:
            ik = self.k_mpc.size - 2
            fk = 1.0
        else:
            ik, fk = self._coordinate(np.log(self.k_mpc), math.log(ksafe))
        delta = np.empty(self.local_rates.shape[1])
        for component in range(self.local_rates.shape[1]):
            lower = self._signed_geometric(
                self.delta_rates[iz, ik, component],
                self.delta_rates[iz, ik + 1, component],
                fk,
            )
            upper = self._signed_geometric(
                self.delta_rates[iz + 1, ik, component],
                self.delta_rates[iz + 1, ik + 1, component],
                fk,
            )
            delta[component] = low_scale * self._signed_geometric(lower, upper, fz)
        return delta if differential_only else local + delta


def solve_scalar_basis(
    homogeneous: HomogeneousLine,
    k_mpc: float,
    name: str,
    *,
    lmax: int = 12,
    bins_per_width: float = 2.0,
    tolerance: float = 5.0e-5,
    max_nodes: int = 300000,
    method: str = "bvp",
) -> HierarchyBasis:
    """Solve one of the scalar A, B, or C first-order basis hierarchies."""

    if name not in ("A", "B", "C"):
        raise ValueError("basis name must be A, B, or C")
    if k_mpc < 0 or lmax < 2:
        raise ValueError("k must be non-negative and lmax must be >= 2")
    if method not in ("bvp", "sparse"):
        raise ValueError("hierarchy solver method must be 'bvp' or 'sparse'")
    background = homogeneous.background
    xmax = max(abs(float(homogeneous.x[0])), abs(float(homogeneous.x[-1])))
    mesh = _mesh(xmax, bins_per_width)
    tau = background.tau_s
    tau_sc = tau * background.p_sc
    k_cgs = k_mpc / lmhd.MPC_CM
    streaming = (
        lmhd.CLIGHT
        * k_cgs
        * background.doppler_width
        / (background.scale_factor * background.hubble)
    )

    derivative_spline = CubicSpline(
        homogeneous.x,
        homogeneous.derivative,
        bc_type="natural",
        extrapolate=False,
    )

    def homogeneous_derivative(x: np.ndarray) -> np.ndarray:
        derivative = derivative_spline(x)
        return (
            derivative / background.f_equilibrium
            if name in ("A", "B")
            else derivative
        )

    if method == "sparse":
        profile = np.maximum(voigt_profile(mesh, background.damping), 1.0e-300)
        source_shape = homogeneous_derivative(mesh)
        variables = lmax + 3
        flux_index = lmax + 1
        integral_index = lmax + 2
        matrices = np.zeros((variables, variables, mesh.size))
        sources = np.zeros((variables, mesh.size))
        matrices[0, flux_index] = 1.0 / profile
        factor = 2.0 / tau_sc
        matrices[flux_index, 0] = factor * tau * background.p_ab * profile
        matrices[flux_index, 1] = factor * streaming / 3.0
        matrices[flux_index, flux_index] = -factor / profile
        if name == "A":
            sources[flux_index] = factor * source_shape
        elif name == "B":
            sources[flux_index] = -factor * source_shape / 3.0
        else:
            sources[flux_index] = -factor * tau * profile
        for ell in range(1, lmax + 1):
            matrices[ell, ell] = tau * profile
            matrices[ell, ell - 1] = streaming * (
                -ell / (2.0 * ell - 1.0)
            )
            if ell < lmax:
                matrices[ell, ell + 1] = streaming * (
                    (ell + 1.0) / (2.0 * ell + 3.0)
                )
        if name == "B":
            sources[2] = 2.0 * source_shape / 3.0
        matrices[integral_index, 0] = profile
        boundary_rows = [
            ([(0, flux_index, 1.0)], 0.0),
            ([(0, integral_index, 1.0)], 0.0),
        ]
        boundary_rows.extend(
            ([(-1, ell, 1.0)], 0.0) for ell in range(lmax + 1)
        )
        solution, residual = _solve_linear_bvp_trapezoid(
            mesh, matrices, sources, boundary_rows
        )
        physical_scale = background.f_equilibrium if name in ("A", "B") else 1.0
        return HierarchyBasis(
            name=name,
            x=mesh,
            multipoles=solution[: lmax + 1] * physical_scale,
            line_average=float(solution[integral_index, -1] * physical_scale),
            k_mpc=k_mpc,
            lmax=lmax,
            residual_max=residual,
        )

    def rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        source_shape = homogeneous_derivative(x)
        moments = y[: lmax + 1]
        flux = y[lmax + 1]
        result = np.zeros_like(y)
        result[0] = flux / profile
        source0 = np.zeros_like(x)
        if name == "A":
            source0 = source_shape
        elif name == "B":
            source0 = -source_shape / 3.0
        else:
            source0 = -tau * profile
        result[lmax + 1] = (2.0 / tau_sc) * (
            source0
            - flux / profile
            + tau * background.p_ab * profile * moments[0]
            + (streaming / 3.0) * moments[1]
        )
        for ell in range(1, lmax + 1):
            lower = -ell * moments[ell - 1] / (2.0 * ell - 1.0)
            upper = (
                (ell + 1.0) * moments[ell + 1] / (2.0 * ell + 3.0)
                if ell < lmax
                else 0.0
            )
            source = (2.0 / 3.0) * source_shape if name == "B" and ell == 2 else 0.0
            result[ell] = streaming * (lower + upper) + tau * profile * moments[ell] + source
        result[lmax + 2] = profile * moments[0]
        return result

    def rhs_jacobian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        variables = lmax + 3
        flux_index = lmax + 1
        integral_index = lmax + 2
        jacobian = np.zeros((variables, variables, x.size))
        jacobian[0, flux_index] = 1.0 / profile
        factor = 2.0 / tau_sc
        jacobian[flux_index, 0] = factor * tau * background.p_ab * profile
        jacobian[flux_index, 1] = factor * streaming / 3.0
        jacobian[flux_index, flux_index] = -factor / profile
        for ell in range(1, lmax + 1):
            jacobian[ell, ell] = tau * profile
            jacobian[ell, ell - 1] = streaming * (
                -ell / (2.0 * ell - 1.0)
            )
            if ell < lmax:
                jacobian[ell, ell + 1] = streaming * (
                    (ell + 1.0) / (2.0 * ell + 3.0)
                )
        jacobian[integral_index, 0] = profile
        return jacobian

    # All perturbation multipoles vanish on the blue boundary. The monopole's
    # second-order equation receives its other BC from zero red-wing flux.
    def boundary(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        values = [red[lmax + 1], red[lmax + 2]]
        values.extend(blue[: lmax + 1])
        return np.asarray(values)

    def boundary_jacobian(red: np.ndarray, blue: np.ndarray):
        del red, blue
        variables = lmax + 3
        flux_index = lmax + 1
        integral_index = lmax + 2
        dred = np.zeros((variables, variables))
        dblue = np.zeros((variables, variables))
        dred[0, flux_index] = 1.0
        dred[1, integral_index] = 1.0
        for ell in range(lmax + 1):
            dblue[ell + 2, ell] = 1.0
        return dred, dblue

    guess = np.zeros((lmax + 3, mesh.size))
    solution = solve_bvp(
        rhs,
        boundary,
        mesh,
        guess,
        tol=tolerance,
        max_nodes=max_nodes,
        fun_jac=rhs_jacobian,
        bc_jac=boundary_jacobian,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError(f"Lyman-alpha {name} hierarchy failed: {solution.message}")
    residual = boundary(solution.y[:, 0], solution.y[:, -1])
    physical_scale = background.f_equilibrium if name in ("A", "B") else 1.0
    return HierarchyBasis(
        name=name,
        x=solution.x,
        multipoles=solution.y[: lmax + 1] * physical_scale,
        line_average=float(solution.y[lmax + 2, -1] * physical_scale),
        k_mpc=k_mpc,
        lmax=lmax,
        residual_max=float(np.max(np.abs(residual))),
    )


def solve_vector_basis(
    homogeneous: HomogeneousLine,
    k_mpc: float,
    polarization: str,
    *,
    lmax: int = 12,
    bins_per_width: float = 2.0,
    tolerance: float = 5.0e-5,
    max_nodes: int = 300000,
    method: str = "bvp",
) -> VectorHierarchyBasis:
    """Solve the first-order `m=+1` or `m=-1` D hierarchy.

    This implements Schiff--Venumadhav Eq. (55) after factoring out the
    transverse velocity amplitudes as in their Eq. (56). Row zero of the
    returned multipole array is identically zero because vector harmonics
    begin at ell=1.
    """

    if polarization not in ("+", "-"):
        raise ValueError("vector polarization must be '+' or '-'")
    if k_mpc < 0 or lmax < 2:
        raise ValueError("k must be non-negative and lmax must be >= 2")
    if method not in ("bvp", "sparse", "marching"):
        raise ValueError(
            "hierarchy solver method must be 'bvp', 'sparse', or 'marching'"
        )
    background = homogeneous.background
    xmax = max(abs(float(homogeneous.x[0])), abs(float(homogeneous.x[-1])))
    mesh = _mesh(xmax, bins_per_width)
    tau = background.tau_s
    k_cgs = k_mpc / lmhd.MPC_CM
    streaming = (
        lmhd.CLIGHT
        * k_cgs
        * background.doppler_width
        / (background.scale_factor * background.hubble)
    )
    derivative_spline = CubicSpline(
        homogeneous.x,
        homogeneous.derivative,
        bc_type="natural",
        extrapolate=False,
    )

    # Scaling by f_eq keeps the linear solve well conditioned while preserving
    # the physical phase-space-density normalization in the returned result.
    def scaled_derivative(x: np.ndarray) -> np.ndarray:
        return derivative_spline(x) / background.f_equilibrium

    def lower_coefficient(ell: int) -> float:
        numerator = ell - 1 if polarization == "+" else ell + 1
        return -streaming * numerator / (2.0 * ell - 1.0)

    def upper_coefficient(ell: int) -> float:
        numerator = ell + 2 if polarization == "+" else ell
        return streaming * numerator / (2.0 * ell + 3.0)

    def physical_derivatives(
        x: np.ndarray, multipoles: np.ndarray
    ) -> np.ndarray:
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        result = np.zeros_like(multipoles)
        for ell in range(1, lmax + 1):
            result[ell] = tau * profile * multipoles[ell]
            if ell > 1:
                result[ell] += lower_coefficient(ell) * multipoles[ell - 1]
            if ell < lmax:
                result[ell] += upper_coefficient(ell) * multipoles[ell + 1]
        result[2] -= derivative_spline(x) / 6.0
        return result

    def interval_derivatives(
        x: np.ndarray, multipoles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        spacing = np.diff(x)
        midpoint = 0.5 * (x[:-1] + x[1:])
        return midpoint, np.diff(multipoles, axis=1) / spacing[np.newaxis, :]

    if method == "marching":
        profile = np.maximum(voigt_profile(mesh, background.damping), 1.0e-300)
        variables = lmax
        lower = np.asarray(
            [lower_coefficient(ell) for ell in range(2, lmax + 1)]
        )
        upper = np.asarray(
            [upper_coefficient(ell) for ell in range(1, lmax)]
        )
        source = np.zeros((variables, mesh.size))
        source[1] = -scaled_derivative(mesh) / 6.0
        solution = np.zeros_like(source)
        cprime = np.empty(max(1, variables - 1))
        dprime = np.empty(variables)

        def apply_matrix(node: int, vector: np.ndarray) -> np.ndarray:
            product = tau * profile[node] * vector
            product = product.copy()
            product[1:] += lower * vector[:-1]
            product[:-1] += upper * vector[1:]
            return product

        def solve_left(
            spacing: float, node: int, right_hand_side: np.ndarray
        ) -> np.ndarray:
            diagonal = -1.0 - 0.5 * spacing * tau * profile[node]
            lower_left = -0.5 * spacing * lower
            upper_left = -0.5 * spacing * upper
            if variables == 1:
                return right_hand_side / diagonal
            cprime[0] = upper_left[0] / diagonal
            dprime[0] = right_hand_side[0] / diagonal
            for row in range(1, variables):
                denominator = diagonal - lower_left[row - 1] * cprime[row - 1]
                if row < variables - 1:
                    cprime[row] = upper_left[row] / denominator
                dprime[row] = (
                    right_hand_side[row] - lower_left[row - 1] * dprime[row - 1]
                ) / denominator
            result = np.empty(variables)
            result[-1] = dprime[-1]
            for row in range(variables - 2, -1, -1):
                result[row] = dprime[row] - cprime[row] * result[row + 1]
            return result

        for node in range(mesh.size - 2, -1, -1):
            spacing = mesh[node + 1] - mesh[node]
            right_block = (
                solution[:, node + 1]
                - 0.5 * spacing * apply_matrix(node + 1, solution[:, node + 1])
            )
            right_hand_side = (
                0.5 * spacing * (source[:, node] + source[:, node + 1])
                - right_block
            )
            solution[:, node] = solve_left(spacing, node, right_hand_side)

        residual = 0.0
        for node in range(mesh.size - 1):
            spacing = mesh[node + 1] - mesh[node]
            defect = (
                solution[:, node + 1]
                - solution[:, node]
                - 0.5
                * spacing
                * (
                    apply_matrix(node, solution[:, node])
                    + source[:, node]
                    + apply_matrix(node + 1, solution[:, node + 1])
                    + source[:, node + 1]
                )
            )
            residual = max(residual, float(np.max(np.abs(defect))))
        multipoles = np.zeros((lmax + 1, mesh.size))
        multipoles[1:] = solution * background.f_equilibrium
        derivative_x, derivatives = interval_derivatives(mesh, multipoles)
        return VectorHierarchyBasis(
            polarization=polarization,
            x=mesh,
            multipoles=multipoles,
            derivative_x=derivative_x,
            derivatives=derivatives,
            equation_derivatives=physical_derivatives(mesh, multipoles),
            k_mpc=k_mpc,
            lmax=lmax,
            residual_max=residual,
        )

    if method == "sparse":
        profile = np.maximum(voigt_profile(mesh, background.damping), 1.0e-300)
        variables = lmax
        matrices = np.zeros((variables, variables, mesh.size))
        sources = np.zeros((variables, mesh.size))
        for ell in range(1, lmax + 1):
            index = ell - 1
            matrices[index, index] = tau * profile
            if ell > 1:
                matrices[index, index - 1] = lower_coefficient(ell)
            if ell < lmax:
                matrices[index, index + 1] = upper_coefficient(ell)
        sources[1] = -scaled_derivative(mesh) / 6.0
        boundary_rows = [
            ([(-1, index, 1.0)], 0.0) for index in range(variables)
        ]
        solution, residual = _solve_linear_bvp_trapezoid(
            mesh, matrices, sources, boundary_rows
        )
        multipoles = np.zeros((lmax + 1, mesh.size))
        multipoles[1:] = solution * background.f_equilibrium
        derivative_x, derivatives = interval_derivatives(mesh, multipoles)
        return VectorHierarchyBasis(
            polarization=polarization,
            x=mesh,
            multipoles=multipoles,
            derivative_x=derivative_x,
            derivatives=derivatives,
            equation_derivatives=physical_derivatives(mesh, multipoles),
            k_mpc=k_mpc,
            lmax=lmax,
            residual_max=residual,
        )

    def rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        result = tau * profile[np.newaxis, :] * y
        for ell in range(1, lmax + 1):
            index = ell - 1
            if ell > 1:
                result[index] += lower_coefficient(ell) * y[index - 1]
            if ell < lmax:
                result[index] += upper_coefficient(ell) * y[index + 1]
        result[1] -= scaled_derivative(x) / 6.0
        return result

    def rhs_jacobian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        jacobian = np.zeros((lmax, lmax, x.size))
        for ell in range(1, lmax + 1):
            index = ell - 1
            jacobian[index, index] = tau * profile
            if ell > 1:
                jacobian[index, index - 1] = lower_coefficient(ell)
            if ell < lmax:
                jacobian[index, index + 1] = upper_coefficient(ell)
        return jacobian

    def boundary(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        del red
        return blue

    def boundary_jacobian(red: np.ndarray, blue: np.ndarray):
        del red, blue
        return np.zeros((lmax, lmax)), np.eye(lmax)

    solution = solve_bvp(
        rhs,
        boundary,
        mesh,
        np.zeros((lmax, mesh.size)),
        tol=tolerance,
        max_nodes=max_nodes,
        fun_jac=rhs_jacobian,
        bc_jac=boundary_jacobian,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError(
            f"Lyman-alpha D{polarization} hierarchy failed: {solution.message}"
        )
    multipoles = np.zeros((lmax + 1, solution.x.size))
    multipoles[1:] = solution.y * background.f_equilibrium
    residual = boundary(solution.y[:, 0], solution.y[:, -1])
    derivative_x, derivatives = interval_derivatives(solution.x, multipoles)
    return VectorHierarchyBasis(
        polarization=polarization,
        x=solution.x,
        multipoles=multipoles,
        derivative_x=derivative_x,
        derivatives=derivatives,
        equation_derivatives=physical_derivatives(solution.x, multipoles),
        k_mpc=k_mpc,
        lmax=lmax,
        residual_max=float(np.max(np.abs(residual))),
    )


def solve_second_order_source(
    homogeneous: HomogeneousLine,
    source_x: np.ndarray,
    *,
    source_grid: np.ndarray | None = None,
    bins_per_width: float = 2.0,
    tolerance: float = 5.0e-5,
    max_nodes: int = 300000,
    method: str = "bvp",
) -> SecondOrderMonopole:
    """Solve `D_x s = source_x` for the homogeneous second-order monopole.

    `source_x` is sampled on `source_grid` (or `homogeneous.x` when omitted)
    and already includes the `nu_Lya*Delta_H` conversion from the paper's
    frequency derivative. The red-wing flux is zero and the perturbation
    vanishes at the blue boundary.
    """

    if method not in ("bvp", "sparse"):
        raise ValueError("second-order solver method must be 'bvp' or 'sparse'")
    source_x = np.asarray(source_x, dtype=float)
    source_grid = (
        homogeneous.x if source_grid is None else np.asarray(source_grid, dtype=float)
    )
    if (
        source_x.shape != source_grid.shape
        or source_grid.ndim != 1
        or source_grid.size < 3
        or not np.all(np.isfinite(source_x))
        or not np.all(np.isfinite(source_grid))
        or not np.all(np.diff(source_grid) > 0)
    ):
        raise ValueError("second-order source/grid must be finite and monotonic")
    source_scale = float(np.max(np.abs(source_x)))
    if source_scale == 0.0:
        mesh = _mesh(
            max(abs(float(homogeneous.x[0])), abs(float(homogeneous.x[-1]))),
            bins_per_width,
        )
        return SecondOrderMonopole(
            x=mesh,
            occupation=np.zeros_like(mesh),
            line_average=0.0,
            residual_max=0.0,
        )
    normalized_source = source_x / source_scale
    background = homogeneous.background
    xmax = max(abs(float(homogeneous.x[0])), abs(float(homogeneous.x[-1])))
    mesh = _mesh(xmax, bins_per_width)
    source_spline = CubicSpline(
        source_grid, normalized_source, bc_type="natural", extrapolate=True
    )
    tau = background.tau_s
    tau_sc = tau * background.p_sc

    if method == "sparse":
        profile = np.maximum(voigt_profile(mesh, background.damping), 1.0e-300)
        matrices = np.zeros((3, 3, mesh.size))
        sources = np.zeros((3, mesh.size))
        factor = 2.0 / tau_sc
        matrices[0, 1] = 1.0 / profile
        matrices[1, 0] = factor * tau * background.p_ab * profile
        matrices[1, 1] = -factor / profile
        matrices[2, 0] = profile
        sources[1] = factor * source_spline(mesh)
        boundary_rows = [
            ([(0, 1, 1.0)], 0.0),
            ([(0, 2, 1.0)], 0.0),
            ([(-1, 0, 1.0)], 0.0),
        ]
        solution, residual = _solve_linear_bvp_trapezoid(
            mesh, matrices, sources, boundary_rows
        )
        return SecondOrderMonopole(
            x=mesh,
            occupation=solution[0] * source_scale,
            line_average=float(solution[2, -1] * source_scale),
            residual_max=residual * source_scale,
        )

    def rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        result = np.zeros_like(y)
        result[0] = y[1] / profile
        result[1] = (2.0 / tau_sc) * (
            source_spline(x)
            - y[1] / profile
            + tau * background.p_ab * profile * y[0]
        )
        result[2] = profile * y[0]
        return result

    def rhs_jacobian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        del y
        profile = np.maximum(voigt_profile(x, background.damping), 1.0e-300)
        jacobian = np.zeros((3, 3, x.size))
        jacobian[0, 1] = 1.0 / profile
        factor = 2.0 / tau_sc
        jacobian[1, 0] = factor * tau * background.p_ab * profile
        jacobian[1, 1] = -factor / profile
        jacobian[2, 0] = profile
        return jacobian

    def boundary(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        return np.asarray((red[1], red[2], blue[0]))

    def boundary_jacobian(red: np.ndarray, blue: np.ndarray):
        del red, blue
        dred = np.zeros((3, 3))
        dblue = np.zeros((3, 3))
        dred[0, 1] = 1.0
        dred[1, 2] = 1.0
        dblue[2, 0] = 1.0
        return dred, dblue

    solution = solve_bvp(
        rhs,
        boundary,
        mesh,
        np.zeros((3, mesh.size)),
        tol=tolerance,
        max_nodes=max_nodes,
        fun_jac=rhs_jacobian,
        bc_jac=boundary_jacobian,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError("second-order Lyman-alpha solve failed: " + solution.message)
    residual = boundary(solution.y[:, 0], solution.y[:, -1])
    return SecondOrderMonopole(
        x=solution.x,
        occupation=solution.y[0] * source_scale,
        line_average=float(solution.y[2, -1] * source_scale),
        residual_max=float(np.max(np.abs(residual))) * source_scale,
    )


def first_order_response(
    cosmology: lmhd.Cosmology,
    ionization,
    redshift: float,
    k_mpc: float,
    **solver_options,
) -> Dict[str, object]:
    """Return the converged scalar line-response coefficients at `(z,k)`."""

    background = lya_background(cosmology, redshift, ionization)
    homogeneous_keys = ("xmax", "bins_per_width", "tolerance", "max_nodes", "method")
    homogeneous_options = {
        key: solver_options[key] for key in homogeneous_keys if key in solver_options
    }
    homogeneous = solve_homogeneous_line(background, **homogeneous_options)
    hierarchy_options = dict(solver_options)
    hierarchy_options.pop("xmax", None)
    bases = {
        name: solve_scalar_basis(homogeneous, k_mpc, name, **hierarchy_options)
        for name in ("A", "B", "C")
    }
    cbar = bases["C"].line_average
    denominator = 1.0 + background.p_sc * cbar
    escape = (1.0 - background.p_ab * cbar) / denominator
    return {
        "background": background,
        "homogeneous": homogeneous,
        "bases": bases,
        "Abar": bases["A"].line_average,
        "Bbar": bases["B"].line_average,
        "Cbar": cbar,
        "denominator": denominator,
        "escape_response": escape,
    }


def equilibrium_source_coefficients(
    first_order: Dict[str, object], cosmology: lmhd.Cosmology
) -> EquilibriumSourceCoefficients:
    """Return Eq. (79) of Venumadhav--Hirata in the SV25 basis.

    The barred form is the source multiplying the C hierarchy and is the
    quantity denoted by barred alpha, beta, and gamma in SV25 Appendix C.
    """

    background = first_order["background"]
    if not isinstance(background, LyaBackground):
        raise TypeError("first_order response has no LyaBackground")
    xe = background.xe
    x1s = 1.0 - xe
    n_h = cosmology.hydrogen_density(background.redshift)
    abar = float(first_order["Abar"])
    bbar = float(first_order["Bbar"])
    cbar = float(first_order["Cbar"])
    line_denominator = 1.0 + background.p_sc * cbar
    a_effective = abar / line_denominator
    b_effective = bbar / line_denominator
    escape = (1.0 - background.p_ab * cbar) / line_denominator
    population_denominator = (
        3.0 * A_LYA * escape + lmhd.LAMBDA_2S1S + 4.0 * background.beta_b
    )
    recombination = n_h * background.alpha_b
    feq_density = (
        recombination * xe * xe / x1s + 3.0 * A_LYA * a_effective
    ) / population_denominator
    feq_ground = (
        -recombination * xe * xe / x1s
        - 2.0 * recombination * xe
        + 3.0 * A_LYA * a_effective
    ) / population_denominator
    feq_velocity = (
        3.0 * A_LYA * b_effective / population_denominator
    )
    return EquilibriumSourceCoefficients(
        feq_density=feq_density,
        feq_ground=feq_ground,
        feq_velocity=feq_velocity,
        alpha_bar=(feq_density - background.p_sc * abar) / line_denominator,
        beta_bar=(feq_ground - background.p_sc * abar) / line_denominator,
        gamma_bar=(feq_velocity - background.p_sc * bbar) / line_denominator,
    )


def recombination_response(first_order: Dict[str, object], cosmology: lmhd.Cosmology) -> RecombinationResponse:
    """Map line-basis averages to the free-electron rate response.

    The displayed RHS in Schiff--Venumadhav Eq. (49) is inherited from
    Venumadhav--Hirata Eq. (80), where it is explicitly
    `delta dot{x}_{1s}`. The newer paper labels the left-hand side as
    `delta dot{x}_e` without changing that RHS. Hydrogen conservation requires
    `delta dot{x}_e = -delta dot{x}_{1s}`. The sign used here is independently
    fixed by the short-wavelength `P -> 1` limit, which must reduce to their
    Eq. (B39): both density and ionization perturbations increase recombination
    and therefore have negative free-electron rate coefficients.
    """

    background = first_order["background"]
    if not isinstance(background, LyaBackground):
        raise TypeError("first_order response has no LyaBackground")
    redshift = background.redshift
    xe = background.xe
    x1s = 1.0 - xe
    n_h = cosmology.hydrogen_density(redshift)
    cbar = float(first_order["Cbar"])
    denominator = 1.0 + background.p_sc * cbar
    a_effective = float(first_order["Abar"]) / denominator
    b_effective = float(first_order["Bbar"]) / denominator
    escape = (1.0 - background.p_ab * cbar) / denominator
    lya_escape = 3.0 * A_LYA * escape
    peebles = (lya_escape + lmhd.LAMBDA_2S1S) / (
        lya_escape + lmhd.LAMBDA_2S1S + 4.0 * background.beta_b
    )
    alpha = background.alpha_b
    beta = background.beta_b
    density_rate = (
        -peebles * n_h * xe * xe * alpha
        + 3.0 * (1.0 - peebles) * x1s * A_LYA * a_effective
    )
    ionization_bracket = (
        (1.0 - peebles) * n_h * xe * xe * alpha / x1s
        - 2.0 * peebles * n_h * xe * alpha
        - 4.0 * background.f_equilibrium * beta
    )
    ionization_rate = (
        ionization_bracket
        - 3.0 * (1.0 - peebles) * A_LYA * a_effective
    )
    velocity_rate = 3.0 * (
        1.0 - peebles
    ) * x1s * A_LYA * b_effective
    temperature = cosmology.tcmb0 * (1.0 + redshift)
    alpha_slope = lmhd.case_b_recombination_log_slope(temperature)
    # SV25 Appendix A states that baryon-temperature perturbations enter the
    # first-order recombination equation through delta(alpha_B).  Eliminating
    # the quasi-steady n=2 population gives the same k-dependent Peebles factor
    # that multiplies the direct density recombination source.
    temperature_rate = -peebles * n_h * xe * xe * alpha * alpha_slope
    return RecombinationResponse(
        peebles_factor=peebles,
        density_rate=density_rate,
        ionization_rate=ionization_rate,
        velocity_divergence_rate=velocity_rate,
        temperature_rate=temperature_rate,
        A_effective=a_effective,
        B_effective=b_effective,
        escape_response=escape,
    )
