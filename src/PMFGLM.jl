"""
    PMFGLM

Single-node primordial magnetic field MHD, photon moments, primordial chemistry,
and particle gravity. CPU, CUDA, and Metal kernels share KernelAbstractions.jl
implementations. See `RunConfig`, `run_simulation`, and the bundled examples.
"""
module PMFGLM

import MHDKernels
import PoissonKernels
import ChemistryKernels
import EmissionKernels
import KernelAbstractions
using Dates
using SHA
using TOML

export MHDKernels, PoissonKernels, ChemistryKernels, EmissionKernels
export RunConfig, load_config, validate_config, run_simulation, run_environment
export backend, allocate_state, init_pmf_batchelor!, pmf_batchelor_field
export step!, evolve!, compute_dt, fields_to_host, max_divb, source_fingerprint
export magnetic_code_amplitude

const backend = MHDKernels.backend
const allocate_state = MHDKernels.allocate_state
const init_pmf_batchelor! = MHDKernels.init_pmf_batchelor!
const pmf_batchelor_field = MHDKernels.pmf_batchelor_field
const step! = MHDKernels.step!
const evolve! = MHDKernels.evolve!
const compute_dt = MHDKernels.compute_dt
const fields_to_host = MHDKernels.fields_to_host
const max_divb = MHDKernels.max_divb
const PACKAGE_ROOT = dirname(@__DIR__)

include("configuration.jl")
include("provenance.jl")

"Convert comoving nG to B/sqrt(4pi*rho)/velocity_unit at the initial redshift."
function magnetic_code_amplitude(b0_ng::Real,z::Real;
        h::Real=0.674,omega_b::Real=0.17037*0.315,velocity_unit_cms::Real=8e5)
    all(isfinite,(b0_ng,z,h,omega_b,velocity_unit_cms)) &&
        b0_ng >= 0 && z >= 0 && h > 0 && omega_b > 0 && velocity_unit_cms > 0 ||
        throw(ArgumentError("finite nonnegative B,z and positive h,omega_b,velocity units required"))
    rho = omega_b*1.8788e-29*h^2*(1+z)^3
    b0_ng*1e-9*(1+z)^2/(velocity_unit_cms*sqrt(4pi*rho))
end

end
