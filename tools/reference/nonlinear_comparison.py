"""Generate, run, and assess a real standalone mean-field MHD/P1 comparison.

This is a controlled weak-perturbation test with full local chemistry. A
supplied Ly-alpha table can add its nonlocal response. It does not certify
random-field nonlinear recombination or a full photon Boltzmann hierarchy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np

from . import lmhd, lya_rt, matched_units

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    "B_y": "b_y_over_B0", "gas_delta": "delta_b",
    "gas_v_x": "v_b_z", "gas_v_y": "v_b_y",
    "photon_v_y": "v_gamma_y",
    "photon_baryon_slip_x": "slip_z", "photon_baryon_slip_y": "slip_y",
    "dm_delta": "delta_dm", "dm_v_x": "v_dm_z", "x_HII": "delta_xe",
}
VELOCITIES = {name for name in FIELDS if "_v_" in name or "slip_" in name}


def mode_at(rows, field, z):
    selected = sorted((-math.log1p(float(r["z"])),
                       complex(float(r["real"]),float(r["imag"])))
                      for r in rows if r["field"] == field)
    if not selected:
        raise ValueError(f"missing production Fourier mode: {field}")
    x, values = np.asarray([v[0] for v in selected]), np.asarray([v[1] for v in selected])
    target = -math.log1p(z)
    if target < x[0]-1e-9 or target > x[-1]+1e-9:
        raise ValueError(f"mode {field} does not span z={z}")
    return complex(np.interp(target,x,values.real),np.interp(target,x,values.imag))


def physical_mode(value, field, z, handoff):
    if field in ("B_y","B_z"):
        return value / float(handoff["guide_field_code"])
    if field in VELOCITIES and field != "dm_v_x":
        value *= float(handoff["velocity_unit_cm_s_at_handoff"]) * math.sqrt(
            (1+z)/(1+float(handoff["zhandoff"])))
    return value


def write_config(path, settings):
    # JSON scalar/string/array representations are valid for these TOML values.
    path.write_text("\n\n".join("["+section+"]\n"+"\n".join(
        f"{key} = {json.dumps(value,allow_nan=False)}" for key,value in values.items()
    ) for section,values in settings.items())+"\n")


def case_settings(handoff, directory, args):
    radiation = 4.15e-5 / args.hubble**2
    settings = {
        "run": {"n":args.resolution,"backend":args.backend,"cfl":args.cfl,
                "reconstruction":"ppm","riemann":"hlld","max_cycles":3000000,
                "print_every":1000,"checkpoint_final":True},
        "cosmology": {"zstart":args.zstart,"zend":args.zend,
            "box_ckpc_h":args.box_ckpc_h,"h":args.hubble,
            "omega_m":matched_units.OMEGA_M,"omega_b":matched_units.OMEGA_B,
            "omega_r":radiation,"omega_lambda":1-matched_units.OMEGA_M-radiation},
        "magnetic": {"initial_condition":"meanfield_linear","b0_ng":args.field_pg*0.001},
        "chemistry": {"mode":"mixing","helium":True,"expansion":"on",
            "coupling":"strang","hydrogen_storage":"centered","smoothing":"local",
            "ionized_fast_zmin":0,"x_hii_initial":handoff["background"]["xHII"],
            "velocity_unit_cms":handoff["velocity_unit_cm_s_at_handoff"]},
        "physics": {"gravity":True,"gas_gravity":True,"compton_drag":True,
            "drag_mode":"radiation","photon_response":"moments_exponential",
            "nonlinear_cfl":0.1,"glm_project_every":max(1,round(1.6/args.cfl)),
            "pressure":handoff["background"]["pressure_code_mean"]},
        "diagnostics": {"redshifts":np.linspace(args.zstart,args.zend,args.outputs).tolist(),
            "angle_bins":1,"spectrum_bins":max(4,args.resolution//2),"energy_ledger_every":1},
        "environment": {"MHD_INITIAL_STATE_DIR":str(directory.resolve()),
            "MHD_MEANFIELD_THETA":args.theta,"MHD_MEANFIELD_PERT_FRAC":args.epsilon,
            "MHD_MEANFIELD_POLARIZATION":"magnetosonic","MHD_PMF_KMODE":1,
            "MHD_PMF_NORMALIZATION_Z":args.zstart,"MHD_CHEM_FH":matched_units.HYDROGEN_MASS_FRACTION,
            "MHD_CHEM_FALPHA":0,"MHD_CHEM_ALLOW_LOCAL_MIXING":1,
            "MHD_RADIATION_MAGNETIC_COMPENSATION":1,"MHD_RADIATION_FREE_STREAM_TOL":0,
            "MHD_DRAG_HELIUM_ELECTRONS":1,"MHD_PROFILE_SKIP_CYCLES":5},
    }
    if args.response_table:
        settings["chemistry"]["nonlocal_table"] = str(args.response_table.resolve())
    return settings


def assess(output, handoff, args):
    run = output / "run"
    metadata = tomllib.loads((run/"metadata.toml").read_text())
    if metadata["status"] != "COMPLETED":
        raise ValueError(f"production run is {metadata['status']}; no science PASS can be assigned")
    with (run/"modes.tsv").open() as stream:
        rows = list(csv.DictReader(stream,delimiter="\t"))
    redshifts = sorted({float(r["z"]) for r in rows if r["field"]=="B_y"},reverse=True)
    if len(redshifts)<2:
        raise ValueError("at least two evolved epochs are required")
    z0 = redshifts[0]
    radiation = 4.15e-5/args.hubble**2
    cosmo = lmhd.Cosmology(h=args.hubble,omega_m=matched_units.OMEGA_M,
        omega_b=matched_units.OMEGA_B,omega_lambda=1-matched_units.OMEGA_M-radiation,
        omega_r=radiation,hydrogen_mass_fraction=matched_units.HYDROGEN_MASS_FRACTION)
    hydrogen = lmhd.PeeblesIonization(cosmo,zmax=3600,zmin=min(400,args.zend),samples=12000,recfast_v2=True)
    temperature = lmhd.ComptonTemperature(cosmo,hydrogen,zmax=3600,zmin=min(400,args.zend),samples=12000)
    table = lya_rt.ResponseTable.load(args.response_table) if args.response_table else None
    k = matched_units.comoving_k_mpc(1,args.box_ckpc_h,args.hubble)

    def rates(z):
        local = lmhd.recfast_v2_local_response(cosmo,z,hydrogen(z),temperature(z))
        if table is not None and table.redshift[0] <= z <= table.redshift[-1]:
            delta = table.rates(z,k,differential_only=True)
            delta[2] = table.rates(z,k)[2]
            local = local + delta
        return local

    # Use the actual transferred Fourier state, without fitting any evolved
    # amplitude. Runtime axes (x,y,z) map to oracle axes (z,y,x).
    mode = lambda field: physical_mode(mode_at(rows,field,z0),field,z0,handoff)
    velocity = handoff["velocity_unit_cm_s_at_handoff"]
    initial = np.array([
        mode("B_z"), mode("B_y"),mode_at(rows,"gas_delta",z0),
        mode_at(rows,"gas_v_z",z0)*velocity,mode("gas_v_y"),mode("gas_v_x"),
        mode_at(rows,"photon_delta",z0),mode_at(rows,"photon_v_z",z0)*velocity,
        mode("photon_v_y"),mode_at(rows,"photon_v_x",z0)*velocity,
        mode("dm_delta"),mode("dm_v_x"),mode("x_HII"),
        mode_at(rows,"gas_temperature_fraction",z0)*temperature(z0),
    ],dtype=np.complex128)
    reference = lmhd.evolve_photon_p1(lmhd.ReferenceMode(k_mpc=k,theta=args.theta,
        b0_gauss=args.field_pg*1e-12,perturbation_fraction=args.epsilon),cosmo,z0,
        redshifts[-1],redshifts,hydrogen,max_dln_a=args.max_dlna,
        include_gravity=True,include_dark_matter=True,include_photon_stress=True,
        include_helium_electrons=True,recombination_rates=rates,
        baryon_temperature=temperature,initial_state=initial)
    comparisons=[]
    for i,z in enumerate(reference.redshift):
        for field,target in FIELDS.items():
            candidate=physical_mode(mode_at(rows,field,float(z)),field,float(z),handoff)
            expected=complex(reference.field(target)[i])
            floor=1e-3 if field in VELOCITIES else 1e-6 if field=="x_HII" else 1e-7
            status="INITIAL" if i==0 else "BELOW_FLOOR" if abs(expected)<floor else "PASS"
            # The reduced oracle does not evolve explicit helium perturbations.
            if field=="x_HII" and z>1700:
                status="HELIUM_MODEL_DIAGNOSTIC"
            amplitude=abs(abs(candidate)/abs(expected)-1) if abs(expected)>floor else math.nan
            phase=abs(np.angle(candidate/expected)) if abs(expected)>floor else math.nan
            if status=="PASS" and (not np.isfinite(amplitude+phase) or
                    amplitude>args.amplitude_tolerance or phase>args.phase_tolerance):
                status="FAIL"
            comparisons.append(dict(z=float(z),field=field,real=candidate.real,imag=candidate.imag,
                reference_real=expected.real,reference_imag=expected.imag,
                amplitude_error=amplitude,phase_error_rad=phase,signal_floor=floor,status=status))
    active=[r for r in comparisons if r["status"] in ("PASS","FAIL")]
    if not active:
        raise ValueError("no resolved evolved observables; comparison would be vacuous")
    passed=all(r["status"]=="PASS" for r in active)
    gate=dict(status="PASS" if passed else "FAIL",scope="resolved mean-field full-MHD/shared photon-P1 modes",
        source_fingerprint_sha256=metadata["source_fingerprint_sha256"],
        reference_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),Path(lmhd.__file__),Path(lya_rt.__file__))},
        initialization="actual jointly imported modes; no evolved amplitude fitting",
        background="independent RECFAST-v2 hydrogen/temperature and Saha helium",
        explicit_helium_perturbation="model extension; xHII above z1700 diagnostic",
        thermal_perturbations="full-network extension; not gated against reduced oracle",
        response_table=str(args.response_table) if args.response_table else None,
        amplitude_tolerance=args.amplitude_tolerance,phase_tolerance_rad=args.phase_tolerance,
        active_observables=len(active),failures=[r for r in active if r["status"]=="FAIL"],
        max_amplitude_error=max(r["amplitude_error"] for r in active),
        max_phase_error_rad=max(r["phase_error_rad"] for r in active),
        mode_input_sha256=hashlib.sha256((run/"modes.tsv").read_bytes()).hexdigest())
    with (output/"comparison.tsv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader(); writer.writerows(comparisons)
    (output/"gate.json").write_text(json.dumps(gate,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout="constrained")
    for ax,field in zip(axes.flat,("B_y","gas_delta","gas_v_x","dm_delta")):
        selected=[r for r in comparisons if r["field"]==field]
        ax.plot([r["z"] for r in selected],[abs(complex(r["real"],r["imag"])) for r in selected],label="3D MHD")
        ax.plot([r["z"] for r in selected],[abs(complex(r["reference_real"],r["reference_imag"])) for r in selected],"--",label="P1 oracle")
        ax.invert_xaxis(); ax.set(xlabel="Redshift",ylabel=field)
    axes.flat[0].legend()
    fig.suptitle(f"Standalone mean-field comparison: {gate['status']}")
    fig.savefig(output/"comparison.pdf");fig.savefig(output/"comparison.png",dpi=160);plt.close(fig)
    return gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--backend",choices=("cpu","metal","cuda"),default="cpu")
    parser.add_argument("--resolution",type=int,default=32)
    parser.add_argument("--source-zstart",type=float,default=1e5)
    parser.add_argument("--zstart",type=float,default=3200)
    parser.add_argument("--zend",type=float,default=3100)
    parser.add_argument("--field-pg",type=float,default=300)
    parser.add_argument("--epsilon",type=float,default=0.025)
    parser.add_argument("--theta",type=float,default=math.pi/4)
    parser.add_argument("--box-ckpc-h",type=float,default=1)
    parser.add_argument("--hubble",type=float,default=0.674)
    parser.add_argument("--outputs",type=int,default=9)
    parser.add_argument("--cfl",type=float,default=0.2)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--julia",default="julia")
    parser.add_argument("--max-dlna",type=float,default=1e-4)
    parser.add_argument("--amplitude-tolerance",type=float,default=0.05)
    parser.add_argument("--phase-tolerance",type=float,default=0.03)
    parser.add_argument("--response-table",type=Path)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.source_zstart>args.zstart>args.zend>0 or args.outputs<2 or args.resolution<8 or args.resolution%2:
        parser.error("require source-zstart > zstart > zend >0, even resolution >=8, outputs >=2")
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("use a new or empty comparison directory")
    output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","tools.reference.build_p1_meanfield_handoff",str(output/"handoff"),
        "--resolution",str(args.resolution),"--zstart",str(args.source_zstart),
        "--zhandoff",str(args.zstart),"--field-pg",str(args.field_pg),"--epsilon",str(args.epsilon),
        "--theta",str(args.theta),"--box-ckpc-h",str(args.box_ckpc_h),"--hubble",str(args.hubble),
        "--max-dlna",str(args.max_dlna),"--hydrogen-storage","centered"]
    if args.response_table:
        command += ["--response-table",str(args.response_table.resolve())]
    with (output/"handoff.log").open("w") as stream:
        subprocess.run(command,cwd=ROOT,check=True,stdout=stream)
    handoff=json.loads((output/"handoff"/"metadata.json").read_text())
    settings=case_settings(handoff,output/"handoff",args)
    config=output/"case.toml"
    write_config(config,settings)
    project=ROOT if args.backend=="cpu" else ROOT/"environments"/args.backend
    launch=[args.julia,f"--project={project}",f"--threads={args.threads}",str(ROOT/"scripts/run_pmf.jl"),str(config),f"--output={output/'run'}"]
    print("Prepared:",config,"\nLaunch:"," ".join(launch),flush=True)
    if not args.execute:
        print("Add --execute in a new output directory to run and assess the complete comparison.")
        return 0
    subprocess.run(launch,cwd=ROOT,check=True)
    gate=assess(output,handoff,args)
    print(json.dumps(gate,indent=2))
    return 0 if gate["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
