

from __future__ import annotations
import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")
import time
import numpy as np
import amr
import anisotropy
import functions
import fan
import mesh as mesh_lib   # aliased to avoid collision with local `mesh` variable
import optimizer
import visualization


# ─── (1) PARAMETERS ──────────────────────────────────────────────────────────

PARAM = {
    "yield_criterion": "mohr_coulomb",   # tresca | mohr_coulomb
    "uw": 20.0,
    "fi": 25.0,
    "cohesion": "computed",              # derived from attrac: c = a * tan(phi)
    "attrac": 10.0,
}

MODE = "load_max"                      # load_max | body_force_max | none
SOLVER = "CLARABEL"                       # MOSEK | ECOS | CLARABEL | CVXOPT

GEOMETRY = {
    "kind": "strip_footing",
    "B": 3.0,                                 # full footing width (the strip is centred on x=0)
    "W": 30.0,   # total domain width (full extent, regardless of `mirrored`)
                 # Interpretation:
                 #   mirrored=True  -> full domain:  x ∈ [-W/2, W/2]
                 #   mirrored=False -> half domain:  x ∈ [0, W/2]
    "D": 15,                                 # domain depth: y ∈ [-D, 0]
    "mirrored": False,                        # False = half-domain (exploits symmetry); True = full domain
    "n_elements": 1000,
}

ROUGHNESS = 1   # 0 = smooth (sigma_xy = 0 enforced); 1 = rough (sigma_xy free)


RELAX_FOOTING_CORNER = True

BOUNDARY_CONDITIONS = [
    {"boundary": "free",     "fixed": [0.0, 0.0]},
    {"boundary": "footing",  "scaled": [0.0, -1.0],
     "constrain_x": (ROUGHNESS == 0), "constrain_y": True},
    {"boundary": "symmetry", "fixed": [0.0, 0.0],
     "constrain_x": False, "constrain_y": True},
]

# AMR
AMR = True
METHOD = "L hessian"                 # L value | L gradient | L hessian
TARGET_N = 500
N_ITERATIONS = 3
GROWTH_FACTOR = 1.5

# Nested AMR: when True, AMR refines by conforming longest-edge bisection of
# the previous mesh instead of gmsh remeshing -> the lower-bound q/c' is
# monotone non-decreasing. A fan is allowed (built once at iter 0, then
# refined in place by bisection). Implies AMR (no need to also set AMR=True);
# requires ANISOTROPY=False.
NESTED = True
DORFLER_THETA = 2    # bulk-marking fraction (nested adaptive only)
NESTED_UNIFORM = False        # when NESTED: refine the largest-area elements
                                # (quasi-uniform) to grow NE by ~GROWTH_FACTOR
                                # per step, instead of adaptive marking.

# The nested-uniform convergence baseline is run on ECOS only: when
# NESTED_UNIFORM is active, force SOLVER = "ECOS" regardless of the value
# set above. (CLARABEL/MOSEK remain as the optimizer's failure-only fallback.)
if NESTED_UNIFORM:
    SOLVER = "ECOS"


ANISOTROPY = False
ANISO_RATIO_MAX =5.0              # cap on h_long / h_short per element

# uniform-Fan
UNIFORM_FAN = False
FAN_RADIUS = 1.0
M_RADIAL = 3        
N_ANGULAR = 18     
                     

# Adaptive-fan tuners (only used when ADAPTIVE_FAN = True)
ADAPTIVE_FAN =True
ADAPTIVE_FAN_GROW_N = True
ADAPTIVE_FAN_MAX_N =64
ADAPTIVE_FAN_CONC_THRESHOLD = 2.0
ADAPTIVE_FAN_GROWTH_PER_ITER = 1.5
ADAPTIVE_FAN_MATCH_H_MIN =True

# Outputs
LABEL = "Case_drained_soil_with_self_weight"
QUICK_VIEW = True
MAKE_VTK = False
SAVE_FIGS = False
SAVE_LOG = False


# ─── (2) COMPOSITION (print/table/plot helpers) ──────────────────────────────

def strategy_label() -> str:
    
    fan_kind = ("AF" if ADAPTIVE_FAN
                else "UF" if UNIFORM_FAN else None)
    if not AMR and not NESTED:
        return "Uniform" + (f" + {fan_kind}" if fan_kind else "")
    if ANISOTROPY:
        # Scheme 7 (no fan) or scheme 8 (with fan).
        base = "L Hess + A"
        return f"{base} + {fan_kind}" if fan_kind else base
    method_short = {"L value": "L Value", "L gradient": "L Grad", "L hessian": "L Hess"}
    base = method_short.get(METHOD, METHOD)
    if NESTED:
        # LEB (nested-subdivision) table -- the table title carries "nested",
        # so the row only names the strategy. "+ fan" => fan built once at
        # iter 0, then bisection-refined. Uniform mode ignores METHOD.
        if NESTED_UNIFORM:
            return "Uniform + fan" if fan_kind else "Uniform"
        return f"{base} + fan" if fan_kind else base
    return f"{base} + {fan_kind}" if fan_kind else base


CSV_COLUMNS = [
    "Mesh_type", "AMR_it", "NE", "q", "Neq", "N",
    "T_sol", "T_tot", "Solver",
    "Pres", "Dres", "Gap",
]


def csv_row(
    step: amr.AdaptiveStep, *,
    label: str, cohesion: float, t0: float,
) -> list:
    
    qc = step.solution.alpha / cohesion if cohesion > 0 else float("nan")
    # Single N_angular cell. Half-mode has only the right fan, so we just show
    # that number; full-mode normally has R == L for symmetric loads, so still
    # one number; only asymmetric full-mode runs need both, formatted "R/L".
    if step.n_angular_R is None:
        n_str = ""
    elif step.n_angular_L is None or step.n_angular_L == step.n_angular_R:
        n_str = str(step.n_angular_R)
    else:
        n_str = f"{step.n_angular_R}/{step.n_angular_L}"
    stats = step.solution.solver_stats or {}
    solver_used = stats.get("solver", "")
    solve_t = stats.get("solve_time")
    solve_t_str = f"{solve_t:.3f}" if solve_t is not None else ""
    pres = stats.get("pres")
    dres = stats.get("dres")
    gap = stats.get("gap")
    pres_str = f"{pres:.2e}" if isinstance(pres, (int, float)) else ""
    dres_str = f"{dres:.2e}" if isinstance(dres, (int, float)) else ""
    gap_str = f"{gap:.2e}" if isinstance(gap, (int, float)) else ""
    elapsed = time.time() - t0
    return [
        label,
        step.iteration,
        step.mesh.n_tri,
        f"{step.solution.alpha:.2f}",
        f"{qc:.3f}",
        n_str,
        solve_t_str, f"{elapsed:.3f}", solver_used,
        pres_str, dres_str, gap_str,
    ]


_ITER_COL_INDEX = 1  # CSV_COLUMNS index of "AMR_it"


_TABLE_COL_WIDTHS = {
    "Mesh_type": 15, "AMR_it": 4, "NE": 4,
    "q": 6, "Neq": 5, "N": 2,
    "T_sol": 6, "T_tot": 6, "Solver": 8,
    "Pres": 8, "Dres": 8, "Gap": 8,
}


def _table_row(values: list, columns: list[str]) -> str:
    """One pipe-bordered row matching CSV_COLUMNS widths."""
    cells = []
    for col, val in zip(columns, values):
        w = max(len(col), _TABLE_COL_WIDTHS.get(col, 10))
        cells.append(str(val).ljust(w))
    return "| " + " | ".join(cells) + " |"


def _table_separator(columns: list[str]) -> str:
    """Dashed border row matching the per-column widths used by _table_row.
    Each column gap (`| `, ` | `, ` |`) contributes 2 padding chars to span."""
    parts = []
    for col in columns:
        w = max(len(col), _TABLE_COL_WIDTHS.get(col, 10))
        parts.append("-" * (w + 2))
    return "|" + "|".join(parts) + "|"


def print_results(step: amr.AdaptiveStep, *, cohesion: float, t0: float) -> None:
    """Build the row for `step` and echo it to the terminal as one row of
    an aligned table (header printed once on iter 0).
    """
    row = csv_row(step, label=strategy_label(), cohesion=cohesion, t0=t0)
    if step.iteration == 0:
        print(_table_row(CSV_COLUMNS, CSV_COLUMNS))
        print(_table_separator(CSV_COLUMNS))
    print(_table_row(row, CSV_COLUMNS))


def print_header(*, cohesion: float, fan_on: bool, amr_on: bool) -> None:
    geo_str = (
        f"strip_footing ({'full' if GEOMETRY.get('mirrored') else 'half'}, "
        f"B={GEOMETRY['B']}, W={GEOMETRY['W']}, D={GEOMETRY['D']})"
    )
    if amr_on:
        if ANISOTROPY:
            amr_str = (f"on -- anisotropic (Hessian, "
                       f"ratio<={ANISO_RATIO_MAX}), target_n={TARGET_N}, "
                       f"n_iter={N_ITERATIONS}, growth={GROWTH_FACTOR}")
        elif NESTED and NESTED_UNIFORM:
            amr_str = (f"on -- nested uniform (largest-area LEB, "
                       f"growth={GROWTH_FACTOR}, seed target_n={TARGET_N}, "
                       f"n_iter={N_ITERATIONS})")
        elif NESTED:
            amr_str = (f"on -- nested LEB (method={METHOD}, "
                       f"theta={DORFLER_THETA}, seed target_n={TARGET_N}, "
                       f"n_iter={N_ITERATIONS})")
        else:
            amr_str = (f"on (method={METHOD}, target_n={TARGET_N}, "
                       f"n_iter={N_ITERATIONS}, growth={GROWTH_FACTOR})")
    else:
        amr_str = "off"
    if not fan_on:
        fan_str = "none"
    else:
        kind = ("nested-refined" if NESTED
                else "adaptive" if ADAPTIVE_FAN else "uniform")
        fan_str = (f"{kind} (R={FAN_RADIUS}, M={M_RADIAL}, N={N_ANGULAR})")
        if ADAPTIVE_FAN and not amr_on:
            fan_str += f", n_iter={N_ITERATIONS}"

    footing_str = f"{'smooth' if ROUGHNESS == 0 else 'rough'} (R = {ROUGHNESS})"
    uw_val = float(PARAM.get("uw", 0.0))
    fixed_bf, scaled_bf = optimizer.body_force_for_mode(MODE, uw=uw_val)
    bf_str = (f"fixed = {fixed_bf}, scaled = {scaled_bf}  "
              f"(b_y_fixed = {fixed_bf[1]:.2f}, gamma = {uw_val:.2f})")
    print(
        "Configuration\n"
        f"  yield_criterion : {PARAM['yield_criterion']} "
        f"(fi={PARAM['fi']}, c={cohesion:.4f})\n"
        f"  mode            : {MODE}\n"
        f"  geometry        : {geo_str}\n"
        f"  Footing         : {footing_str}\n"
        f"  Body force      : {bf_str}\n"
        f"  AMR             : {amr_str}\n"
        f"  Fan             : {fan_str}\n"
        f"  Solver          : {SOLVER}\n"
    )




# ─── (3) CORNER-SINGULARITY relaxation ──────────────────────────────────────

_ORIGINAL_BOUNDARY_CONDITIONS = functions.boundaries_condition

# Relaxation-zone radius -- fixed at 0.14, validated by the r-sweep
# (outputs/sweep_q_vs_r.png). Defined here, not in the parameter block, so it
# is treated as a fixed property of the method rather than a per-run setting.
CORNER_RELAX_RADIUS = 0.15


def _in_corner_zone(coord) -> bool:
    """True if (x, y) is within CORNER_RELAX_RADIUS of a footing-edge
    singular corner.

    Half-mode: corner at (+B/2, 0). Full-mode (mirrored): also at (-B/2, 0).
    Uses Euclidean distance so the zone is a disk, not an L-infinity box.
    """
    cx = GEOMETRY["B"] / 2.0
    r2 = CORNER_RELAX_RADIUS ** 2
    x = float(coord[0])
    y = float(coord[1])
    if (x - cx) ** 2 + y ** 2 < r2:
        return True
    if GEOMETRY.get("mirrored", False):
        if (x + cx) ** 2 + y ** 2 < r2:
            return True
    return False


def _boundary_conditions_relaxed(mesh, boundary_conditions):
   
    A_orig, b_fix_orig, b_alpha_orig = _ORIGINAL_BOUNDARY_CONDITIONS(mesh, boundary_conditions)

    if not RELAX_FOOTING_CORNER:
        return A_orig, b_fix_orig, b_alpha_orig

    from optimizer import TractionBC
    bc_map = {bc.boundary: bc for bc in boundary_conditions if isinstance(bc, TractionBC)}

    keep = np.ones(A_orig.shape[0], dtype=bool)
    next_row = 0
    n_skipped = 0
    for ib in range(mesh.n_boundary_edges):
        _, na, nb = (int(x) for x in mesh.boundary_edges[ib])
        tag = mesh.boundary_edge_tag[ib]
        if tag not in bc_map:
            continue
        bc = bc_map[tag]
        rows_added = int(bc.constrain_x) + int(bc.constrain_y)
        # Relax this edge ONLY if the WHOLE edge is in the corner zone --
        # the both-endpoints test that keeps the refined LB nested.
        edge_relaxed = (_in_corner_zone(mesh.nodes[na])
                        and _in_corner_zone(mesh.nodes[nb]))
        for _ in range(2):                      # one block per edge endpoint
            if edge_relaxed:
                for r in range(rows_added):
                    keep[next_row + r] = False
                n_skipped += rows_added
            next_row += rows_added

    if n_skipped == 0:
        return A_orig, b_fix_orig, b_alpha_orig
    return A_orig[keep], b_fix_orig[keep], b_alpha_orig[keep]


functions.boundaries_condition = _boundary_conditions_relaxed


# ─── (4) MAIN (run configuration → optimizer → plot) ─────────────────────────

def _build_fan_cfg() -> dict:
    
    return {
        "geometry": GEOMETRY,
        "fan_radius": FAN_RADIUS,
        "M_radial": M_RADIAL,
        "N_angular": N_ANGULAR,
        "method": METHOD,
        "target_n": TARGET_N,
        "n_iterations": N_ITERATIONS,
        "growth_factor": GROWTH_FACTOR,
        "adaptive_fan": ADAPTIVE_FAN,
        "adaptive_fan_grow_N": ADAPTIVE_FAN_GROW_N,
        "adaptive_fan_max_N": ADAPTIVE_FAN_MAX_N,
        "adaptive_fan_conc_threshold": ADAPTIVE_FAN_CONC_THRESHOLD,
        "adaptive_fan_growth_per_iter": ADAPTIVE_FAN_GROWTH_PER_ITER,
        "adaptive_fan_match_h_min": ADAPTIVE_FAN_MATCH_H_MIN,
        # Nested fan AMR (run_fan_nested_loop): marking fraction; uniform=True
        # -> grow NE by ~growth_factor via largest-area refinement.
        "dorfler_theta": DORFLER_THETA,
        "uniform": NESTED_UNIFORM,
        # Scheme 8: when True, the outer (non-fan) mesh is rebuilt each
        # iteration via gmsh BAMG under an anisotropic metric tensor derived
        # from the recovered Hessian of lambda. The fan structure is unchanged.
        "anisotropy": ANISOTROPY,
        "aniso_ratio_max": ANISO_RATIO_MAX,
    }


def main() -> int:
    if SAVE_LOG:
        visualization.start_log_capture(f"outputs/{LABEL}.txt")
    if UNIFORM_FAN and ADAPTIVE_FAN:
        raise ValueError("UNIFORM_FAN and ADAPTIVE_FAN cannot both be True.")
    if NESTED and ANISOTROPY:
        raise ValueError(
            "NESTED and ANISOTROPY cannot both be True. A fan is allowed "
            "(built once, then refined by longest-edge bisection)."
        )
    fan_on = UNIFORM_FAN or ADAPTIVE_FAN
    # NESTED is a refinement-loop mode, so it implies AMR: turning on NESTED
    # alone is enough (no need to also set AMR=True).
    amr_on = AMR or NESTED

    t0 = time.time()
    yc = functions.from_config(PARAM)
    cohesion = functions.cohesion_from_param(PARAM, float(PARAM.get("fi", 0.0)), verbose=False)
    bcs = optimizer.boundary_conditions_from_config(BOUNDARY_CONDITIONS)

    fixed_bf, scaled_bf = optimizer.body_force_for_mode(MODE, uw=float(PARAM.get("uw", 0.0)))

    # MODE = "none" disables load maximization: solve a feasibility-only LB
    # (alpha pinned to 0), then check that the recovered sigma_yy along a
    # vertical column matches the geostatic relation sigma_yy = gamma * y
    # (tension-positive, y < 0). Useful as a self-weight stability test --
    # if the soil at the given (c, phi, gamma) cannot carry its own weight,
    # MOSEK returns infeasible.
    if MODE == "none":
        print_header(cohesion=cohesion, fan_on=fan_on, amr_on=amr_on)
        print("MODE = 'none': feasibility-only solve (alpha pinned to 0).")
        geom = mesh_lib.geometry_from_config(GEOMETRY)
        mesh = mesh_lib.mesh_parameter(
            geom, target_n=GEOMETRY["n_elements"], verbose=False,
        )
        sol = optimizer.optimize_socp(
            mesh, yc,
            fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
            boundary_conditions=bcs, solver=SOLVER,
            feasibility_only=True,
        )
        gamma = float(PARAM.get("uw", 0.0))
        # Sample sigma_yy at element centroids in a thin vertical column near
        # x = W/4 (away from the footing-edge perturbation) and bin by depth.
        x_target = 0.25 * float(GEOMETRY["W"])
        D = float(GEOMETRY["D"])
        cx = mesh.nodes[mesh.elements].mean(axis=1)   # (n_tri, 2)
        col_mask = np.abs(cx[:, 0] - x_target) < 0.5   # ~1m wide column
        depths = np.linspace(0.0, D, 11)[1:]            # 10 sample depths
        print(f"\nGeostatic check (column at x ~ {x_target:.1f}):")
        print(f"  {'depth z':>8}  {'sigma_yy (LB)':>14}  {'-gamma*z':>10}  {'rel err %':>10}")
        for z in depths:
            band = col_mask & (np.abs(cx[:, 1] + z) < 0.5)
            if not band.any():
                continue
            syy_lb = float(sol.sigma[band, :, 1].mean())
            syy_geo = -gamma * z
            err = 100.0 * abs(syy_lb - syy_geo) / abs(syy_geo) if syy_geo != 0 else 0.0
            print(f"  {z:>8.2f}  {syy_lb:>14.3f}  {syy_geo:>10.3f}  {err:>10.2f}")
        visualization.make_POS(mesh, sol, path=f"outputs/{LABEL}.pos")
        if QUICK_VIEW:
            visualization.plot_solution_summary(
                mesh, sol, label=LABEL, alpha_fmt=".4f",
                save_dir="outputs" if SAVE_FIGS else None,
            )
        return 0

    print_header(cohesion=cohesion, fan_on=fan_on, amr_on=amr_on)


    def on_iter(step: amr.AdaptiveStep) -> None:
        print_results(step, cohesion=cohesion, t0=t0)

    history: list[amr.AdaptiveStep] | None = None

    geom = mesh_lib.geometry_from_config(GEOMETRY)
    seed_target = TARGET_N if amr_on else int(GEOMETRY["n_elements"])
    cfg: dict | None = None
    if fan_on:
        cfg = _build_fan_cfg()
        half_sym = not bool(GEOMETRY.get("mirrored", False))
        seed_mesh = fan.mesh_fan(
            cfg, target_n=seed_target,
            n_angular_R=N_ANGULAR,
            n_angular_L=None if half_sym else N_ANGULAR,
            fan_h_min=None,
        )
    else:
        seed_mesh = mesh_lib.mesh_parameter(
            geom, target_n=seed_target, verbose=False,
        )

    if fan_on and amr_on:
        if NESTED:
            # Fan built once at iter 0, then the whole mesh (fan + outer) is
            # refined by conforming longest-edge bisection -> monotone q/c'.
            history = amr.run_fan_nested_loop(
                seed_mesh, cfg, yc,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER, on_iter=on_iter,
            )
        else:
            history = amr.run_fan_amr_loop(
                seed_mesh, cfg, yc,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER, on_iter=on_iter,
            )
        sol = history[-1].solution
        mesh = history[-1].mesh
    elif fan_on and not amr_on:
        # No AMR -> use GEOMETRY["n_elements"], matching the plain uniform-mesh
        # branch, so a single setting controls element count across all
        # non-AMR strategies.
        if ADAPTIVE_FAN:
            # Uniform outer mesh + adaptive-fan iteration: rebuild only the
            # fan's theta_R/theta_L each iteration; outer mesh stays uniform.
            history = amr.run_fan_only_loop(
                seed_mesh, cfg, yc,
                target_n=int(GEOMETRY["n_elements"]),
                n_iterations=N_ITERATIONS,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER, on_iter=on_iter,
            )
            sol = history[-1].solution
            mesh = history[-1].mesh
        else:
            # Uniform fan: angular spacing is fixed, so iterating produces an
            # identical mesh -- single solve.
            mesh = seed_mesh
            sol = optimizer.optimize_socp(
                mesh, yc,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER,
            )
            half_sym = not bool(GEOMETRY.get("mirrored", False))
            step = amr.AdaptiveStep(
                0, mesh, sol,
                n_angular_R=N_ANGULAR,
                n_angular_L=None if half_sym else N_ANGULAR,
            )
            print_results(step, cohesion=cohesion, t0=t0)
    elif not fan_on and amr_on:
        if ANISOTROPY:
            history = anisotropy.run_anisotropic_amr_loop(
                seed_mesh, geom, yc,
                target_n=TARGET_N,
                n_iterations=N_ITERATIONS, growth_factor=GROWTH_FACTOR,
                aniso_ratio_max=ANISO_RATIO_MAX,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER, on_iter=on_iter,
            )
        else:
            history = amr.run_loop(
                seed_mesh, geom, yc,
                method=METHOD, target_n=TARGET_N,
                n_iterations=N_ITERATIONS, growth_factor=GROWTH_FACTOR,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER, on_iter=on_iter,
                nested=NESTED, dorfler_theta=DORFLER_THETA,
                uniform=NESTED_UNIFORM,
            )
        sol = history[-1].solution
        mesh = history[-1].mesh
    else:
        mesh = seed_mesh
        sol = optimizer.optimize_socp(
            mesh, yc,
            fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
            boundary_conditions=bcs, solver=SOLVER,
        )
        step = amr.AdaptiveStep(0, mesh, sol)
        print_results(step, cohesion=cohesion, t0=t0)

    print(f"\nTotal execution time: {time.time() - t0:.2f} seconds")


    if MAKE_VTK:
        vtk_path = visualization.get_next_available_filename("outputs", LABEL, ".vtk")
        visualization.make_VTK(mesh, sol, path=vtk_path)
        print(f"  vtk     -> {vtk_path}")

    visualization.make_POS(mesh, sol, path=f"outputs/{LABEL}.pos")

    if QUICK_VIEW:
        if history is not None and len(history) > 1:
            visualization.plot_amr_history(
                history, label=LABEL,
                reference_value=cohesion, reference_label="q/c'",
                fmt=".4f", adaptive_fan=ADAPTIVE_FAN,
                gradient_cmap="viridis",
                save_dir="outputs" if SAVE_FIGS else None,
            )
        else:
            visualization.plot_solution_summary(
                mesh, sol, label=LABEL, alpha_fmt=".4f",
                save_dir="outputs" if SAVE_FIGS else None,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
