

from __future__ import annotations
import time
import amr
import anisotropy
import fan
import mesh as mesh_lib   
import optimizer
import visualization
import functions



# ─── (1) PARAMETERS ──────────────────────────────────────────────────────────

PARAM = {
    "yield_criterion": "tresca",   
    "uw": 0.0,
    "fi": 0.0,
    "su": 10.0,                               
}

MODE = "load_max"                  # load_max 
SOLVER = "MOSEK"                   # MOSEK | ECOS | CLARABEL | CVXOPT

GEOMETRY = {
    "kind": "strip_footing",
    "B": 3.0,                                 # full footing width (the strip is centred on x=0)
    "W": 30.0,                                # domain width (interpreted by the chosen builder):
                                              #   mirrored=False  -> meshed extent x ∈ [0, W]      (W = half-width)
                                              #   mirrored=True   -> meshed extent x ∈ [-W/2, W/2] (W = full width)
    "D":10,                                  # domain depth: y ∈ [-D, 0]
    "mirrored": True,                         # False = half-domain (exploits symmetry); True = full domain
    "n_elements": 500
}

BOUNDARY_CONDITIONS = [
    {"boundary": "free",     "fixed": [0.0, 0.0]},
    {"boundary": "footing",  "scaled": [0.0, -1.0]},
    {"boundary": "symmetry", "fixed": [0.0, 0.0],
     "constrain_x": False, "constrain_y": True},
]

#
AMR = True
METHOD = "L hessian"
TARGET_N = 300
N_ITERATIONS = 2
GROWTH_FACTOR = 3              # gmsh AMR: target NE x this per iteration.
                                   
NESTED = True
DORFLER_THETA = 6             # bulk-marking fraction (nested AMR only)
NESTED_UNIFORM = False              # when NESTED: refine the largest-area
                                    


ANISOTROPY = False
ANISO_RATIO_MAX =20.0              # cap on h_long / h_short per element

# uniform-Fan
UNIFORM_FAN = False
FAN_RADIUS = 1
M_RADIAL = 3
N_ANGULAR = 18

# Adaptive-fan tuners (only used when ADAPTIVE_FAN = True)
ADAPTIVE_FAN = True
ADAPTIVE_FAN_GROW_N = True
ADAPTIVE_FAN_MAX_N = 64
ADAPTIVE_FAN_CONC_THRESHOLD = 2.0
ADAPTIVE_FAN_GROWTH_PER_ITER = 1.5
ADAPTIVE_FAN_MATCH_H_MIN = True

# Outputs
LABEL = "Case_undrained_soil"
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
    if NESTED:
        # LEB (nested-subdivision) table. The table title carries the
        # "nested" distinction, so the row label names only the strategy.
        # "+ fan" => fan built once at iter 0, then bisection-refined.
        if NESTED_UNIFORM:
            return "Uniform + fan" if fan_kind else "Uniform"
        if fan_kind:
            short = {"L value": "L value", "L gradient": "L grad",
                     "L hessian": "L Hess"}
            return short.get(METHOD, METHOD) + " + fan"
        full = {"L value": "L value", "L gradient": "L gradient",
                "L hessian": "L Hessian"}
        return full.get(METHOD, METHOD)

    # gmsh (size-field remeshing) table.
    method_short = {"L value": "L Value", "L gradient": "L Grad", "L hessian": "L Hess"}
    base = method_short.get(METHOD, METHOD)
    return f"{base} + {fan_kind}" if fan_kind else base


CSV_COLUMNS = [
    "Mesh_type", "AMR_it", "NE", "Nc", "N",
    "T_sol", "T_tot", "Solver",
    "Pres", "Dres", "Gap",
]


def csv_row(
    step: amr.AdaptiveStep, *,
    label: str, su: float, t0: float,
) -> list:
    
    Nc = step.solution.alpha / su if su > 0 else float("nan")
    # N_angular cell: single number = max(right, left) fan wedge count, so
    # asymmetric AMR growth still displays as one value. "-" when there is
    # no fan in this run.
    if step.n_angular_R is None:
        n_str = "-"
    elif step.n_angular_L is None:
        n_str = str(step.n_angular_R)
    else:
        n_str = str(max(int(step.n_angular_R), int(step.n_angular_L)))
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
        f"{Nc:.3f}",
        n_str,
        solve_t_str, f"{elapsed:.3f}", solver_used,
        pres_str, dres_str, gap_str,
    ]


_ITER_COL_INDEX = 1  # CSV_COLUMNS index of "AMR_it"


_TABLE_COL_WIDTHS = {
    "Mesh_type": 13, "AMR_it": 4, "NE": 5, "Nc": 5, "N": 2,
    "T_sol": 6, "T_tot": 6, "Solver": 5,
    "Pres": 9, "Dres": 9, "Gap": 9,
}


_NUMERIC_COLS = {"AMR_it", "NE", "Nc", "N", "T_sol", "T_tot", "Pres", "Dres", "Gap"}


def _table_row(values: list, columns: list[str]) -> str:
    
    cells = []
    for col, val in zip(columns, values):
        w = max(len(col), _TABLE_COL_WIDTHS.get(col, 10))
        s = str(val)
        if col in _NUMERIC_COLS:
            cells.append(s.rjust(w) + " ")
        else:
            cells.append(" " + s.ljust(w) + " ")
    return "|" + "|".join(cells) + "|"


def _table_separator(columns: list[str]) -> str:
    
    parts = []
    for col in columns:
        w = max(len(col), _TABLE_COL_WIDTHS.get(col, 10))
        span = (w + 1) if col in _NUMERIC_COLS else (w + 2)
        parts.append("-" * span)
    return "|" + "|".join(parts) + "|"


_PREV_ROW: list | None = None
_DISPLAY_ITER = 0     # consecutive AMR_iter numbering across accepted rows
_NE_COL_INDEX = 2     # CSV_COLUMNS index of "NE"
_N_COL_INDEX = 4      # CSV_COLUMNS index of "N"


def _is_duplicate_row(row: list, prev: list | None) -> bool:
    
    if prev is None:
        return False
    try:
        ne_same = int(row[_NE_COL_INDEX]) == int(prev[_NE_COL_INDEX])
        n_same = str(row[_N_COL_INDEX]) == str(prev[_N_COL_INDEX])
        return ne_same and n_same
    except (ValueError, TypeError, IndexError):
        return False


def print_results(step: amr.AdaptiveStep, *, su: float, t0: float) -> None:
    
    global _PREV_ROW, _DISPLAY_ITER
    row = csv_row(step, label=strategy_label(), su=su, t0=t0)
    if step.iteration == 0:
        print(_table_row(CSV_COLUMNS, CSV_COLUMNS))
        print(_table_separator(CSV_COLUMNS))
    if _is_duplicate_row(row, _PREV_ROW):
        return
    # Overwrite the AMR_iter cell with the consecutive accepted-row counter.
    row[_ITER_COL_INDEX] = _DISPLAY_ITER
    print(_table_row(row, CSV_COLUMNS))
    _PREV_ROW = list(row)
    _DISPLAY_ITER += 1


def print_header(*, su: float, fan_on: bool, amr_on: bool) -> None:
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
        fan_str = f"{kind} (R={FAN_RADIUS}, M={M_RADIAL}, N={N_ANGULAR})"
        # Adaptive fan iterates even when AMR is off (uniform outer + fan-only
        # iteration); the iteration count is reported here so the run header
        # remains explicit about it.
        if ADAPTIVE_FAN and not amr_on:
            fan_str += f", n_iter={N_ITERATIONS}"

    print(
        "Configuration\n"
        f"  yield_criterion : {PARAM['yield_criterion']} "
        f"(fi={PARAM['fi']}, su={su:.3f})\n"
        f"  mode            : {MODE}\n"
        f"  geometry        : {geo_str}\n"
        f"  AMR             : {amr_str}\n"
        f"  Fan             : {fan_str}\n"
        f"  Solver          : {SOLVER}\n"
    )


# ─── (3) MAIN (run configuration → optimizer → plot) ─────────────────────────

def _build_fan_cfg() -> dict:
    """Pack the fan-related globals into the dict shape that fan.mesh_fan
    and amr.run_fan_amr_loop expect."""
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
        # Nested fan AMR (run_fan_nested_loop): bulk-marking fraction, and
        # uniform=True -> split every triangle 1->4 instead of marking.
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
    # alone is enough to run the nested loop (no need to also set AMR=True).
    amr_on = AMR or NESTED

    t0 = time.time()

    # 1. Build initial mesh
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

    # 2. Convert config to typed objects
    yc = functions.from_config(PARAM)
    su = functions.cohesion_from_param(PARAM, float(PARAM.get("fi", 0.0)), verbose=False)
    bcs = optimizer.boundary_conditions_from_config(BOUNDARY_CONDITIONS)
    fixed_bf, scaled_bf = optimizer.body_force_for_mode(MODE, uw=float(PARAM.get("uw", 0.0)))

    print_header(su=su, fan_on=fan_on, amr_on=amr_on)

    def on_iter(step: amr.AdaptiveStep) -> None:
        print_results(step, su=su, t0=t0)

    history: list[amr.AdaptiveStep] | None = None

    # 3. Dispatch (passes pre-built seed mesh)
    if fan_on and amr_on:
        if NESTED:
            # Fan built once at iter 0, then the whole mesh (fan + outer) is
            # refined by conforming longest-edge bisection -> monotone Nc.
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
        if ADAPTIVE_FAN:
            # Uniform outer mesh + adaptive-fan iteration: rebuild only the
            # fan's theta_R/theta_L each iteration; outer mesh stays uniform.
            history = amr.run_fan_only_loop(
                seed_mesh, cfg, yc,
                target_n=seed_target,
                n_iterations=N_ITERATIONS,
                fixed_body_force=fixed_bf, scaled_body_force=scaled_bf,
                boundary_conditions=bcs, solver=SOLVER, on_iter=on_iter,
            )
            sol = history[-1].solution
            mesh = history[-1].mesh
        else:
            # Uniform fan: angular spacing is fixed, so iterating produces an
            # identical mesh -- single solve on the seed mesh.
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
            print_results(step, su=su, t0=t0)
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
        print_results(step, su=su, t0=t0)

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
                reference_value=su, reference_label="Nc",
                adaptive_fan=ADAPTIVE_FAN,
                save_dir="outputs" if SAVE_FIGS else None,
            )
        else:
            visualization.plot_solution_summary(
                mesh, sol, label=LABEL,
                save_dir="outputs" if SAVE_FIGS else None,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 