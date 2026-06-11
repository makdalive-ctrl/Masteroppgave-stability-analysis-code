import time
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import cvxpy as cp
import numpy as np

from functions import (
    YieldCriterion,
    assembly_of_equality_constraints,
    objective_function,
)
from mesh import LBMesh


_MOSEK_DEFAULT_TOL = 1.0e-6

_SUPPORTED_SOLVERS = ("MOSEK", "CLARABEL", "ECOS", "SCS", "CVXOPT")

_SOLVER_FALLBACK_CHAIN = {
    "MOSEK":    ("CLARABEL", "ECOS"),
    "CLARABEL": ("ECOS", "MOSEK"),
    "ECOS":     ("CLARABEL", "MOSEK"),
}




@dataclass
class TractionBC:
    boundary: str
    fixed: tuple[float, float] = (0.0, 0.0)
    scaled: tuple[float, float] = (0.0, 0.0)
    constrain_x: bool = True
    constrain_y: bool = True


@dataclass
class LBSolution:
    alpha: float
    sigma: np.ndarray
    lagrange_multiplier: np.ndarray
    status: str
    solver_stats: dict = field(default_factory=dict)





def _solver_opts(name: str) -> dict:
    name = name.upper()
    if name == "MOSEK":
        return {"mosek_params": {
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": _MOSEK_DEFAULT_TOL,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": _MOSEK_DEFAULT_TOL,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": _MOSEK_DEFAULT_TOL,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": _MOSEK_DEFAULT_TOL,
            "MSK_IPAR_NUM_THREADS": 1,
        }}
    if name == "CLARABEL":
        return {"tol_feas": 1.0e-10, "tol_gap_abs": 1.0e-10,
                "tol_gap_rel": 1.0e-10, "max_iter": 500}
    if name == "ECOS":
        return {"abstol": 1.0e-6, "reltol": 1.0e-6,
                "feastol": 1.0e-6, "max_iters": 500}
    return {}




def _build_problem(
    mesh: LBMesh,
    yield_crit: YieldCriterion,
    *,
    fixed_body_force: tuple[float, float],
    scaled_body_force: tuple[float, float],
    boundary_conditions: Sequence[TractionBC],
    feasibility_only: bool,
):
    n_nodes = mesh.n_nodes

    sigma = cp.Variable((n_nodes, 3), name="sigma")
    rho = cp.Variable((n_nodes, 3), name="rho")
    alpha = cp.Variable(name="alpha")

    A_eq, b_fixed, b_alpha_coef = assembly_of_equality_constraints(
        mesh,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
    )
    sigma_flat = cp.vec(sigma.T, order="F")

    D_kls = yield_crit.D
    d_kls = yield_crit.d
    constraints: list = [
        A_eq @ sigma_flat == b_fixed + alpha * b_alpha_coef,
        rho == sigma @ D_kls.T + d_kls,
    ]
    if feasibility_only:
        constraints.append(alpha == 0.0)

    soc_constraints: list = []
    for nid in range(n_nodes):
        soc = cp.SOC(rho[nid, 0], rho[nid, 1:])
        soc_constraints.append(soc)
        constraints.append(soc)

    problem = cp.Problem(objective_function(alpha), constraints)

    return (sigma, rho, alpha, A_eq, b_fixed, b_alpha_coef,
            soc_constraints, problem, constraints)



def _prepare_solver_opts(
    solver: str, solver_opts: dict | None,
) -> tuple[str, dict, bool]:
    solver_uc = solver.upper()
    if solver_uc not in _SUPPORTED_SOLVERS:
        raise ValueError(
            f"unknown solver {solver!r}; supported: {_SUPPORTED_SOLVERS}"
        )
    if solver_uc not in cp.installed_solvers():
        raise RuntimeError(
            f"solver {solver_uc!r} requested but not installed; "
            f"installed: {cp.installed_solvers()}"
        )

    opts = dict(solver_opts or {})
    if solver_uc == "MOSEK" and "mosek_params" not in opts:
        opts["mosek_params"] = {
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": _MOSEK_DEFAULT_TOL,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": _MOSEK_DEFAULT_TOL,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": _MOSEK_DEFAULT_TOL,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": _MOSEK_DEFAULT_TOL,
            "MSK_IPAR_NUM_THREADS": 1,
        }
        opts.setdefault("accept_unknown", True)
    if solver_uc == "CLARABEL":
        opts.setdefault("tol_feas", 1.0e-10)
        opts.setdefault("tol_gap_abs", 1.0e-10)
        opts.setdefault("tol_gap_rel", 1.0e-10)
        opts.setdefault("max_iter", 500)
    if solver_uc == "ECOS":
        opts.setdefault("abstol", 1.0e-6)
        opts.setdefault("reltol", 1.0e-6)
        opts.setdefault("feastol", 1.0e-6)
        opts.setdefault("max_iters", 500)
    accept_unknown = opts.pop("accept_unknown", False)
    return solver_uc, opts, accept_unknown


def _solve_with_fallback(
    problem: cp.Problem,
    solver_uc: str,
    opts: dict,
    accept_unknown: bool,
    alpha: cp.Variable,
    verbose: bool,
) -> tuple[str, float]:
    used_solver = solver_uc
    t_solve_start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*don't support CPP backend.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Solution may be inaccurate.*",
                category=UserWarning,
            )
            problem.solve(solver=solver_uc, verbose=verbose, **opts)
    except cp.error.SolverError:
        _installed = cp.installed_solvers()
        _solved = False
        for _fb in _SOLVER_FALLBACK_CHAIN.get(solver_uc, ()):
            if _fb not in _installed:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*don't support CPP backend.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Solution may be inaccurate.*",
                        category=UserWarning,
                    )
                    problem.solve(solver=_fb, verbose=verbose,
                                  **_solver_opts(_fb))
            except cp.error.SolverError:
                continue
            used_solver = _fb
            _solved = True
            break
        if not _solved:
            if accept_unknown and alpha.value is not None:
                pass
            else:
                raise
    solve_time = time.perf_counter() - t_solve_start
    return used_solver, solve_time


def _check_status(
    problem: cp.Problem,
    alpha: cp.Variable,
    sigma: cp.Variable,
    accept_unknown: bool,
) -> str:
    status = problem.status
    if status not in ("optimal", "optimal_inaccurate"):
        if accept_unknown and alpha.value is not None and sigma.value is not None:
            status = f"{status} (accepted)"
        else:
            raise RuntimeError(f"Solver returned status {problem.status!r}")
    return status




def _extract_solution(
    alpha: cp.Variable,
    sigma: cp.Variable,
    soc_constraints: list,
    n_tri: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    alpha_val_num = float(alpha.value)
    sigma_val = sigma.value.reshape(n_tri, 3, 3)
    plastic = np.zeros((n_tri, 3))
    for k, soc in enumerate(soc_constraints):
        e, li = divmod(k, 3)
        dual = soc.dual_value
        if dual is None:
            continue
        first = dual[0] if isinstance(dual, (list, tuple)) else dual
        arr = np.asarray(first).ravel()
        plastic[e, li] = float(arr[0])
    return alpha_val_num, sigma_val, plastic


def _compute_residuals(
    A_eq,
    b_fixed: np.ndarray,
    b_alpha_coef: np.ndarray,
    alpha_val_num: float,
    sigma: cp.Variable,
    rho: cp.Variable,
    constraints: list,
    problem: cp.Problem,
) -> tuple[float, float, float]:
    sigma_flat_val = np.asarray(sigma.value).ravel()

    eq_residual = A_eq @ sigma_flat_val - (b_fixed + alpha_val_num * b_alpha_coef)
    pres_eq = float(np.max(np.abs(eq_residual))) if eq_residual.size else 0.0
    rho_val = rho.value
    cone_viol_arr = (np.linalg.norm(rho_val[:, 1:], axis=1) - rho_val[:, 0])
    pres_cone = float(np.maximum(cone_viol_arr, 0.0).max()) if rho_val.size else 0.0
    pres = max(pres_eq, pres_cone)

    u_eq = constraints[0].dual_value
    if u_eq is not None:
        inner = float(np.dot(np.asarray(u_eq).ravel(), b_alpha_coef))
        dres = float(abs(abs(inner) - 1.0))
    else:
        dres = float("nan")

    try:
        cvxpy_primal = float(problem.value)
        gap = float(abs(alpha_val_num - cvxpy_primal))
    except Exception:
        gap = float("nan")
    if not np.isnan(pres) and not np.isnan(dres):
        gap = max(gap, max(pres, dres))

    return pres, dres, gap




def optimize_socp(
    mesh: LBMesh,
    yield_crit: YieldCriterion,
    *,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence[TractionBC] = (),
    solver: str = "MOSEK",
    solver_opts: dict | None = None,
    verbose: bool = False,
    feasibility_only: bool = False,
) -> LBSolution:
    (sigma, rho, alpha, A_eq, b_fixed, b_alpha_coef,
     soc_constraints, problem, constraints) = _build_problem(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        feasibility_only=feasibility_only,
    )

    solver_uc, opts, accept_unknown = _prepare_solver_opts(solver, solver_opts)

    used_solver, solve_time = _solve_with_fallback(
        problem, solver_uc, opts, accept_unknown, alpha, verbose,
    )

    status = _check_status(problem, alpha, sigma, accept_unknown)

    alpha_val_num, sigma_val, plastic = _extract_solution(
        alpha, sigma, soc_constraints, mesh.n_tri,
    )

    pres, dres, gap = _compute_residuals(
        A_eq, b_fixed, b_alpha_coef,
        alpha_val_num, sigma, rho, constraints, problem,
    )

    return LBSolution(
        alpha=alpha_val_num,
        sigma=sigma_val,
        lagrange_multiplier=plastic,
        status=status,
        solver_stats={
            "solver": used_solver,
            "n_constraints": int(A_eq.shape[0]) + len(soc_constraints),
            "solve_time": solve_time,
            "pres": pres,
            "dres": dres,
            "gap": gap,
            "exit_flag": status,
        },
    )




def boundary_conditions_from_config(bc_list: list) -> list:
    out: list = []
    for bc in bc_list:
        kw: dict = {"boundary": bc["boundary"]}
        if "fixed" in bc:
            kw["fixed"] = tuple(float(x) for x in bc["fixed"])
        if "scaled" in bc:
            kw["scaled"] = tuple(float(x) for x in bc["scaled"])
        if "constrain_x" in bc:
            kw["constrain_x"] = bool(bc["constrain_x"])
        if "constrain_y" in bc:
            kw["constrain_y"] = bool(bc["constrain_y"])
        out.append(TractionBC(**kw))
    return out


def body_force_for_mode(
    mode: str, *, uw: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    mode = mode.lower()
    if mode == "body_force_max":
        return (0.0, 0.0), (0.0, -1.0)
    if mode == "load_max":
        return (0.0, -uw), (0.0, 0.0)
    if mode == "none":
        return (0.0, -uw), (0.0, 0.0)
    raise ValueError(f"unknown mode {mode!r}; use load_max, body_force_max, or none")
