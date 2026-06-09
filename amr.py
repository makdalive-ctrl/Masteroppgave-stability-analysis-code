from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Callable, Sequence

import matplotlib.tri as mtri
import numpy as np

from cvxpy.error import SolverError
from optimizer import LBSolution, TractionBC, optimize_socp
from mesh import (
    LBMesh,
    mesh_parameter,
    gmsh_mesh_amr,
    gmsh_mesh_uniform,
    leb_mesh_uniform,
    _leb_bisect_marked,
)
from functions import YieldCriterion


_AMR_RESCALE_TOL = 0.05
_AMR_RESCALE_MAX_ITER = 4


# ─── (1) COMPUTE indicators from a solution ───────────────────────────────────

# --- Element-level utilities ---

def _element_areas(mesh: LBMesh) -> np.ndarray:
    p = mesh.nodes[mesh.elements]
    return 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )


def _element_centroids(mesh: LBMesh) -> np.ndarray:
    return mesh.nodes[mesh.elements].mean(axis=1)


def _element_gradient(mesh: LBMesh, nodal_field: np.ndarray) -> np.ndarray:
    p = mesh.nodes[mesh.elements]
    twoA_signed = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                   - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    b = np.stack([p[:, 1, 1] - p[:, 2, 1],
                  p[:, 2, 1] - p[:, 0, 1],
                  p[:, 0, 1] - p[:, 1, 1]], axis=1)
    c = np.stack([p[:, 2, 0] - p[:, 1, 0],
                  p[:, 0, 0] - p[:, 2, 0],
                  p[:, 1, 0] - p[:, 0, 0]], axis=1)
    gx = (b * nodal_field).sum(axis=1) / twoA_signed
    gy = (c * nodal_field).sum(axis=1) / twoA_signed
    return np.stack([gx, gy], axis=1)


def _geometric_size_bounds(
    mesh: LBMesh,
    *,
    floor_frac: float = 0.001,
    ceiling_frac: float = 0.10,
) -> tuple[float, float]:
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    L = float(min(x.max() - x.min(), y.max() - y.min()))
    return floor_frac * L, ceiling_frac * L


# --- Hessian recovery (SPR on the original continuous mesh) ---

def _element_to_nodal(mesh: LBMesh, elem_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_nodes = mesh.n_nodes
    tri = mesh.elements
    elem_values = np.asarray(elem_values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if elem_values.ndim == 1:
        nodal = np.zeros(n_nodes)
        wsum = np.zeros(n_nodes)
        for li in range(3):
            np.add.at(nodal, tri[:, li], weights * elem_values)
            np.add.at(wsum, tri[:, li], weights)
        return nodal / np.maximum(wsum, 1e-30)

    k = elem_values.shape[1]
    nodal = np.zeros((n_nodes, k))
    wsum = np.zeros(n_nodes)
    for li in range(3):
        for j in range(k):
            np.add.at(nodal[:, j], tri[:, li], weights * elem_values[:, j])
        np.add.at(wsum, tri[:, li], weights)
    return nodal / np.maximum(wsum, 1e-30)[:, None]


def _lambda_to_original_nodes(mesh: LBMesh, lambda_per_node: np.ndarray) -> np.ndarray:
    n_orig = mesh.nodes_orig.shape[0]
    lam_sum = np.zeros(n_orig)
    counts = np.zeros(n_orig)
    flat = np.asarray(lambda_per_node, dtype=float).reshape(-1)
    np.add.at(lam_sum, mesh.dup_to_orig, flat)
    np.add.at(counts, mesh.dup_to_orig, 1)
    return lam_sum / np.maximum(counts, 1.0)


def _spr_hessian_at_nodes(
    coords: np.ndarray,
    triangles: np.ndarray,
    nodal_field: np.ndarray,
    *,
    min_patch_size: int = 6,
    max_rings: int = 3,
) -> np.ndarray:
    n_nodes = coords.shape[0]
    n_tri = triangles.shape[0]
    H = np.zeros((n_nodes, 2, 2))

    patches: list[list[int]] = [[] for _ in range(n_nodes)]
    for e in range(n_tri):
        for li in range(3):
            patches[int(triangles[e, li])].append(e)

    neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
    for n in range(n_nodes):
        for e in patches[n]:
            for li in range(3):
                m = int(triangles[e, li])
                if m != n:
                    neighbors[n].add(m)

    for n in range(n_nodes):
        patch_nodes: set[int] = {n} | neighbors[n]

        rings = 1
        while len(patch_nodes) < min_patch_size and rings < max_rings:
            frontier: set[int] = set()
            for m in patch_nodes:
                frontier |= neighbors[m]
            if not (frontier - patch_nodes):
                break
            patch_nodes |= frontier
            rings += 1

        if len(patch_nodes) < min_patch_size:
            continue

        idx = np.fromiter(patch_nodes, dtype=np.int64)
        xy = coords[idx] - coords[n]
        f = nodal_field[idx]
        X = np.column_stack([
            np.ones(idx.size),
            xy[:, 0], xy[:, 1],
            xy[:, 0] ** 2, xy[:, 0] * xy[:, 1], xy[:, 1] ** 2,
        ])
        try:
            coeffs, *_ = np.linalg.lstsq(X, f, rcond=None)
        except np.linalg.LinAlgError:
            continue
        d, exy, g = coeffs[3], coeffs[4], coeffs[5]
        H[n, 0, 0] = 2.0 * d
        H[n, 1, 1] = 2.0 * g
        H[n, 0, 1] = exy
        H[n, 1, 0] = exy
    return H


def recovered_hessian(mesh: LBMesh, lambda_per_node: np.ndarray) -> np.ndarray:
    lam_orig = _lambda_to_original_nodes(mesh, lambda_per_node)
    H_orig = _spr_hessian_at_nodes(mesh.nodes_orig, mesh.elements_orig, lam_orig)
    H_elem = np.zeros((mesh.n_tri, 2, 2))
    for li in range(3):
        H_elem += H_orig[mesh.elements_orig[:, li]]
    H_elem /= 3.0
    return H_elem


def dominant_hessian_eigenvalue(H: np.ndarray) -> np.ndarray:
    a = H[:, 0, 0]
    b = H[:, 1, 1]
    c = H[:, 0, 1]
    half_tr = 0.5 * (a + b)
    disc = np.sqrt(np.maximum(0.25 * (a - b) ** 2 + c ** 2, 0.0))
    lam1 = half_tr + disc
    lam2 = half_tr - disc
    return np.maximum(np.abs(lam1), np.abs(lam2))


# --- Size-field schemes (3 / 4 / 5): indicator → per-element h ---

def L_value_based_refinement(
    mesh: LBMesh,
    lambda_per_node: np.ndarray,
    target_n: int,
    *,
    p: float = 2.0,
    eps_floor: float = 1e-3,
    h_min: float | None = None,
    h_max: float | None = None,
) -> np.ndarray:
    lam_T = np.maximum(np.abs(lambda_per_node).mean(axis=1), eps_floor * np.abs(lambda_per_node).max())
    areas = _element_areas(mesh)

    exp_size = -1.0 / (p + 1.0)
    exp_count = -2.0 * exp_size
    raw = float(np.sum(areas * lam_T**exp_count))
    K = np.sqrt((4.0 / (np.sqrt(3.0) * target_n)) * raw)
    h_new = K * lam_T**exp_size

    if h_min is None or h_max is None:
        gmin, gmax = _geometric_size_bounds(mesh, floor_frac=0.0025, ceiling_frac=0.10)
        if h_min is None:
            h_min = gmin
        if h_max is None:
            h_max = gmax
    return np.clip(h_new, h_min, h_max)


def L_gradient_based_refinement(
    mesh: LBMesh,
    lambda_per_node: np.ndarray,
    target_n: int,
    *,
    eps_floor: float = 0.10,
    h_min: float | None = None,
    h_max: float | None = None,
) -> np.ndarray:
    grad = _element_gradient(mesh, lambda_per_node)
    grad_norm = np.linalg.norm(grad, axis=1)
    grad_norm = np.maximum(grad_norm, eps_floor * float(grad_norm.max()))

    areas = _element_areas(mesh)
    raw = float(np.sum(areas * grad_norm**2))
    alpha = np.sqrt((4.0 / (np.sqrt(3.0) * target_n)) * raw)
    h_new = alpha / grad_norm

    if h_min is None or h_max is None:
        gmin, gmax = _geometric_size_bounds(mesh, floor_frac=0.0025, ceiling_frac=0.10)
        if h_min is None:
            h_min = gmin
        if h_max is None:
            h_max = gmax
    return np.clip(h_new, h_min, h_max)


def L_hessian_based_refinement(
    mesh: LBMesh,
    lambda_per_node: np.ndarray,
    target_n: int,
    *,
    p: float = 2.0,
    eps_floor: float = 0.02,
    h_min: float | None = None,
    h_max: float | None = None,
) -> np.ndarray:
    H = recovered_hessian(mesh, lambda_per_node)
    rho = dominant_hessian_eigenvalue(H)

    rho = np.maximum(rho, eps_floor * float(rho.max()))

    areas = _element_areas(mesh)
    exp_size = -p / (2.0 * (p + 1.0))
    raw = float(np.sum(areas * rho ** (-2.0 * exp_size)))
    K = np.sqrt((4.0 / (np.sqrt(3.0) * target_n)) * raw)
    h_new = K * rho ** exp_size

    if h_min is None or h_max is None:
        gmin, gmax = _geometric_size_bounds(mesh, floor_frac=0.001, ceiling_frac=0.10)
        if h_min is None:
            h_min = gmin
        if h_max is None:
            h_max = gmax
    return np.clip(h_new, h_min, h_max)


# --- Refinement dispatcher (used by LEB path) ---

_METHOD_FNS_PLAIN = {
    "L value": L_value_based_refinement,
    "L gradient": L_gradient_based_refinement,
    "L hessian": L_hessian_based_refinement,
}


def control_variable(
    method: str, mesh: LBMesh, lambda_per_node: np.ndarray,
) -> np.ndarray:
    if method == "L value":
        return np.abs(lambda_per_node).mean(axis=1)
    if method == "L gradient":
        return np.linalg.norm(_element_gradient(mesh, lambda_per_node), axis=1)
    if method == "L hessian":
        H = recovered_hessian(mesh, lambda_per_node)
        return dominant_hessian_eigenvalue(H)
    raise ValueError(
        f"unknown method {method!r}; use 'L value', 'L gradient', or 'L hessian'"
    )


# --- LEB marking (Dörfler bulk-marking) ---

_MARK_QUANTUM = 1.0e-9


def dorfler_mark(contrib: np.ndarray, theta: float = 0.5) -> np.ndarray:
    contrib = np.asarray(contrib, dtype=float)
    if contrib.ndim != 1:
        raise ValueError("contrib must be 1-D (one weight per element)")
    total = float(contrib.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.empty(0, dtype=np.int64)
    theta = float(np.clip(theta, 1e-6, 1.0))

    scale = float(np.max(contrib))
    if scale > 0.0:
        q = _MARK_QUANTUM * scale
        contrib = np.round(contrib / q) * q
    total = float(contrib.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.empty(0, dtype=np.int64)

    n = contrib.size
    idx = np.arange(n, dtype=np.int64)
    order = np.lexsort((idx, -contrib))
    cum = np.cumsum(contrib[order])
    k = int(np.searchsorted(cum, theta * total, side="left")) + 1
    k = min(k, order.size)
    return np.sort(order[:k]).astype(np.int64)


def leb_mesh_amr(
    mesh: LBMesh, indicator: np.ndarray, theta: float,
) -> LBMesh | None:
    marked = dorfler_mark(indicator, theta)
    if marked.size == 0:
        return None
    return _leb_bisect_marked(mesh, marked)


# --- Diagnostic: η_T, η, μ ---

def error_indicator(
    mesh: LBMesh,
    lambda_per_node: np.ndarray,
    *,
    p: float = 2.0,
) -> tuple[np.ndarray, float, float]:
    H = recovered_hessian(mesh, lambda_per_node)
    rho = dominant_hessian_eigenvalue(H)

    areas = _element_areas(mesh)
    h_T = np.sqrt(4.0 * areas / np.sqrt(3.0))

    eta_T = 2.0 * areas ** (1.0 / p) * rho * h_T ** 2
    eta = float(np.sum(eta_T ** p) ** (1.0 / p))

    lam_T = np.abs(lambda_per_node).mean(axis=1)
    lam_norm_p = float(np.sum(areas * lam_T ** p))
    denom = (lam_norm_p + eta ** p) ** (1.0 / p)
    mu = eta / denom if denom > 0 else 0.0

    return eta_T, eta, mu


# ─── (2) GENERATE new mesh from indicator ─────────────────────────────────────

def _smooth_size_field_gradation(
    mesh: LBMesh,
    h_per_element: np.ndarray,
    *,
    grad_ratio: float = 1.4,
    max_iter: int = 50,
) -> np.ndarray:
    h = np.asarray(h_per_element, dtype=float).copy()
    if grad_ratio <= 1.0:
        return h
    valid = np.isfinite(h)

    tri = mesh.elements_orig
    edge_to_elems: dict[tuple[int, int], list[int]] = {}
    for e in range(tri.shape[0]):
        for li in range(3):
            a = int(tri[e, (li + 1) % 3])
            b = int(tri[e, (li + 2) % 3])
            key = (a, b) if a < b else (b, a)
            edge_to_elems.setdefault(key, []).append(e)

    pairs_list: list[tuple[int, int]] = []
    for es in edge_to_elems.values():
        if len(es) == 2 and valid[es[0]] and valid[es[1]]:
            pairs_list.append((es[0], es[1]))
    if not pairs_list:
        return h

    pairs = np.asarray(pairs_list, dtype=np.int64)
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    for _ in range(max_iter):
        h_old = h.copy()
        np.minimum.at(h, j_idx, grad_ratio * h[i_idx])
        np.minimum.at(h, i_idx, grad_ratio * h[j_idx])
        delta = h_old[valid] - h[valid]
        scale = np.maximum(h_old[valid], 1e-30)
        if float(np.max(delta / scale)) < 1e-6:
            break

    return h


def make_size_callback(mesh: LBMesh, h_per_element: np.ndarray, fallback: float) -> Callable[[float, float], float]:
    triangulation = mtri.Triangulation(
        mesh.nodes_orig[:, 0], mesh.nodes_orig[:, 1], mesh.elements_orig
    )
    finder = triangulation.get_trifinder()
    sizes = np.asarray(h_per_element, dtype=float)

    def cb(x: float, y: float) -> float:
        e = int(finder(x, y))
        if e < 0:
            return fallback
        return float(sizes[e])

    return cb


# ─── (3) AMR drivers ──────────────────────────────────────────────────────────

# --- Shared dataclass + type alias ---

@dataclass
class AdaptiveStep:
    iteration: int
    mesh: LBMesh
    solution: LBSolution
    eta: float = 0.0
    mu: float = 0.0
    eta_T: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_angular_R: int | None = None
    n_angular_L: int | None = None
    fan_h_min: float | None = None


SizeFieldFn = Callable[[LBMesh, np.ndarray, int], np.ndarray]


# --- Main driver ---

def run_loop(
    seed_mesh: LBMesh,
    geometry_builder: Callable[[], dict[int, str]],
    yield_crit: YieldCriterion,
    *,
    method: str = "L value",
    target_n: int,
    n_iterations: int = 4,
    growth_factor: float = 1.0,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence[TractionBC] = (),
    solver: str = "MOSEK",
    grad_ratio: float = 1.4,
    on_iter: Callable[["AdaptiveStep"], None] | None = None,
    feasibility_only: bool = False,
    nested: bool = False,
    dorfler_theta: float = 0.5,
    uniform: bool = False,
) -> list[AdaptiveStep]:
    method_fn = _METHOD_FNS_PLAIN.get(method)
    if method_fn is None:
        raise ValueError(
            f"unknown method {method!r}; use 'L value', 'L gradient', or 'L hessian'"
        )

    history: list[AdaptiveStep] = []
    current_target = int(target_n)

    mesh = seed_mesh
    sol = optimize_socp(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        solver=solver,
        feasibility_only=feasibility_only,
    )
    eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
    step = AdaptiveStep(0, mesh, sol, eta=eta, mu=mu, eta_T=eta_T)
    history.append(step)
    if on_iter is not None:
        on_iter(step)

    for it in range(1, n_iterations + 1):
        gc.collect()

        if nested and uniform:
            try:
                best_mesh: LBMesh | None = leb_mesh_uniform(mesh, growth_factor)
            except (RuntimeError, MemoryError) as e:
                print(f"[AMR iter {it}] uniform refine aborted: {e}; "
                      f"returning {len(history)} iterations.")
                break
        elif nested:
            indic = control_variable(method, mesh, sol.lagrange_multiplier)
            try:
                best_mesh = leb_mesh_amr(
                    mesh, _element_areas(mesh) * indic, dorfler_theta,
                )
            except (RuntimeError, MemoryError) as e:
                print(f"[AMR iter {it}] nested refine aborted: {e}; "
                      f"returning {len(history)} iterations.")
                break
            if best_mesh is None:
                print(f"[AMR iter {it}] nested: indicator carries no weight, "
                      f"nothing to refine; returning {len(history)} iterations.")
                break
        elif uniform:
            current_target = int(round(current_target * growth_factor))
            try:
                best_mesh = gmsh_mesh_uniform(geometry_builder, current_target)
            except (RuntimeError, MemoryError) as e:
                print(f"[AMR iter {it}] gmsh uniform refine aborted: {e}; "
                      f"returning {len(history)} iterations.")
                break
        else:
            current_target = int(round(current_target * growth_factor))
            h_new = method_fn(mesh, sol.lagrange_multiplier, current_target)
            h_new = _smooth_size_field_gradation(mesh, h_new, grad_ratio=grad_ratio)

            scale = 1.0
            best_mesh = None
            best_err = float("inf")
            build_err: Exception | None = None
            for _rescale_it in range(_AMR_RESCALE_MAX_ITER):
                h_scaled = h_new * scale
                cb = make_size_callback(mesh, h_scaled, fallback=float(np.median(h_scaled)))
                try:
                    cand_mesh = gmsh_mesh_amr(geometry_builder, cb)
                except (RuntimeError, MemoryError) as e:
                    build_err = e
                    break
                rel_err = abs(cand_mesh.n_tri - current_target) / current_target
                if rel_err < best_err:
                    best_mesh, best_err = cand_mesh, rel_err
                if rel_err < _AMR_RESCALE_TOL:
                    break
                scale *= float(np.sqrt(cand_mesh.n_tri / current_target))

            if best_mesh is None:
                print(f"[AMR iter {it}] mesh build aborted: {build_err}; "
                      f"returning {len(history)} iterations.")
                break

        try:
            sol_next = optimize_socp(
                best_mesh, yield_crit,
                fixed_body_force=fixed_body_force,
                scaled_body_force=scaled_body_force,
                boundary_conditions=boundary_conditions,
                solver=solver,
                feasibility_only=feasibility_only,
            )
        except (RuntimeError, MemoryError, SolverError) as e:
            print(f"[AMR iter {it}] solver aborted: {e}; returning {len(history)} iterations.")
            break

        mesh, sol = best_mesh, sol_next
        eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
        step = AdaptiveStep(it, mesh, sol, eta=eta, mu=mu, eta_T=eta_T)
        history.append(step)
        if on_iter is not None:
            on_iter(step)

    return history


# --- Fan-specific drivers ---

def run_fan_only_loop(
    seed_mesh: LBMesh,
    cfg: dict,
    yield_crit: YieldCriterion,
    *,
    target_n: int,
    n_iterations: int = 3,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence[TractionBC] = (),
    solver: str = "MOSEK",
    on_iter: Callable[["AdaptiveStep"], None] | None = None,
) -> list[AdaptiveStep]:
    if not cfg.get("adaptive_fan", False):
        raise ValueError(
            "run_fan_only_loop requires cfg['adaptive_fan'] = True; for a "
            "uniform-spacing fan, call mesh_fan + optimize_socp directly."
        )
    if cfg.get("anisotropy", False):
        raise ValueError(
            "run_fan_only_loop does not support anisotropy: the anisotropy "
            "scheme rebuilds the outer mesh under a metric tensor each "
            "iteration, which contradicts the 'fixed uniform outer mesh' "
            "premise of this driver."
        )

    from fan import (
        mesh_fan,
        circumferential_stress_gradient_norm,
        fan_centers_from_config,
        matched_fan_h_min,
        maybe_grow_N,
        theta_array_from_indicator,
    )

    geo = cfg["geometry"]
    fan_radius = float(cfg["fan_radius"])
    N = int(cfg["N_angular"])
    mirrored = bool(geo.get("mirrored", False))
    half_sym = not mirrored
    fan_centers_list = fan_centers_from_config(geo)

    grow_N = bool(cfg.get("adaptive_fan_grow_N", False))
    max_N = int(cfg.get("adaptive_fan_max_N", 64))
    conc_threshold = float(cfg.get("adaptive_fan_conc_threshold", 2.0))
    growth_per_iter = float(cfg.get("adaptive_fan_growth_per_iter", 1.5))
    match_h_min = bool(cfg.get("adaptive_fan_match_h_min", False))

    current_N_R = N
    current_N_L = None if half_sym else N
    fan_h_min_iter: float | None = None

    mesh = seed_mesh
    sol = optimize_socp(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        solver=solver,
    )
    eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
    step = AdaptiveStep(
        0, mesh, sol, eta=eta, mu=mu, eta_T=eta_T,
        n_angular_R=current_N_R,
        n_angular_L=current_N_L,
        fan_h_min=fan_h_min_iter,
    )
    history: list[AdaptiveStep] = [step]
    if on_iter is not None:
        on_iter(step)

    for it in range(1, n_iterations + 1):
        gc.collect()
        _, rate_R, diag_R = circumferential_stress_gradient_norm(
            mesh, sol.sigma, fan_centers_list[-1], fan_radius,
            n_bins=max(16, current_N_R), half_disk=True,
        )
        new_N_R = maybe_grow_N(
            current_N_R, diag_R["concentration"],
            grow=grow_N, threshold=conc_threshold,
            growth_per_iter=growth_per_iter, max_N=max_N,
        )
        theta_R = theta_array_from_indicator(rate_R, new_N_R)
        current_N_R = new_N_R

        theta_L = None
        if not half_sym:
            _, rate_L, diag_L = circumferential_stress_gradient_norm(
                mesh, sol.sigma, fan_centers_list[0], fan_radius,
                n_bins=max(16, current_N_L), half_disk=True,
            )
            new_N_L = maybe_grow_N(
                current_N_L, diag_L["concentration"],
                grow=grow_N, threshold=conc_threshold,
                growth_per_iter=growth_per_iter, max_N=max_N,
            )
            theta_L = theta_array_from_indicator(rate_L, new_N_L)
            current_N_L = new_N_L

        if match_h_min:
            fan_h_min_iter = matched_fan_h_min(fan_radius, theta_R, theta_L)

        try:
            mesh_next = mesh_fan(
                cfg, target_n=target_n,
                theta_R=theta_R, theta_L=theta_L,
                outer_size_callback=None,
                fan_h_min=fan_h_min_iter,
                n_angular_R=current_N_R,
                n_angular_L=current_N_L,
            )
            sol_next = optimize_socp(
                mesh_next, yield_crit,
                fixed_body_force=fixed_body_force,
                scaled_body_force=scaled_body_force,
                boundary_conditions=boundary_conditions,
                solver=solver,
            )
        except (RuntimeError, MemoryError) as e:
            print(f"[fan-only iter {it}] aborted: {e}; returning {len(history)} iterations.")
            break
        mesh, sol = mesh_next, sol_next
        eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
        step = AdaptiveStep(
            it, mesh, sol, eta=eta, mu=mu, eta_T=eta_T,
            n_angular_R=current_N_R,
            n_angular_L=current_N_L,
            fan_h_min=fan_h_min_iter,
        )
        history.append(step)
        if on_iter is not None:
            on_iter(step)

    return history


def run_fan_amr_loop(
    seed_mesh: LBMesh,
    cfg: dict,
    yield_crit: YieldCriterion,
    *,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence[TractionBC] = (),
    solver: str = "MOSEK",
    on_iter: Callable[["AdaptiveStep"], None] | None = None,
) -> list[AdaptiveStep]:
    from fan import (
        mesh_fan,
        circumferential_stress_gradient_norm,
        fan_centers_from_config,
        h_far_from_target,
        L_based_refinement_outer,
        make_outer_size_callback,
        matched_fan_h_min,
        maybe_grow_N,
        theta_array_from_indicator,
    )

    geo = cfg["geometry"]
    fan_radius = float(cfg["fan_radius"])
    M = int(cfg["M_radial"])
    N = int(cfg["N_angular"])
    mirrored = bool(geo.get("mirrored", False))
    half_sym = not mirrored
    n_fans = 1 if half_sym else 2
    fan_centers_list = fan_centers_from_config(geo)

    method = cfg.get("method", "L value")
    if method not in ("L value", "L gradient", "L hessian"):
        raise ValueError(
            f"unknown method {method!r}; use 'L value', 'L gradient', or 'L hessian'"
        )

    is_adaptive_fan = bool(cfg.get("adaptive_fan", False))
    is_anisotropy = bool(cfg.get("anisotropy", False))
    aniso_ratio_max = float(cfg.get("aniso_ratio_max", 5.0))
    target_n = int(cfg["target_n"])
    growth = float(cfg.get("growth_factor", 1.5))
    n_iter = int(cfg.get("n_iterations", 4))

    grow_N = bool(cfg.get("adaptive_fan_grow_N", False))
    max_N = int(cfg.get("adaptive_fan_max_N", 64))
    conc_threshold = float(cfg.get("adaptive_fan_conc_threshold", 2.0))
    growth_per_iter = float(cfg.get("adaptive_fan_growth_per_iter", 1.5))
    match_h_min = bool(cfg.get("adaptive_fan_match_h_min", False))
    grad_ratio = float(cfg.get("grad_ratio", 1.4))

    current_N_R = N
    current_N_L = None if half_sym else N
    fan_h_min_iter: float | None = None

    mesh = seed_mesh
    sol = optimize_socp(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        solver=solver,
    )
    eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
    step = AdaptiveStep(
        0, mesh, sol, eta=eta, mu=mu, eta_T=eta_T,
        n_angular_R=current_N_R,
        n_angular_L=current_N_L,
        fan_h_min=fan_h_min_iter,
    )
    history: list[AdaptiveStep] = [step]
    if on_iter is not None:
        on_iter(step)

    current_target = target_n
    for it in range(1, n_iter + 1):
        gc.collect()
        current_target = int(round(current_target * growth))
        n_fan_total_val = (
            current_N_R * (2 * M - 1)
            + (0 if half_sym else current_N_L * (2 * M - 1))
        )
        n_outer_target = max(50, current_target - n_fan_total_val)
        h_far_iter = h_far_from_target(
            geo, fan_radius, n_fans, n_fan_total_val, current_target
        )

        cb = None
        outer_metric_pos: str | None = None
        if is_anisotropy:
            import anisotropy as _aniso
            outer_metric_pos = _aniso.write_outer_metric_pos(
                mesh, sol.lagrange_multiplier, n_outer_target,
                aniso_ratio_max=aniso_ratio_max,
            )
        else:
            h_per_elem, _ = L_based_refinement_outer(
                method, mesh, sol.lagrange_multiplier, n_outer_target,
                fan_centers_list, fan_radius,
            )
            h_per_elem = _smooth_size_field_gradation(
                mesh, h_per_elem, grad_ratio=grad_ratio,
            )
            cb = make_outer_size_callback(mesh, h_per_elem, fallback=h_far_iter)

        theta_R = None
        theta_L = None
        if is_adaptive_fan:
            _, rate_R, diag_R = circumferential_stress_gradient_norm(
                mesh, sol.sigma, fan_centers_list[-1], fan_radius,
                n_bins=max(16, current_N_R), half_disk=True,
            )
            new_N_R = maybe_grow_N(
                current_N_R, diag_R["concentration"],
                grow=grow_N, threshold=conc_threshold,
                growth_per_iter=growth_per_iter, max_N=max_N,
            )
            theta_R = theta_array_from_indicator(rate_R, new_N_R)
            current_N_R = new_N_R

            if not half_sym:
                _, rate_L, diag_L = circumferential_stress_gradient_norm(
                    mesh, sol.sigma, fan_centers_list[0], fan_radius,
                    n_bins=max(16, current_N_L), half_disk=True,
                )
                new_N_L = maybe_grow_N(
                    current_N_L, diag_L["concentration"],
                    grow=grow_N, threshold=conc_threshold,
                    growth_per_iter=growth_per_iter, max_N=max_N,
                )
                theta_L = theta_array_from_indicator(rate_L, new_N_L)
                current_N_L = new_N_L

            if match_h_min:
                fan_h_min_iter = matched_fan_h_min(fan_radius, theta_R, theta_L)

        try:
            try:
                mesh_next = mesh_fan(
                    cfg, target_n=current_target,
                    theta_R=theta_R, theta_L=theta_L,
                    outer_size_callback=cb,
                    outer_metric_pos_path=outer_metric_pos,
                    fan_h_min=fan_h_min_iter,
                    n_angular_R=current_N_R,
                    n_angular_L=current_N_L,
                )
            finally:
                if outer_metric_pos is not None:
                    import os as _os
                    try:
                        _os.unlink(outer_metric_pos)
                    except OSError:
                        pass
            sol_next = optimize_socp(
                mesh_next, yield_crit,
                fixed_body_force=fixed_body_force,
                scaled_body_force=scaled_body_force,
                boundary_conditions=boundary_conditions,
                solver=solver,
            )
        except (RuntimeError, MemoryError) as e:
            print(f"[fan-AMR iter {it}] aborted: {e}; returning {len(history)} iterations.")
            break
        mesh, sol = mesh_next, sol_next
        eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
        step = AdaptiveStep(
            it, mesh, sol, eta=eta, mu=mu, eta_T=eta_T,
            n_angular_R=current_N_R,
            n_angular_L=current_N_L,
            fan_h_min=fan_h_min_iter,
        )
        history.append(step)
        if on_iter is not None:
            on_iter(step)

    return history


def run_fan_nested_loop(
    seed_mesh: LBMesh,
    cfg: dict,
    yield_crit: YieldCriterion,
    *,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence[TractionBC] = (),
    solver: str = "MOSEK",
    on_iter: Callable[["AdaptiveStep"], None] | None = None,
) -> list[AdaptiveStep]:
    geo = cfg["geometry"]
    mirrored = bool(geo.get("mirrored", False))
    half_sym = not mirrored
    N = int(cfg["N_angular"])
    n_angular_R = N
    n_angular_L = None if half_sym else N

    method = cfg.get("method", "L value")
    target_n = int(cfg["target_n"])
    n_iter = int(cfg.get("n_iterations", 4))
    dorfler_theta = float(cfg.get("dorfler_theta", 0.5))
    uniform = bool(cfg.get("uniform", False))
    growth_factor = float(cfg.get("growth_factor", 2.0))

    mesh = seed_mesh
    sol = optimize_socp(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        solver=solver,
    )
    eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
    step = AdaptiveStep(
        0, mesh, sol, eta=eta, mu=mu, eta_T=eta_T,
        n_angular_R=n_angular_R, n_angular_L=n_angular_L,
    )
    history: list[AdaptiveStep] = [step]
    if on_iter is not None:
        on_iter(step)

    for it in range(1, n_iter + 1):
        gc.collect()
        try:
            if uniform:
                mesh_next = leb_mesh_uniform(mesh, growth_factor)
            else:
                indic = control_variable(
                    method, mesh, sol.lagrange_multiplier)
                mesh_next = leb_mesh_amr(
                    mesh, _element_areas(mesh) * indic, dorfler_theta,
                )
                if mesh_next is None:
                    print(f"[fan-nested iter {it}] indicator carries no "
                          f"weight, nothing to refine; returning "
                          f"{len(history)} iterations.")
                    break
            sol_next = optimize_socp(
                mesh_next, yield_crit,
                fixed_body_force=fixed_body_force,
                scaled_body_force=scaled_body_force,
                boundary_conditions=boundary_conditions,
                solver=solver,
            )
        except (RuntimeError, MemoryError) as e:
            print(f"[fan-nested iter {it}] aborted: {e}; "
                  f"returning {len(history)} iterations.")
            break
        mesh, sol = mesh_next, sol_next
        eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
        step = AdaptiveStep(
            it, mesh, sol, eta=eta, mu=mu, eta_T=eta_T,
            n_angular_R=n_angular_R, n_angular_L=n_angular_L,
        )
        history.append(step)
        if on_iter is not None:
            on_iter(step)

    return history


# --- Legacy entry points (subsumed by run_loop; kept for back-compat) ---

def _adaptive_loop(
    geometry_builder: Callable[[], dict[int, str]],
    yield_crit: YieldCriterion,
    size_field_fn: SizeFieldFn,
    *,
    method_label: str,
    target_n: int,
    n_iterations: int = 4,
    growth_factor: float = 1.0,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: tuple[TractionBC, ...] = (),
    solver: str = "MOSEK",
    grad_ratio: float = 1.4,
    verbose: bool = False,
) -> list[AdaptiveStep]:
    history: list[AdaptiveStep] = []
    current_target = int(target_n)

    mesh = mesh_parameter(geometry_builder, target_n=current_target, verbose=verbose)
    sol = optimize_socp(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        solver=solver,
    )
    eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
    history.append(AdaptiveStep(0, mesh, sol, eta=eta, mu=mu, eta_T=eta_T))
    if verbose:
        print(f"[{method_label} iter 0] uniform: target NE = {current_target}, "
              f"got NE = {mesh.n_tri}, alpha = {sol.alpha:.5f}, mu = {mu:.3e}")

    for it in range(1, n_iterations + 1):
        gc.collect()
        current_target = int(round(current_target * growth_factor))
        h_new = size_field_fn(mesh, sol.lagrange_multiplier, current_target)
        h_new = _smooth_size_field_gradation(mesh, h_new, grad_ratio=grad_ratio)

        scale = 1.0
        best_mesh: LBMesh | None = None
        best_err = float("inf")
        build_err: Exception | None = None
        for _rescale_it in range(_AMR_RESCALE_MAX_ITER):
            h_scaled = h_new * scale
            fallback_h = float(np.median(h_scaled))
            cb = make_size_callback(mesh, h_scaled, fallback=fallback_h)
            try:
                cand_mesh = gmsh_mesh_amr(geometry_builder, cb)
            except (RuntimeError, MemoryError) as e:
                build_err = e
                break
            rel_err = abs(cand_mesh.n_tri - current_target) / current_target
            if rel_err < best_err:
                best_mesh, best_err = cand_mesh, rel_err
            if rel_err < _AMR_RESCALE_TOL:
                break
            scale *= float(np.sqrt(cand_mesh.n_tri / current_target))

        if best_mesh is None:
            print(f"[{method_label} iter {it}] mesh build aborted: {build_err}; "
                  f"returning {len(history)} iterations.")
            break

        try:
            sol_next = optimize_socp(
                best_mesh, yield_crit,
                fixed_body_force=fixed_body_force,
                scaled_body_force=scaled_body_force,
                boundary_conditions=boundary_conditions,
                solver=solver,
            )
        except (RuntimeError, MemoryError) as e:
            print(f"[{method_label} iter {it}] solver aborted: {e}; "
                  f"returning {len(history)} iterations.")
            break

        mesh, sol = best_mesh, sol_next
        eta_T, eta, mu = error_indicator(mesh, sol.lagrange_multiplier)
        history.append(AdaptiveStep(it, mesh, sol, eta=eta, mu=mu, eta_T=eta_T))
        if verbose:
            print(f"[{method_label} iter {it}] target NE = {current_target}, "
                  f"got NE = {mesh.n_tri}, alpha = {sol.alpha:.5f}, mu = {mu:.3e}")

    return history


def adaptive_value_based(
    geometry_builder: Callable[[], dict[int, str]],
    yield_crit: YieldCriterion,
    *,
    target_n: int,
    n_iterations: int = 4,
    growth_factor: float = 1.0,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: tuple[TractionBC, ...] = (),
    solver: str = "MOSEK",
    p: float = 2.0,
    verbose: bool = False,
) -> list[AdaptiveStep]:
    def size_fn(m, lam, n):
        return L_value_based_refinement(m, lam, n, p=p)
    return _adaptive_loop(
        geometry_builder, yield_crit, size_fn,
        method_label="L-value",
        target_n=target_n, n_iterations=n_iterations, growth_factor=growth_factor,
        fixed_body_force=fixed_body_force, scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions, solver=solver, verbose=verbose,
    )


def adaptive_gradient_based(
    geometry_builder: Callable[[], dict[int, str]],
    yield_crit: YieldCriterion,
    *,
    target_n: int,
    n_iterations: int = 4,
    growth_factor: float = 1.0,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: tuple[TractionBC, ...] = (),
    solver: str = "MOSEK",
    verbose: bool = False,
) -> list[AdaptiveStep]:
    return _adaptive_loop(
        geometry_builder, yield_crit, L_gradient_based_refinement,
        method_label="L-grad",
        target_n=target_n, n_iterations=n_iterations, growth_factor=growth_factor,
        fixed_body_force=fixed_body_force, scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions, solver=solver, verbose=verbose,
    )


def adaptive_hessian_based(
    geometry_builder: Callable[[], dict[int, str]],
    yield_crit: YieldCriterion,
    *,
    target_n: int,
    n_iterations: int = 4,
    growth_factor: float = 1.0,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: tuple[TractionBC, ...] = (),
    solver: str = "MOSEK",
    p: float = 2.0,
    verbose: bool = False,
) -> list[AdaptiveStep]:
    def size_fn(m, lam, n):
        return L_hessian_based_refinement(m, lam, n, p=p)
    return _adaptive_loop(
        geometry_builder, yield_crit, size_fn,
        method_label="L-hess",
        target_n=target_n, n_iterations=n_iterations, growth_factor=growth_factor,
        fixed_body_force=fixed_body_force, scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions, solver=solver, verbose=verbose,
    )
