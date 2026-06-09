from __future__ import annotations
import os
import tempfile
from typing import Callable, Sequence
import gmsh
import numpy as np

import amr
import mesh as mesh_lib
import optimizer
import functions


# ─── (1) COMPUTE anisotropic metric tensor from solution ──────────────────────

def _element_eigendecomp_hessian(
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H_sym = 0.5 * (H + H.swapaxes(1, 2))
    w, V = np.linalg.eigh(H_sym)
    abs_w = np.abs(w)
    order = np.argsort(abs_w, axis=1)
    abs_w_sorted = np.take_along_axis(abs_w, order, axis=1)
    V_sorted = np.take_along_axis(V, order[:, None, :], axis=2)
    return abs_w_sorted[:, 0], abs_w_sorted[:, 1], V_sorted


def anisotropic_metric_per_element(
    mesh: mesh_lib.LBMesh,
    lambda_per_node: np.ndarray,
    target_n: int,
    *,
    p: float = 2.0,
    eps_floor: float = 1.0e-8,
    h_min_factor: float = 0.005,
    h_max_factor: float = 3.0,
    aniso_ratio_max: float = 5.0,
) -> np.ndarray:
    H = amr.recovered_hessian(mesh, lambda_per_node)
    lam_min, lam_max, R = _element_eigendecomp_hessian(H)

    lam_max_ref = float(lam_max.max()) if lam_max.size else 1.0
    floor = eps_floor * lam_max_ref if lam_max_ref > 0 else eps_floor
    lam_min_f = np.maximum(lam_min, floor)
    lam_max_f = np.maximum(lam_max, floor)

    areas = amr._element_areas(mesh)
    exp_size = -p / (2.0 * (p + 1.0))
    raw = float(np.sum(areas * lam_max_f ** (-2.0 * exp_size)))
    K = np.sqrt((4.0 / (np.sqrt(3.0) * target_n)) * raw)
    h_iso = K * lam_max_f ** exp_size

    h_typical = float(np.sqrt(areas.mean()))
    h_iso = np.clip(h_iso, h_min_factor * h_typical, h_max_factor * h_typical)

    r = np.sqrt(lam_max_f / lam_min_f)
    r = np.clip(r, 1.0, float(aniso_ratio_max))

    h_long = h_iso * np.sqrt(r)
    h_short = h_iso / np.sqrt(r)

    inv_long_sq = 1.0 / (h_long ** 2)
    inv_short_sq = 1.0 / (h_short ** 2)

    e_long = R[:, :, 0]
    e_short = R[:, :, 1]

    n_tri = mesh.n_tri
    M = np.empty((n_tri, 2, 2))
    M[:, 0, 0] = inv_long_sq * e_long[:, 0] ** 2 + inv_short_sq * e_short[:, 0] ** 2
    M[:, 1, 1] = inv_long_sq * e_long[:, 1] ** 2 + inv_short_sq * e_short[:, 1] ** 2
    M[:, 0, 1] = (inv_long_sq * e_long[:, 0] * e_long[:, 1]
                  + inv_short_sq * e_short[:, 0] * e_short[:, 1])
    M[:, 1, 0] = M[:, 0, 1]
    return M


def _metric_per_orig_node(mesh: mesh_lib.LBMesh, M_per_elem: np.ndarray) -> np.ndarray:
    areas = amr._element_areas(mesh)
    n_orig = mesh.nodes_orig.shape[0]
    M_node = np.zeros((n_orig, 2, 2))
    w_node = np.zeros(n_orig)
    for li in range(3):
        idx = mesh.elements_orig[:, li]
        for i in range(2):
            for j in range(2):
                np.add.at(M_node[:, i, j], idx, areas * M_per_elem[:, i, j])
        np.add.at(w_node, idx, areas)
    M_node /= np.maximum(w_node, 1e-30)[:, None, None]
    return M_node


# ─── (2) WRITE metric to gmsh .pos ────────────────────────────────────────────

def _write_metric_pos(
    coords: np.ndarray,
    triangles: np.ndarray,
    M_per_node: np.ndarray,
    path: str,
    name: str = "metric",
) -> None:
    lines: list[str] = [f'View "{name}" {{']
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        xa, ya = float(coords[a, 0]), float(coords[a, 1])
        xb, yb = float(coords[b, 0]), float(coords[b, 1])
        xc, yc = float(coords[c, 0]), float(coords[c, 1])
        coord_str = (
            f"{xa},{ya},0,"
            f"{xb},{yb},0,"
            f"{xc},{yc},0"
        )
        vals: list[float] = []
        for n in (a, b, c):
            M = M_per_node[n]
            vals.extend([
                M[0, 0], M[0, 1], 0.0,
                M[1, 0], M[1, 1], 0.0,
                0.0,     0.0,     1.0,
            ])
        val_str = ",".join(f"{v:.10g}" for v in vals)
        lines.append(f"  TT({coord_str}){{{val_str}}};")
    lines.append("};")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Bridge for fan AMR (scheme 8): compute metric + write to temp .pos, return path.
def write_outer_metric_pos(
    mesh: mesh_lib.LBMesh,
    lambda_per_node: np.ndarray,
    target_n: int,
    *,
    aniso_ratio_max: float = 5.0,
) -> str:
    M_elem = anisotropic_metric_per_element(
        mesh, lambda_per_node, target_n,
        aniso_ratio_max=aniso_ratio_max,
    )
    M_node = _metric_per_orig_node(mesh, M_elem)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pos", delete=False, encoding="utf-8",
    ) as tf:
        pos_path = tf.name
    _write_metric_pos(
        mesh.nodes_orig,
        mesh.elements_orig,
        M_node,
        pos_path,
    )
    return pos_path


# ─── (3) REMESH with BAMG under the metric ────────────────────────────────────

def mesh_anisotropy(
    geometry_builder: Callable[[], dict[int, str]],
    prev_mesh: mesh_lib.LBMesh,
    M_per_orig_node: np.ndarray,
    *,
    name: str = "lb_aniso",
    quiet: bool = True,
    h_floor: float = 1.0e-4,
    h_ceiling: float = 1.0,
    aniso_max: float = 100.0,
) -> mesh_lib.LBMesh:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pos", delete=False, encoding="utf-8",
    ) as tf:
        pos_path = tf.name
    try:
        _write_metric_pos(
            prev_mesh.nodes_orig,
            prev_mesh.elements_orig,
            M_per_orig_node,
            pos_path,
        )

        gmsh.initialize()
        try:
            if quiet:
                gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(name)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", h_floor)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h_ceiling)

            phys_lookup = geometry_builder()

            existing_views = set(gmsh.view.getTags())
            gmsh.merge(pos_path)
            new_views = [t for t in gmsh.view.getTags() if t not in existing_views]
            if not new_views:
                raise RuntimeError(f"failed to load metric view from {pos_path}")
            view_tag = new_views[-1]

            field_tag = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

            gmsh.option.setNumber("Mesh.Algorithm", 7)
            gmsh.option.setNumber("Mesh.AnisoMax", float(aniso_max))
            gmsh.option.setNumber("Mesh.SmoothRatio", 1.8)

            gmsh.model.mesh.generate(2)
            return mesh_lib._extract_lbmesh(phys_lookup)
        finally:
            gmsh.finalize()
    finally:
        try:
            os.unlink(pos_path)
        except OSError:
            pass


# ─── (4) AMR driver ───────────────────────────────────────────────────────────

def run_anisotropic_amr_loop(
    seed_mesh: mesh_lib.LBMesh,
    geometry_builder: Callable[[], dict[int, str]],
    yield_crit: functions.YieldCriterion,
    *,
    target_n: int,
    n_iterations: int = 4,
    growth_factor: float = 1.0,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence[optimizer.TractionBC] = (),
    solver: str = "MOSEK",
    aniso_ratio_max: float = 5.0,
    on_iter: Callable[[amr.AdaptiveStep], None] | None = None,
) -> list[amr.AdaptiveStep]:
    history: list[amr.AdaptiveStep] = []
    current_target = int(target_n)

    mesh = seed_mesh
    sol = optimizer.optimize_socp(
        mesh, yield_crit,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
        boundary_conditions=boundary_conditions,
        solver=solver,
    )
    eta_T, eta, mu = amr.error_indicator(mesh, sol.lagrange_multiplier)
    step = amr.AdaptiveStep(0, mesh, sol, eta=eta, mu=mu, eta_T=eta_T)
    history.append(step)
    if on_iter is not None:
        on_iter(step)

    for it in range(1, n_iterations + 1):
        current_target = int(round(current_target * growth_factor))
        M_elem = anisotropic_metric_per_element(
            mesh, sol.lagrange_multiplier, current_target,
            aniso_ratio_max=aniso_ratio_max,
        )
        M_node = _metric_per_orig_node(mesh, M_elem)
        mesh = mesh_anisotropy(geometry_builder, mesh, M_node)
        sol = optimizer.optimize_socp(
            mesh, yield_crit,
            fixed_body_force=fixed_body_force,
            scaled_body_force=scaled_body_force,
            boundary_conditions=boundary_conditions,
            solver=solver,
        )
        eta_T, eta, mu = amr.error_indicator(mesh, sol.lagrange_multiplier)
        step = amr.AdaptiveStep(it, mesh, sol, eta=eta, mu=mu, eta_T=eta_T)
        history.append(step)
        if on_iter is not None:
            on_iter(step)

    return history
