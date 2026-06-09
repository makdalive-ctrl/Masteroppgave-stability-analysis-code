from __future__ import annotations
import math
from typing import Callable, Sequence, Tuple
import gmsh
import matplotlib.tri as mtri
import numpy as np

from amr import (
    _element_gradient,
    recovered_hessian,
    dominant_hessian_eigenvalue,
)
from mesh import LBMesh, unique_nodes

_EPS = 1e-9


# ─── (1) BUILD a fan mesh ─────────────────────────────────────────────────────

# --- Public API ---

def mesh_fan(
    cfg: dict, *,
    target_n: int,
    theta_R: np.ndarray | None = None,
    theta_L: np.ndarray | None = None,
    outer_size_callback: Callable[[float, float], float] | None = None,
    outer_metric_pos_path: str | None = None,
    fan_h_min: float | None = None,
    n_angular_R: int | None = None,
    n_angular_L: int | None = None,
) -> LBMesh:
    geo = cfg["geometry"]
    if geo["kind"] != "strip_footing":
        raise ValueError(
            f"fan dispatch requires geometry.kind = strip_footing; got {geo['kind']!r}"
        )
    fan_radius = float(cfg["fan_radius"])
    M = int(cfg["M_radial"])
    N_default = int(cfg["N_angular"])
    mirrored = bool(geo.get("mirrored", False))
    half_sym = not mirrored
    n_fans = 1 if half_sym else 2

    N_R = n_angular_R if n_angular_R is not None else N_default
    N_L = n_angular_L if n_angular_L is not None else N_default
    n_fan_total = N_R * (2 * M - 1) + (0 if half_sym else N_L * (2 * M - 1))
    h_far = h_far_from_target(geo, fan_radius, n_fans, n_fan_total, target_n)

    if cfg.get("adaptive_fan"):
        if theta_R is None:
            theta_R = np.linspace(np.pi, 2.0 * np.pi, N_R + 1)
        if not half_sym and theta_L is None:
            theta_L = np.linspace(np.pi, 2.0 * np.pi, N_L + 1)
        return gmsh_adaptive_fan(
            B=float(geo["B"]), W=float(geo["W"]), D=float(geo["D"]),
            fan_radius=fan_radius, M_radial=M,
            theta_array_right=theta_R,
            theta_array_left=theta_L,
            h_far=h_far,
            fan_h_min=fan_h_min,
            half_symmetric=half_sym,
            outer_size_callback=outer_size_callback,
            outer_metric_pos_path=outer_metric_pos_path,
        )
    return gmsh_uniform_fan(
        B=float(geo["B"]), W=float(geo["W"]), D=float(geo["D"]),
        fan_radius=fan_radius, M_radial=M, N_angular=N_R,
        h_far=h_far,
        fan_h_min=fan_h_min,
        half_symmetric=half_sym,
        outer_size_callback=outer_size_callback,
        outer_metric_pos_path=outer_metric_pos_path,
    )


def gmsh_uniform_fan(
    *,
    B: float = 1.0,
    W: float = 5.0,
    D: float = 2.5,
    fan_radius: float = 1.35,
    M_radial: int = 2,
    N_angular: int = 16,
    h_far: float = 0.20,
    fan_h_min: float | None = None,
    half_symmetric: bool = True,
    outer_size_callback: Callable[[float, float], float] | None = None,
    outer_metric_pos_path: str | None = None,
    quiet: bool = True,
) -> LBMesh:
    theta_R = np.linspace(math.pi, 2.0 * math.pi, N_angular + 1)
    theta_L = None if half_symmetric else theta_R.copy()
    return gmsh_adaptive_fan(
        B=B, W=W, D=D,
        fan_radius=fan_radius, M_radial=M_radial,
        theta_array_right=theta_R, theta_array_left=theta_L,
        h_far=h_far, fan_h_min=fan_h_min,
        half_symmetric=half_symmetric,
        outer_size_callback=outer_size_callback,
        outer_metric_pos_path=outer_metric_pos_path,
        quiet=quiet,
    )


def gmsh_adaptive_fan(
    *,
    B: float = 1.0,
    W: float = 5.0,
    D: float = 2.5,
    fan_radius: float = 0.45,
    M_radial: int = 4,
    theta_array_right: Sequence[float],
    theta_array_left: Sequence[float] | None = None,
    h_far: float = 0.20,
    fan_h_min: float | None = None,
    half_symmetric: bool = True,
    outer_size_callback: Callable[[float, float], float] | None = None,
    outer_metric_pos_path: str | None = None,
    quiet: bool = True,
) -> LBMesh:
    theta_R = np.asarray(theta_array_right, dtype=float)
    if half_symmetric:
        if 2.0 * fan_radius >= B:
            raise ValueError(
                f"fan_radius={fan_radius} too large; require "
                f"2*fan_radius < B={B}.")
        N_max = theta_R.size - 1
        theta_L = None
    else:
        if theta_array_left is None:
            raise ValueError(
                "theta_array_left must be provided when half_symmetric=False.")
        theta_L = np.asarray(theta_array_left, dtype=float)
        if 2.0 * fan_radius >= B:
            raise ValueError(
                f"fan_radius={fan_radius} too large; require "
                f"2*fan_radius < B={B}.")
        N_max = max(theta_L.size - 1, theta_R.size - 1)

    if fan_h_min is None:
        fan_h_min = float(math.pi * fan_radius / N_max)

    if half_symmetric:
        return _build_half_symmetric(
            B, W, D, fan_radius, M_radial, theta_R,
            h_far, fan_h_min, quiet, outer_size_callback,
            outer_metric_pos_path=outer_metric_pos_path,
        )
    return _build_full_symmetric(
        B, W, D, fan_radius, M_radial, theta_L, theta_R,
        h_far, fan_h_min, quiet, outer_size_callback,
        outer_metric_pos_path=outer_metric_pos_path,
    )


# --- Top-level construction dispatch ---

def _build_half_symmetric(
    B: float, W: float, D: float,
    fan_radius: float, M_radial: int,
    theta_array_right: np.ndarray,
    h_far: float, fan_h_min: float,
    quiet: bool,
    outer_size_callback: Callable[[float, float], float] | None,
    outer_metric_pos_path: str | None = None,
) -> LBMesh:
    fan_R_nodes, fan_R_tris, fan_R_grid = _build_fan_from_theta(
        +B / 2.0, 0.0, fan_radius, M_radial, theta_array_right)

    N_R = theta_array_right.size - 1
    fan_R_arc_local_ids = [fan_R_grid[M_radial][i] for i in range(N_R + 1)]
    fan_R_arc_coords = [fan_R_nodes[nid] for nid in fan_R_arc_local_ids]

    outer_nodes, outer_tris, outer_arc_R_ids = _build_outer_mesh_one_fan_half(
        B, W, D, fan_R_arc_coords, fan_radius, h_far, fan_h_min, quiet,
        outer_size_callback=outer_size_callback,
        outer_metric_pos_path=outer_metric_pos_path,
    )

    coords, triangles = _stitch_one_fan_half(
        fan_R_nodes, fan_R_tris, fan_R_arc_local_ids,
        outer_nodes, outer_tris, outer_arc_R_ids,
    )
    edge_phys = _tag_boundary_edges_half(coords, triangles, B, W, D)
    return unique_nodes(coords, triangles, edge_phys)


def _build_full_symmetric(
    B: float, W: float, D: float,
    fan_radius: float, M_radial: int,
    theta_array_left: np.ndarray, theta_array_right: np.ndarray,
    h_far: float, fan_h_min: float,
    quiet: bool,
    outer_size_callback: Callable[[float, float], float] | None,
    outer_metric_pos_path: str | None = None,
) -> LBMesh:
    fan_L_nodes, fan_L_tris, fan_L_grid = _build_fan_from_theta(
        -B / 2.0, 0.0, fan_radius, M_radial, theta_array_left)
    fan_R_nodes, fan_R_tris, fan_R_grid = _build_fan_from_theta(
        +B / 2.0, 0.0, fan_radius, M_radial, theta_array_right)

    N_L = theta_array_left.size - 1
    N_R = theta_array_right.size - 1
    fan_L_arc_local_ids = [fan_L_grid[M_radial][i] for i in range(N_L + 1)]
    fan_R_arc_local_ids = [fan_R_grid[M_radial][i] for i in range(N_R + 1)]
    fan_L_arc_coords = [fan_L_nodes[nid] for nid in fan_L_arc_local_ids]
    fan_R_arc_coords = [fan_R_nodes[nid] for nid in fan_R_arc_local_ids]

    outer_nodes, outer_tris, outer_arc_L_ids, outer_arc_R_ids = _build_outer_mesh_two_fans(
        B, W, D,
        fan_L_arc_coords, fan_R_arc_coords,
        fan_radius, h_far, fan_h_min, quiet,
        outer_size_callback=outer_size_callback,
        outer_metric_pos_path=outer_metric_pos_path,
    )

    coords, triangles = _stitch_two_fans(
        fan_L_nodes, fan_L_tris, fan_L_arc_local_ids,
        fan_R_nodes, fan_R_tris, fan_R_arc_local_ids,
        outer_nodes, outer_tris, outer_arc_L_ids, outer_arc_R_ids,
    )
    edge_phys = _tag_boundary_edges(coords, triangles, B, W, D)
    return unique_nodes(coords, triangles, edge_phys)


# --- Structured fan disk (pure Python, no gmsh) ---

def _build_fan_from_theta(
    cx: float, cy: float, R: float, M: int,
    theta_array: Sequence[float],
) -> Tuple[
    dict[int, tuple[float, float]],
    list[tuple[int, int, int]],
    list[list[int]],
]:
    theta = np.asarray(theta_array, dtype=float)
    N = theta.size - 1
    if N < 1:
        raise ValueError("theta_array must have at least 2 entries (1 wedge).")
    if not np.all(np.diff(theta) > 0):
        raise ValueError("theta_array must be strictly monotone increasing.")
    if abs(theta[0] - math.pi) > 1.0e-9 or abs(theta[-1] - 2.0 * math.pi) > 1.0e-9:
        raise ValueError(
            f"theta_array endpoints must be (pi, 2*pi); got "
            f"({theta[0]:.6f}, {theta[-1]:.6f}). The fan covers the lower "
            f"half-plane only."
        )

    nodes: dict[int, tuple[float, float]] = {}
    grid: list[list[int]] = [[0] * (N + 1) for _ in range(M + 1)]
    nid = 0

    nid += 1
    nodes[nid] = (cx, cy)
    for i in range(N + 1):
        grid[0][i] = nid

    for j in range(1, M + 1):
        rj = R * j / M
        for i in range(N + 1):
            t = float(theta[i])
            x = cx + rj * math.cos(t)
            y = cy + rj * math.sin(t)
            if i == 0:
                x, y = cx - rj, cy
            elif i == N:
                x, y = cx + rj, cy
            nid += 1
            nodes[nid] = (x, y)
            grid[j][i] = nid

    triangles: list[tuple[int, int, int]] = []
    for i in range(N):
        triangles.append((grid[0][i], grid[1][i], grid[1][i + 1]))
    for j in range(1, M):
        for i in range(N):
            ni, nip = grid[j][i], grid[j][i + 1]
            no, nop = grid[j + 1][i], grid[j + 1][i + 1]
            triangles.append((ni, no, nop))
            triangles.append((ni, nop, nip))

    return nodes, triangles, grid


# --- Gmsh outer mesh ---

def _build_outer_mesh_one_fan_half(
    B: float, W: float, D: float,
    fan_R_arc_coords: list[tuple[float, float]],
    fan_radius: float,
    h_far: float,
    fan_h_min: float,
    quiet: bool,
    *,
    outer_size_callback: Callable[[float, float], float] | None = None,
    outer_metric_pos_path: str | None = None,
) -> Tuple[
    dict[int, tuple[float, float]],
    list[tuple[int, int, int]],
    list[int | None],
]:
    cR = +B / 2.0
    R_left = fan_R_arc_coords[0]
    R_right = fan_R_arc_coords[-1]

    gmsh.initialize()
    try:
        if quiet:
            gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("strip_footing_outer_half")

        half_W = 0.5 * W
        TL = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, fan_h_min)
        Rle = gmsh.model.geo.addPoint(R_left[0], R_left[1], 0.0, fan_h_min)
        Rri = gmsh.model.geo.addPoint(R_right[0], R_right[1], 0.0, fan_h_min)
        TR = gmsh.model.geo.addPoint(half_W, 0.0, 0.0, h_far)
        BR = gmsh.model.geo.addPoint(half_W, -D, 0.0, h_far)
        BL = gmsh.model.geo.addPoint(0.0, -D, 0.0, h_far)

        cR_pt = gmsh.model.geo.addPoint(cR, 0.0, 0.0, fan_h_min)

        def _arc_segments(arc_coords, left_endpt_tag, right_endpt_tag, centre_tag):
            inner_pts = []
            for k in range(1, len(arc_coords) - 1):
                x, y = arc_coords[k]
                inner_pts.append(gmsh.model.geo.addPoint(x, y, 0.0, fan_h_min))
            chain = [left_endpt_tag] + inner_pts + [right_endpt_tag]
            return [
                gmsh.model.geo.addCircleArc(chain[k], centre_tag, chain[k + 1])
                for k in range(len(chain) - 1)
            ]

        R_arc_segs = _arc_segments(fan_R_arc_coords, Rle, Rri, cR_pt)

        l_footing_outer = gmsh.model.geo.addLine(TL, Rle)
        l_free = gmsh.model.geo.addLine(Rri, TR)
        l_right = gmsh.model.geo.addLine(TR, BR)
        l_bottom = gmsh.model.geo.addLine(BR, BL)
        l_symmetry = gmsh.model.geo.addLine(BL, TL)

        loop = gmsh.model.geo.addCurveLoop(
            [l_footing_outer] + R_arc_segs + [l_free, l_right, l_bottom, l_symmetry]
        )
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        if outer_metric_pos_path is not None:
            existing_views = set(gmsh.view.getTags())
            gmsh.merge(outer_metric_pos_path)
            new_views = [t for t in gmsh.view.getTags() if t not in existing_views]
            if not new_views:
                raise RuntimeError(
                    f"failed to load metric view from {outer_metric_pos_path}"
                )
            view_tag = new_views[-1]
            field_tag = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        elif outer_size_callback is None:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "PointsList", [cR_pt])
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "InField", 1)
            gmsh.model.mesh.field.setNumber(2, "SizeMin", fan_h_min)
            gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
            gmsh.model.mesh.field.setNumber(2, "DistMin", fan_radius)
            gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * fan_radius)
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
        else:
            def _gmsh_cb(dim, tag, x, y, z, lc):
                try:
                    return float(outer_size_callback(float(x), float(y)))
                except Exception:
                    return float(h_far)
            gmsh.model.mesh.setSizeCallback(_gmsh_cb)

        if outer_metric_pos_path is not None:
            gmsh.option.setNumber("Mesh.Algorithm", 7)
            gmsh.option.setNumber("Mesh.AnisoMax", 500.0)
            gmsh.option.setNumber("Mesh.SmoothRatio", 1.2)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", fan_h_min * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)

        gmsh.model.mesh.generate(2)

        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
        outer_nodes: dict[int, tuple[float, float]] = {}
        for k, tag in enumerate(node_tags):
            outer_nodes[int(tag)] = (float(coords_flat[3 * k]),
                                     float(coords_flat[3 * k + 1]))

        outer_tris: list[tuple[int, int, int]] = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        for et, ent in zip(elem_types, elem_node_tags):
            if et == 2:
                arr = np.asarray(ent, dtype=np.int64).reshape(-1, 3)
                for row in arr:
                    outer_tris.append((int(row[0]), int(row[1]), int(row[2])))

        def _find_arc_ids(arc_coords):
            ids = []
            for ax, ay in arc_coords:
                found = None
                for nid, (nx, ny) in outer_nodes.items():
                    if _close(nx, ax) and _close(ny, ay):
                        found = nid
                        break
                ids.append(found)
            return ids

        outer_arc_R_ids = _find_arc_ids(fan_R_arc_coords)
        return outer_nodes, outer_tris, outer_arc_R_ids

    finally:
        gmsh.finalize()


def _build_outer_mesh_two_fans(
    B: float, W: float, D: float,
    fan_L_arc_coords: list[tuple[float, float]],
    fan_R_arc_coords: list[tuple[float, float]],
    fan_radius: float,
    h_far: float,
    fan_h_min: float,
    quiet: bool,
    *,
    outer_size_callback: Callable[[float, float], float] | None = None,
    outer_metric_pos_path: str | None = None,
) -> Tuple[
    dict[int, tuple[float, float]],
    list[tuple[int, int, int]],
    list[int | None],
    list[int | None],
]:
    cL, cR = -B / 2, +B / 2
    L_left = fan_L_arc_coords[0]
    L_right = fan_L_arc_coords[-1]
    R_left = fan_R_arc_coords[0]
    R_right = fan_R_arc_coords[-1]

    gmsh.initialize()
    try:
        if quiet:
            gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("strip_footing_outer")

        TL = gmsh.model.geo.addPoint(-W / 2, 0.0, 0.0, h_far)
        Lle = gmsh.model.geo.addPoint(L_left[0], L_left[1], 0.0, fan_h_min)
        Lri = gmsh.model.geo.addPoint(L_right[0], L_right[1], 0.0, fan_h_min)
        Rle = gmsh.model.geo.addPoint(R_left[0], R_left[1], 0.0, fan_h_min)
        Rri = gmsh.model.geo.addPoint(R_right[0], R_right[1], 0.0, fan_h_min)
        TR = gmsh.model.geo.addPoint(+W / 2, 0.0, 0.0, h_far)
        BR = gmsh.model.geo.addPoint(+W / 2, -D, 0.0, h_far)
        BL = gmsh.model.geo.addPoint(-W / 2, -D, 0.0, h_far)

        cL_pt = gmsh.model.geo.addPoint(cL, 0.0, 0.0, fan_h_min)
        cR_pt = gmsh.model.geo.addPoint(cR, 0.0, 0.0, fan_h_min)

        def _arc_segments(arc_coords, left_endpt_tag, right_endpt_tag, centre_tag):
            inner_pts = []
            for k in range(1, len(arc_coords) - 1):
                x, y = arc_coords[k]
                inner_pts.append(gmsh.model.geo.addPoint(x, y, 0.0, fan_h_min))
            chain = [left_endpt_tag] + inner_pts + [right_endpt_tag]
            return [
                gmsh.model.geo.addCircleArc(chain[k], centre_tag, chain[k + 1])
                for k in range(len(chain) - 1)
            ]

        L_arc_segs = _arc_segments(fan_L_arc_coords, Lle, Lri, cL_pt)
        R_arc_segs = _arc_segments(fan_R_arc_coords, Rle, Rri, cR_pt)

        l_free_left = gmsh.model.geo.addLine(TL, Lle)
        l_mid = gmsh.model.geo.addLine(Lri, Rle)
        l_free_right = gmsh.model.geo.addLine(Rri, TR)
        l_right = gmsh.model.geo.addLine(TR, BR)
        l_bottom = gmsh.model.geo.addLine(BR, BL)
        l_left = gmsh.model.geo.addLine(BL, TL)

        loop = gmsh.model.geo.addCurveLoop(
            [l_free_left] + L_arc_segs + [l_mid] + R_arc_segs +
            [l_free_right, l_right, l_bottom, l_left]
        )
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        if outer_metric_pos_path is not None:
            existing_views = set(gmsh.view.getTags())
            gmsh.merge(outer_metric_pos_path)
            new_views = [t for t in gmsh.view.getTags() if t not in existing_views]
            if not new_views:
                raise RuntimeError(
                    f"failed to load metric view from {outer_metric_pos_path}"
                )
            view_tag = new_views[-1]
            field_tag = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        elif outer_size_callback is None:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "PointsList", [cL_pt, cR_pt])
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "InField", 1)
            gmsh.model.mesh.field.setNumber(2, "SizeMin", fan_h_min)
            gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
            gmsh.model.mesh.field.setNumber(2, "DistMin", fan_radius)
            gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * fan_radius)
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
        else:
            def _gmsh_cb(dim, tag, x, y, z, lc):
                try:
                    return float(outer_size_callback(float(x), float(y)))
                except Exception:
                    return float(h_far)
            gmsh.model.mesh.setSizeCallback(_gmsh_cb)

        if outer_metric_pos_path is not None:
            gmsh.option.setNumber("Mesh.Algorithm", 7)
            gmsh.option.setNumber("Mesh.AnisoMax", 100.0)
            gmsh.option.setNumber("Mesh.SmoothRatio", 1.8)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", fan_h_min * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)

        gmsh.model.mesh.generate(2)

        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
        outer_nodes: dict[int, tuple[float, float]] = {}
        for k, tag in enumerate(node_tags):
            outer_nodes[int(tag)] = (float(coords_flat[3 * k]),
                                     float(coords_flat[3 * k + 1]))

        outer_tris: list[tuple[int, int, int]] = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        for et, ent in zip(elem_types, elem_node_tags):
            if et == 2:
                arr = np.asarray(ent, dtype=np.int64).reshape(-1, 3)
                for row in arr:
                    outer_tris.append((int(row[0]), int(row[1]), int(row[2])))

        def _find_arc_ids(arc_coords):
            ids = []
            for ax, ay in arc_coords:
                found = None
                for nid, (nx, ny) in outer_nodes.items():
                    if _close(nx, ax) and _close(ny, ay):
                        found = nid
                        break
                ids.append(found)
            return ids

        outer_arc_L_ids = _find_arc_ids(fan_L_arc_coords)
        outer_arc_R_ids = _find_arc_ids(fan_R_arc_coords)

        return outer_nodes, outer_tris, outer_arc_L_ids, outer_arc_R_ids

    finally:
        gmsh.finalize()


# --- Stitching (combine fan + outer) ---

def _stitch_one_fan_half(
    fan_R_nodes: dict[int, tuple[float, float]],
    fan_R_tris: list[tuple[int, int, int]],
    fan_R_arc_local_ids: list[int],
    outer_nodes: dict[int, tuple[float, float]],
    outer_tris: list[tuple[int, int, int]],
    outer_arc_R_ids: list[int | None],
) -> Tuple[np.ndarray, np.ndarray]:
    combined_coords: list[tuple[float, float]] = []
    outer_id_map: dict[int, int] = {}
    fan_id_map: dict[int, int] = {}

    for old_id, (x, y) in outer_nodes.items():
        combined_coords.append((x, y))
        outer_id_map[old_id] = len(combined_coords) - 1

    arc_set = set(fan_R_arc_local_ids)
    for fan_local_id, (x, y) in fan_R_nodes.items():
        if fan_local_id in arc_set:
            arc_index = fan_R_arc_local_ids.index(fan_local_id)
            outer_id = outer_arc_R_ids[arc_index]
            if outer_id is None:
                combined_coords.append((x, y))
                fan_id_map[fan_local_id] = len(combined_coords) - 1
            else:
                fan_id_map[fan_local_id] = outer_id_map[outer_id]
        else:
            combined_coords.append((x, y))
            fan_id_map[fan_local_id] = len(combined_coords) - 1

    triangles: list[tuple[int, int, int]] = []
    for n1, n2, n3 in outer_tris:
        triangles.append((outer_id_map[n1], outer_id_map[n2], outer_id_map[n3]))
    for n1, n2, n3 in fan_R_tris:
        triangles.append((fan_id_map[n1], fan_id_map[n2], fan_id_map[n3]))

    coords = np.asarray(combined_coords, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    return coords, tri


def _stitch_two_fans(
    fan_L_nodes: dict[int, tuple[float, float]],
    fan_L_tris: list[tuple[int, int, int]],
    fan_L_arc_local_ids: list[int],
    fan_R_nodes: dict[int, tuple[float, float]],
    fan_R_tris: list[tuple[int, int, int]],
    fan_R_arc_local_ids: list[int],
    outer_nodes: dict[int, tuple[float, float]],
    outer_tris: list[tuple[int, int, int]],
    outer_arc_L_ids: list[int | None],
    outer_arc_R_ids: list[int | None],
) -> Tuple[np.ndarray, np.ndarray]:
    combined_coords: list[tuple[float, float]] = []
    outer_id_map: dict[int, int] = {}
    fan_L_id_map: dict[int, int] = {}
    fan_R_id_map: dict[int, int] = {}

    for old_id, (x, y) in outer_nodes.items():
        combined_coords.append((x, y))
        outer_id_map[old_id] = len(combined_coords) - 1

    def _add_fan(fan_nodes, fan_arc_local_ids, outer_arc_ids, fan_id_map):
        arc_set = set(fan_arc_local_ids)
        for fan_local_id, (x, y) in fan_nodes.items():
            if fan_local_id in arc_set:
                arc_index = fan_arc_local_ids.index(fan_local_id)
                outer_id = outer_arc_ids[arc_index]
                if outer_id is None:
                    combined_coords.append((x, y))
                    fan_id_map[fan_local_id] = len(combined_coords) - 1
                else:
                    fan_id_map[fan_local_id] = outer_id_map[outer_id]
            else:
                combined_coords.append((x, y))
                fan_id_map[fan_local_id] = len(combined_coords) - 1

    _add_fan(fan_L_nodes, fan_L_arc_local_ids, outer_arc_L_ids, fan_L_id_map)
    _add_fan(fan_R_nodes, fan_R_arc_local_ids, outer_arc_R_ids, fan_R_id_map)

    triangles: list[tuple[int, int, int]] = []
    for n1, n2, n3 in outer_tris:
        triangles.append((outer_id_map[n1], outer_id_map[n2], outer_id_map[n3]))
    for n1, n2, n3 in fan_L_tris:
        triangles.append((fan_L_id_map[n1], fan_L_id_map[n2], fan_L_id_map[n3]))
    for n1, n2, n3 in fan_R_tris:
        triangles.append((fan_R_id_map[n1], fan_R_id_map[n2], fan_R_id_map[n3]))

    coords = np.asarray(combined_coords, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    return coords, tri


# --- Boundary tagging ---

def _tag_boundary_edges_half(
    coords: np.ndarray,
    triangles: np.ndarray,
    B: float, W: float, D: float,
    tol: float = 1e-7,
) -> dict[tuple[int, int], str]:
    edge_use: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for li in range(3):
            a, b = int(tri[(li + 1) % 3]), int(tri[(li + 2) % 3])
            key = (min(a, b), max(a, b))
            edge_use[key] = edge_use.get(key, 0) + 1

    half_B = 0.5 * B
    half_W = 0.5 * W
    edge_phys: dict[tuple[int, int], str] = {}
    for (a, b), count in edge_use.items():
        if count != 1:
            continue
        mx = 0.5 * (coords[a, 0] + coords[b, 0])
        my = 0.5 * (coords[a, 1] + coords[b, 1])
        if abs(my) < tol:
            if -tol < mx < half_B + tol:
                edge_phys[(a, b)] = "footing"
            else:
                edge_phys[(a, b)] = "free"
        elif abs(my + D) < tol:
            edge_phys[(a, b)] = "support_bottom"
        elif abs(mx) < tol:
            edge_phys[(a, b)] = "symmetry"
        elif abs(mx - half_W) < tol:
            edge_phys[(a, b)] = "support_right"
    return edge_phys


def _tag_boundary_edges(
    coords: np.ndarray,
    triangles: np.ndarray,
    B: float, W: float, D: float,
    tol: float = 1e-7,
) -> dict[tuple[int, int], str]:
    edge_use: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for li in range(3):
            a, b = int(tri[(li + 1) % 3]), int(tri[(li + 2) % 3])
            key = (min(a, b), max(a, b))
            edge_use[key] = edge_use.get(key, 0) + 1

    half_B = 0.5 * B
    half_W = 0.5 * W
    edge_phys: dict[tuple[int, int], str] = {}
    for (a, b), count in edge_use.items():
        if count != 1:
            continue
        mx = 0.5 * (coords[a, 0] + coords[b, 0])
        my = 0.5 * (coords[a, 1] + coords[b, 1])
        if abs(my) < tol:
            if -half_B - tol < mx < half_B + tol:
                edge_phys[(a, b)] = "footing"
            else:
                edge_phys[(a, b)] = "free"
        elif abs(my + D) < tol:
            edge_phys[(a, b)] = "support_bottom"
        elif abs(mx + half_W) < tol:
            edge_phys[(a, b)] = "support_left"
        elif abs(mx - half_W) < tol:
            edge_phys[(a, b)] = "support_right"
    return edge_phys


# ─── (2) COMPUTE fan AMR indicators ───────────────────────────────────────────

# --- Angular spacing (theta redistribution) ---

def circumferential_stress_gradient_norm(
    mesh: LBMesh,
    sigma: np.ndarray,
    fan_center: tuple[float, float],
    fan_radius: float,
    *,
    n_bins: int = 24,
    half_disk: bool = True,
    r_min_frac: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cx, cy = fan_center

    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    cdx = centroids[:, 0] - cx
    cdy = centroids[:, 1] - cy
    cr = np.sqrt(cdx * cdx + cdy * cdy)

    if half_disk:
        ctheta = np.arctan2(-cdy, cdx)
        in_half = cdy <= 1.0e-12
    else:
        ctheta = np.arctan2(cdy, cdx)
        in_half = np.ones_like(ctheta, dtype=bool)

    inside = (cr <= fan_radius) & (cr > r_min_frac * fan_radius) & in_half

    grad_sxx = _element_gradient(mesh, sigma[:, :, 0])
    grad_syy = _element_gradient(mesh, sigma[:, :, 1])
    grad_sxy = _element_gradient(mesh, sigma[:, :, 2])

    inv_r = np.where(cr > 0, 1.0 / cr, 0.0)
    e_tx = -cdy * inv_r
    e_ty = cdx * inv_r

    dt_sxx = grad_sxx[:, 0] * e_tx + grad_sxx[:, 1] * e_ty
    dt_syy = grad_syy[:, 0] * e_tx + grad_syy[:, 1] * e_ty
    dt_sxy = grad_sxy[:, 0] * e_tx + grad_sxy[:, 1] * e_ty

    g_theta = np.sqrt(dt_sxx * dt_sxx + dt_syy * dt_syy + 2.0 * dt_sxy * dt_sxy)

    if half_disk:
        bin_edges = np.linspace(0.0, np.pi, n_bins + 1)
    else:
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    th = ctheta[inside]
    g = g_theta[inside]

    rate = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        if k == n_bins - 1:
            m = (th >= bin_edges[k]) & (th <= bin_edges[k + 1])
        else:
            m = (th >= bin_edges[k]) & (th < bin_edges[k + 1])
        counts[k] = int(m.sum())
        if counts[k] >= 1:
            rate[k] = float(g[m].mean())

    populated = counts >= 1
    median_rate = float(np.median(rate[populated])) if populated.any() else 0.0
    max_rate = float(rate.max()) if rate.size else 0.0
    concentration = (max_rate / max(median_rate, 1.0e-12)) if populated.any() else 0.0

    diagnostics = {
        "bin_edges": bin_edges,
        "counts": counts,
        "n_total_in_fan": int(inside.sum()),
        "median_rate": median_rate,
        "max_rate": max_rate,
        "concentration": concentration,
    }
    return bin_centers, rate, diagnostics


def theta_array_from_indicator(
    rate: np.ndarray,
    target_N: int,
    *,
    p: float = 2.0,
    eps_floor: float = 0.05,
    indicator_convention: str = "right_to_left",
) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    if rate.size < 2:
        raise ValueError("need at least 2 bins")
    if indicator_convention == "right_to_left":
        rate = rate[::-1]
    elif indicator_convention != "left_to_right":
        raise ValueError(
            f"indicator_convention must be 'right_to_left' or 'left_to_right'; "
            f"got {indicator_convention!r}.")

    K = rate.size
    theta_min = math.pi
    theta_max = 2.0 * math.pi
    bin_edges = np.linspace(theta_min, theta_max, K + 1)
    bin_width = float(bin_edges[1] - bin_edges[0])

    r_max = float(rate.max()) if rate.max() > 0 else 1.0
    r = np.maximum(rate, eps_floor * r_max)

    density = r ** (1.0 / (p + 1.0))
    cum = np.concatenate([[0.0], np.cumsum(density * bin_width)])
    total = float(cum[-1])
    if total <= 0:
        return np.linspace(theta_min, theta_max, target_N + 1)
    cdf = cum / total

    targets = np.linspace(0.0, 1.0, target_N + 1)
    theta_array = np.interp(targets, cdf, bin_edges)
    theta_array[0] = theta_min
    theta_array[-1] = theta_max

    eps_min = 1.0e-6 * (theta_max - theta_min) / target_N
    for i in range(1, theta_array.size):
        if theta_array[i] - theta_array[i - 1] < eps_min:
            theta_array[i] = theta_array[i - 1] + eps_min

    return theta_array


# --- Outer-mesh size field ---

def _outer_mask_and_areas(
    mesh: LBMesh,
    fan_centers: list[tuple[float, float]],
    fan_radius_check: float,
) -> Tuple[np.ndarray, np.ndarray]:
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    is_outer = np.ones(mesh.n_tri, dtype=bool)
    for cx, cy in fan_centers:
        d = np.sqrt((centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2)
        is_outer &= (d >= fan_radius_check)
    pts = mesh.nodes[mesh.elements]
    areas = 0.5 * np.abs(
        (pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
        - (pts[:, 2, 0] - pts[:, 0, 0]) * (pts[:, 1, 1] - pts[:, 0, 1])
    )
    return is_outer, areas


def L_based_refinement_outer(
    method: str,
    mesh: LBMesh,
    lambda_per_node: np.ndarray,
    target_n_outer: int,
    fan_centers: list[tuple[float, float]],
    fan_radius_check: float,
    *,
    p: float = 2.0,
    eps_floor: float | None = None,
    h_min_factor: float = 0.05,
    h_max_factor: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    is_outer, areas = _outer_mask_and_areas(mesh, fan_centers, fan_radius_check)

    if method == "L value":
        rho_all = np.abs(lambda_per_node).mean(axis=1)
        default_eps = 1.0e-3
    elif method == "L gradient":
        grad = _element_gradient(mesh, lambda_per_node)
        rho_all = np.linalg.norm(grad, axis=1)
        default_eps = 0.10
    elif method == "L hessian":
        H = recovered_hessian(mesh, lambda_per_node)
        rho_all = dominant_hessian_eigenvalue(H)
        default_eps = 0.02
    else:
        raise ValueError(
            f"unknown method {method!r}; use 'L value', 'L gradient', or 'L hessian'"
        )

    eps = default_eps if eps_floor is None else float(eps_floor)

    h_all = np.full(mesh.n_tri, np.nan)
    if not is_outer.any():
        return h_all, is_outer

    rho_outer = rho_all[is_outer]
    areas_outer = areas[is_outer]
    if float(rho_outer.max()) <= 0.0:
        h_uniform = float(np.sqrt(areas_outer.sum() * 4.0 / (np.sqrt(3.0) * target_n_outer)))
        h_all[is_outer] = h_uniform
        return h_all, is_outer

    rho_outer = np.maximum(rho_outer, eps * float(rho_outer.max()))

    if method == "L gradient":
        raw = float(np.sum(areas_outer * rho_outer ** 2))
        alpha = np.sqrt((4.0 / (np.sqrt(3.0) * target_n_outer)) * raw)
        h_outer = alpha / rho_outer
    else:
        exp_size = -p / (2.0 * (p + 1.0))
        raw = float(np.sum(areas_outer * rho_outer ** (-2.0 * exp_size)))
        K = np.sqrt((4.0 / (np.sqrt(3.0) * target_n_outer)) * raw)
        h_outer = K * rho_outer ** exp_size

    h_typical = np.sqrt(areas_outer.mean())
    h_outer = np.clip(h_outer, h_min_factor * h_typical, h_max_factor * h_typical)
    h_all[is_outer] = h_outer
    return h_all, is_outer


# ─── (3) AMR glue (indicators → next-iteration build parameters) ──────────────

def make_outer_size_callback(
    old_mesh: LBMesh,
    h_per_element: np.ndarray,
    fallback: float,
) -> Callable[[float, float], float]:
    triangulation = mtri.Triangulation(
        old_mesh.nodes_orig[:, 0],
        old_mesh.nodes_orig[:, 1],
        old_mesh.elements_orig,
    )
    finder = triangulation.get_trifinder()
    h_arr = np.asarray(h_per_element, dtype=float)

    def cb(x: float, y: float) -> float:
        e = int(finder(x, y))
        if e < 0:
            return fallback
        h = h_arr[e]
        if not np.isfinite(h):
            return fallback
        return float(h)

    return cb


def fan_centers_from_config(geo: dict) -> list[tuple[float, float]]:
    B = float(geo["B"])
    mirrored = bool(geo.get("mirrored", False))
    if not mirrored:
        return [(+B / 2.0, 0.0)]
    return [(-B / 2.0, 0.0), (+B / 2.0, 0.0)]


def h_far_from_target(
    geo: dict, fan_radius: float, n_fans: int, n_fan_total: int, target_total: int,
) -> float:
    n_outer_target = max(50, int(target_total - n_fan_total))
    W = float(geo["W"])
    D = float(geo["D"])
    domain_width = W if bool(geo.get("mirrored", False)) else 0.5 * W
    outer_area = domain_width * D - n_fans * 0.5 * np.pi * fan_radius ** 2
    return float(np.sqrt(4.0 * outer_area / (np.sqrt(3.0) * n_outer_target)))


def maybe_grow_N(
    current_N: int, concentration: float, *,
    grow: bool, threshold: float, growth_per_iter: float, max_N: int,
) -> int:
    if not grow or concentration <= threshold:
        return current_N
    target = current_N * float(np.sqrt(concentration / threshold))
    capped = min(target, current_N * growth_per_iter, float(max_N))
    return max(int(np.ceil(capped)), current_N)


def matched_fan_h_min(
    fan_radius: float, theta_R: np.ndarray, theta_L: np.ndarray | None,
) -> float:
    deltas = [np.diff(theta_R)]
    if theta_L is not None:
        deltas.append(np.diff(theta_L))
    delta_min = float(min(d.min() for d in deltas))
    return fan_radius * delta_min


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _close(a: float, b: float) -> bool:
    return abs(a - b) < _EPS
