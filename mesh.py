from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gmsh
import numpy as np




@dataclass
class LBMesh:
    nodes: np.ndarray
    elements: np.ndarray
    internal_edges: np.ndarray
    boundary_edges: np.ndarray
    boundary_edge_tag: list[str]
    nodes_orig: np.ndarray
    elements_orig: np.ndarray
    dup_to_orig: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_tri(self) -> int:
        return self.elements.shape[0]

    @property
    def n_edges(self) -> int:
        return int(self.internal_edges.shape[0] + self.boundary_edges.shape[0])

    @property
    def n_internal_edges(self) -> int:
        return int(self.internal_edges.shape[0])

    @property
    def n_boundary_edges(self) -> int:
        return int(self.boundary_edges.shape[0])




def unique_nodes(coords: np.ndarray, triangles: np.ndarray, edge_phys: dict[tuple[int, int], str]) -> LBMesh:
    n_tri = int(triangles.shape[0])

    edge_dict: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for ei, tri in enumerate(triangles):
        for li in range(3):
            a, b = int(tri[(li + 1) % 3]), int(tri[(li + 2) % 3])
            key = (min(a, b), max(a, b))
            edge_dict.setdefault(key, []).append((ei, li, a, b))

    coords_dup = np.empty((3 * n_tri, 2), dtype=np.float64)
    dup_to_orig = np.empty(3 * n_tri, dtype=np.int64)
    for e in range(n_tri):
        for li in range(3):
            orig_id = int(triangles[e, li])
            coords_dup[3 * e + li] = coords[orig_id]
            dup_to_orig[3 * e + li] = orig_id
    triangles_dup = np.arange(3 * n_tri, dtype=np.int64).reshape(n_tri, 3)

    def _dup_node(e: int, original_node: int) -> int:
        for li in range(3):
            if int(triangles[e, li]) == int(original_node):
                return 3 * e + li
        raise ValueError(f"node {original_node} is not a vertex of triangle {e}")

    internal_rows: list[tuple[int, int, int, int, int, int]] = []
    boundary_rows: list[tuple[int, int, int]] = []
    boundary_tags: list[str] = []
    for key, owners in edge_dict.items():
        if len(owners) == 1:
            ei, _, a, b = owners[0]
            boundary_rows.append((ei, _dup_node(ei, a), _dup_node(ei, b)))
            boundary_tags.append(edge_phys.get(key, ""))
        elif len(owners) == 2:
            (e1, _, a, b), (e2, _, _, _) = owners
            internal_rows.append((
                e1, e2,
                _dup_node(e1, a), _dup_node(e2, a),
                _dup_node(e1, b), _dup_node(e2, b),
            ))
        else:
            raise ValueError(f"edge {key} has {len(owners)} adjacent elements; expected 1 or 2")

    internal_edges = (np.asarray(internal_rows, dtype=np.int64).reshape(-1, 6)
                      if internal_rows else np.empty((0, 6), dtype=np.int64))
    boundary_edges = (np.asarray(boundary_rows, dtype=np.int64).reshape(-1, 3)
                      if boundary_rows else np.empty((0, 3), dtype=np.int64))

    return LBMesh(
        nodes=coords_dup,
        elements=triangles_dup,
        internal_edges=internal_edges,
        boundary_edges=boundary_edges,
        boundary_edge_tag=boundary_tags,
        nodes_orig=np.asarray(coords, dtype=np.float64),
        elements_orig=np.asarray(triangles, dtype=np.int64),
        dup_to_orig=dup_to_orig,
    )


def _extract_lbmesh(phys_lookup: dict[int, str]) -> LBMesh:
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords3 = np.asarray(coords_flat, dtype=np.float64).reshape(-1, 3)
    coords = coords3[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangles_list: list[np.ndarray] = []
    for et, ent in zip(elem_types, elem_node_tags):
        if et == 2:
            triangles_list.append(np.asarray(ent, dtype=np.int64).reshape(-1, 3))
    if not triangles_list:
        raise RuntimeError("No triangular elements were generated.")
    tri_tags = np.vstack(triangles_list)
    triangles = np.vectorize(tag_to_idx.get)(tri_tags).astype(np.int64)

    edge_phys: dict[tuple[int, int], str] = {}
    for dim, tag in gmsh.model.getPhysicalGroups(dim=1):
        phys_name = phys_lookup.get(tag) or gmsh.model.getPhysicalName(dim, tag) or f"phys_{tag}"
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            types, _, n_tags = gmsh.model.mesh.getElements(dim=1, tag=int(ent))
            for et, nt in zip(types, n_tags):
                if et == 1:
                    pairs = np.asarray(nt, dtype=np.int64).reshape(-1, 2)
                    for p in pairs:
                        a, b = tag_to_idx[int(p[0])], tag_to_idx[int(p[1])]
                        edge_phys[(min(a, b), max(a, b))] = phys_name

    return unique_nodes(coords, triangles, edge_phys)




def mesh_parameter(
    geometry_builder: Callable[[], dict[int, str]],
    target_n: int,
    *,
    initial_mesh_size: float = 0.15,
    tol: float = 0.05,
    max_iter: int = 6,
    name: str = "lb_model",
    quiet: bool = True,
    verbose: bool = False,
) -> LBMesh:
    h = float(initial_mesh_size)
    best: LBMesh | None = None
    best_err = float("inf")
    for it in range(max_iter):
        mesh = gmsh_build_mesh(geometry_builder, mesh_size=h, name=name, quiet=quiet)
        rel_err = abs(mesh.n_tri - target_n) / target_n
        if verbose:
            print(f"[target-N] iter {it}: h={h:.4f} -> {mesh.n_tri} tri (rel_err={rel_err:.3f})")
        if rel_err < best_err:
            best, best_err = mesh, rel_err
        if rel_err < tol:
            return mesh
        h *= float(np.sqrt(mesh.n_tri / target_n))
    return best




def gmsh_build_mesh(
    geometry_builder: Callable[[], dict[int, str]],
    mesh_size: float,
    name: str = "lb_model",
    quiet: bool = True,
) -> LBMesh:
    gmsh.initialize()
    try:
        if quiet:
            gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size / 4)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        phys_lookup = geometry_builder()

        gmsh.model.mesh.generate(2)

        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
        coords3 = np.asarray(coords_flat, dtype=np.float64).reshape(-1, 3)
        coords = coords3[:, :2]
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        triangles_list: list[np.ndarray] = []
        for et, ent in zip(elem_types, elem_node_tags):
            if et == 2:
                arr = np.asarray(ent, dtype=np.int64).reshape(-1, 3)
                triangles_list.append(arr)
        if not triangles_list:
            raise RuntimeError("No triangular elements were generated.")
        tri_tags = np.vstack(triangles_list)
        triangles = np.vectorize(tag_to_idx.get)(tri_tags).astype(np.int64)

        edge_phys: dict[tuple[int, int], str] = {}
        for dim, tag in gmsh.model.getPhysicalGroups(dim=1):
            phys_name = phys_lookup.get(tag) or gmsh.model.getPhysicalName(dim, tag) or f"phys_{tag}"
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                types, _, n_tags = gmsh.model.mesh.getElements(dim=1, tag=int(ent))
                for et, nt in zip(types, n_tags):
                    if et == 1:
                        pairs = np.asarray(nt, dtype=np.int64).reshape(-1, 2)
                        for p in pairs:
                            a, b = tag_to_idx[int(p[0])], tag_to_idx[int(p[1])]
                            edge_phys[(min(a, b), max(a, b))] = phys_name

        return unique_nodes(coords, triangles, edge_phys)
    finally:
        gmsh.finalize()


def gmsh_mesh_amr(
    geometry_builder: Callable[[], dict[int, str]],
    size_callback: Callable[[float, float], float],
    *,
    name: str = "lb_adaptive",
    quiet: bool = True,
    h_floor: float = 1e-4,
    h_ceiling: float = 1.0,
) -> LBMesh:
    gmsh.initialize()
    try:
        if quiet:
            gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_floor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_ceiling)

        phys_lookup = geometry_builder()

        def _cb(dim, tag, x, y, z, lc):
            try:
                return float(np.clip(size_callback(x, y), h_floor, h_ceiling))
            except Exception:
                return float(h_ceiling)

        gmsh.model.mesh.setSizeCallback(_cb)
        gmsh.model.mesh.generate(2)

        return _extract_lbmesh(phys_lookup)
    finally:
        gmsh.finalize()


 

def _leb_bisect_marked(mesh: LBMesh, marked) -> LBMesh:
    edge_phys = _edge_phys_from_mesh(mesh)
    coords, triangles, new_phys = longest_edge_bisection(
        mesh.nodes_orig, mesh.elements_orig, edge_phys, marked,
    )
    return unique_nodes(coords, triangles, new_phys)


def leb_mesh_uniform(mesh: LBMesh, growth_factor: float) -> LBMesh:
    p = mesh.nodes[mesh.elements]
    areas = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    n_mark = int(round(mesh.n_tri * (float(growth_factor) - 1.0)))
    n_mark = max(1, min(n_mark, mesh.n_tri))
    _idx = np.arange(areas.size, dtype=np.int64)
    marked = np.lexsort((_idx, -areas))[:n_mark]
    return _leb_bisect_marked(mesh, marked)


def gmsh_mesh_uniform(
    geometry_builder: Callable[[], dict[int, str]],
    target_n: int,
) -> LBMesh:
    return mesh_parameter(geometry_builder, target_n=target_n)


# --- Bisection kernel ---

def _split_edge_phys(edge_phys: dict, e: tuple[int, int], m: int) -> None:
    tag = edge_phys.pop(e, None)
    if tag is None:
        return
    u, v = e
    edge_phys[(u, m) if u < m else (m, u)] = tag
    edge_phys[(m, v) if m < v else (v, m)] = tag


def longest_edge_bisection(
    coords: np.ndarray,
    triangles: np.ndarray,
    edge_phys: dict[tuple[int, int], str],
    marked,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], str]]:
    coords = [np.asarray(c, dtype=float).copy()
              for c in np.asarray(coords, dtype=float)]
    triangles = np.asarray(triangles)
    tris: dict[int, tuple[int, int, int]] = {
        i: (int(t[0]), int(t[1]), int(t[2])) for i, t in enumerate(triangles)
    }
    edge_phys = {(int(a), int(b)): t for (a, b), t in edge_phys.items()}

    def ekey(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    edge2tris: dict[tuple[int, int], list[int]] = {}
    for tid, (a, b, c) in tris.items():
        for e in (ekey(a, b), ekey(b, c), ekey(c, a)):
            edge2tris.setdefault(e, []).append(tid)

    def elen2(e: tuple[int, int]) -> float:
        d = coords[e[0]] - coords[e[1]]
        return float(d[0] * d[0] + d[1] * d[1])

    def longest(tid: int) -> tuple[int, int]:
        a, b, c = tris[tid]
        return max((ekey(a, b), ekey(b, c), ekey(c, a)),
                   key=lambda e: (elen2(e), e))

    def neighbor(tid: int, e: tuple[int, int]) -> int | None:
        for t in edge2tris.get(e, ()):
            if t != tid:
                return t
        return None

    counter = [len(tris)]

    def remove_tri(tid: int) -> None:
        a, b, c = tris.pop(tid)
        for e in (ekey(a, b), ekey(b, c), ekey(c, a)):
            lst = edge2tris.get(e)
            if lst is not None:
                lst.remove(tid)
                if not lst:
                    del edge2tris[e]

    def add_tri(a: int, b: int, c: int) -> int:
        tid = counter[0]
        counter[0] += 1
        tris[tid] = (a, b, c)
        for e in (ekey(a, b), ekey(b, c), ekey(c, a)):
            edge2tris.setdefault(e, []).append(tid)
        return tid

    def bisect(tid: int, e: tuple[int, int], m: int) -> None:
        v = tris[tid]
        p = q = r = -1
        for i in range(3):
            p, q, r = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
            if ekey(p, q) == e:
                break
        remove_tri(tid)
        add_tri(p, m, r)
        add_tri(m, q, r)

    def midpoint(e: tuple[int, int]) -> int:
        m = len(coords)
        coords.append(0.5 * (coords[e[0]] + coords[e[1]]))
        return m

    stack = [int(t) for t in np.asarray(marked, dtype=np.int64).ravel()]
    max_ops = 200 * len(tris) + 10000
    ops = 0
    while stack:
        ops += 1
        if ops > max_ops:
            raise RuntimeError(
                "longest_edge_bisection did not terminate "
                "(mesh topology may be corrupt)"
            )
        tid = stack[-1]
        if tid not in tris:
            stack.pop()
            continue
        e = longest(tid)
        nb = neighbor(tid, e)
        if nb is None:
            m = midpoint(e)
            _split_edge_phys(edge_phys, e, m)
            bisect(tid, e, m)
            stack.pop()
        elif longest(nb) == e:
            m = midpoint(e)
            _split_edge_phys(edge_phys, e, m)
            bisect(tid, e, m)
            bisect(nb, e, m)
            stack.pop()
        else:
            stack.append(nb)

    new_coords = np.array(coords, dtype=float)
    new_triangles = np.array([tris[t] for t in sorted(tris)], dtype=np.int64)
    return new_coords, new_triangles, edge_phys


# --- LBMesh adapter ---

def _edge_phys_from_mesh(mesh: LBMesh) -> dict[tuple[int, int], str]:
    edge_phys: dict[tuple[int, int], str] = {}
    for ib in range(mesh.boundary_edges.shape[0]):
        _, na_dup, nb_dup = (int(x) for x in mesh.boundary_edges[ib])
        oa = int(mesh.dup_to_orig[na_dup])
        ob = int(mesh.dup_to_orig[nb_dup])
        edge_phys[(min(oa, ob), max(oa, ob))] = mesh.boundary_edge_tag[ib]
    return edge_phys





# ─── (6) DEFINE geometry (gmsh builder closures) ──────────────────────────────

def strip_footing_geometry(B: float = 1.0, W: float = 6.0, D: float = 3.0) -> Callable[[], dict[int, str]]:

    def builder() -> dict[int, str]:
        TL = gmsh.model.geo.addPoint(-W / 2, 0.0, 0.0)
        FL = gmsh.model.geo.addPoint(-B / 2, 0.0, 0.0)
        FR = gmsh.model.geo.addPoint(+B / 2, 0.0, 0.0)
        TR = gmsh.model.geo.addPoint(+W / 2, 0.0, 0.0)
        BR = gmsh.model.geo.addPoint(+W / 2, -D, 0.0)
        BL = gmsh.model.geo.addPoint(-W / 2, -D, 0.0)

        l_free_left = gmsh.model.geo.addLine(TL, FL)
        l_footing = gmsh.model.geo.addLine(FL, FR)
        l_free_right = gmsh.model.geo.addLine(FR, TR)
        l_right = gmsh.model.geo.addLine(TR, BR)
        l_bottom = gmsh.model.geo.addLine(BR, BL)
        l_left = gmsh.model.geo.addLine(BL, TL)

        loop = gmsh.model.geo.addCurveLoop(
            [l_free_left, l_footing, l_free_right, l_right, l_bottom, l_left]
        )
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        free_tag = gmsh.model.addPhysicalGroup(1, [l_free_left, l_free_right], name="free")
        footing_tag = gmsh.model.addPhysicalGroup(1, [l_footing], name="footing")
        right_tag = gmsh.model.addPhysicalGroup(1, [l_right], name="support_right")
        bot_tag = gmsh.model.addPhysicalGroup(1, [l_bottom], name="support_bottom")
        left_tag = gmsh.model.addPhysicalGroup(1, [l_left], name="support_left")

        return {
            free_tag: "free",
            footing_tag: "footing",
            right_tag: "support_right",
            bot_tag: "support_bottom",
            left_tag: "support_left",
        }

    return builder


def strip_footing_geometry_half(B: float = 1.0, W: float = 5.0, D: float = 2.5) -> Callable[[], dict[int, str]]:
    half_W = 0.5 * W

    def builder() -> dict[int, str]:
        TL = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        FR = gmsh.model.geo.addPoint(B / 2.0, 0.0, 0.0)
        TR = gmsh.model.geo.addPoint(half_W, 0.0, 0.0)
        BR = gmsh.model.geo.addPoint(half_W, -D, 0.0)
        BL = gmsh.model.geo.addPoint(0.0, -D, 0.0)

        l_footing = gmsh.model.geo.addLine(TL, FR)
        l_free = gmsh.model.geo.addLine(FR, TR)
        l_right = gmsh.model.geo.addLine(TR, BR)
        l_bottom = gmsh.model.geo.addLine(BR, BL)
        l_symmetry = gmsh.model.geo.addLine(BL, TL)

        loop = gmsh.model.geo.addCurveLoop([l_footing, l_free, l_right, l_bottom, l_symmetry])
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        free_tag = gmsh.model.addPhysicalGroup(1, [l_free], name="free")
        footing_tag = gmsh.model.addPhysicalGroup(1, [l_footing], name="footing")
        right_tag = gmsh.model.addPhysicalGroup(1, [l_right], name="support_right")
        bot_tag = gmsh.model.addPhysicalGroup(1, [l_bottom], name="support_bottom")
        sym_tag = gmsh.model.addPhysicalGroup(1, [l_symmetry], name="symmetry")

        return {
            free_tag: "free",
            footing_tag: "footing",
            right_tag: "support_right",
            bot_tag: "support_bottom",
            sym_tag: "symmetry",
        }

    return builder





def geometry_from_config(geo: dict) -> Callable[[], dict[int, str]]:
    kind = geo["kind"]
    
    if kind == "strip_footing":
        if bool(geo.get("mirrored", False)):
            return strip_footing_geometry(
                B=float(geo["B"]), W=float(geo["W"]), D=float(geo["D"]),
            )
        return strip_footing_geometry_half(
            B=float(geo["B"]), W=float(geo["W"]), D=float(geo["D"]),
        )
    raise ValueError(
        f"unknown geometry kind {kind!r}; use strip_footing"
    )
