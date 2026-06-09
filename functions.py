from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from mesh import LBMesh

if TYPE_CHECKING:
    from optimizer import TractionBC


# ─── (1) GEOMETRIC primitives ─────────────────────────────────────────────────

def _signed_double_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _edge_normal_for_tri(mesh: LBMesh, e: int, na: int, nb: int) -> np.ndarray:
    tri = mesh.elements[e]
    third = next(int(n) for n in tri if int(n) != int(na) and int(n) != int(nb))
    pa = mesh.nodes[na]
    pb = mesh.nodes[nb]
    pc = mesh.nodes[third]
    edge = pb - pa
    n_candidate = np.array([edge[1], -edge[0]])
    if np.dot(n_candidate, pc - 0.5 * (pa + pb)) > 0:
        n_candidate = -n_candidate
    return n_candidate / np.linalg.norm(n_candidate)


def T_transformation(nx: float, ny: float) -> np.ndarray:
    return np.array([
        [nx * nx,  ny * ny,  2.0 * nx * ny],
        [-nx * ny, nx * ny,  nx * nx - ny * ny],
    ], dtype=float)


# ─── (2) BUILD LP equality blocks (equilibrium, discontinuity, BCs) ───────────

def A_equilibrium(
    mesh: LBMesh,
    *,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    n_tri = mesh.n_tri
    n_dof = 3 * mesh.n_nodes

    row_idx = np.empty(12 * n_tri, dtype=np.int64)
    col_idx = np.empty(12 * n_tri, dtype=np.int64)
    val = np.empty(12 * n_tri, dtype=np.float64)
    b_fixed = np.zeros(2 * n_tri)
    b_alpha = np.zeros(2 * n_tri)

    bf_x_fix, bf_y_fix = float(fixed_body_force[0]), float(fixed_body_force[1])
    bf_x_scl, bf_y_scl = float(scaled_body_force[0]), float(scaled_body_force[1])

    k = 0
    for e in range(n_tri):
        tri = mesh.elements[e]
        p = mesh.nodes[tri]
        x1, y1 = p[0]
        x2, y2 = p[1]
        x3, y3 = p[2]
        twoA = _signed_double_area(p[0], p[1], p[2])
        eta = (y2 - y3, y3 - y1, y1 - y2)
        zeta = (x3 - x2, x1 - x3, x2 - x1)

        for li in range(3):
            nid = int(tri[li])
            row_idx[k] = 2 * e + 0
            col_idx[k] = 3 * nid + 0
            val[k] = eta[li]
            k += 1
            row_idx[k] = 2 * e + 0
            col_idx[k] = 3 * nid + 2
            val[k] = zeta[li]
            k += 1
            row_idx[k] = 2 * e + 1
            col_idx[k] = 3 * nid + 1
            val[k] = zeta[li]
            k += 1
            row_idx[k] = 2 * e + 1
            col_idx[k] = 3 * nid + 2
            val[k] = eta[li]
            k += 1

        b_fixed[2 * e + 0] = -twoA * bf_x_fix
        b_fixed[2 * e + 1] = -twoA * bf_y_fix
        b_alpha[2 * e + 0] = -twoA * bf_x_scl
        b_alpha[2 * e + 1] = -twoA * bf_y_scl

    A = sp.csr_matrix(
        (val[:k], (row_idx[:k], col_idx[:k])),
        shape=(2 * n_tri, n_dof),
    )
    return A, b_fixed, b_alpha


def A_discontinuity(mesh: LBMesh) -> tuple[sp.csr_matrix, np.ndarray]:
    n_internal = mesh.n_internal_edges
    n_dof = 3 * mesh.n_nodes

    nnz_max = 24 * n_internal
    row_idx = np.empty(nnz_max, dtype=np.int64)
    col_idx = np.empty(nnz_max, dtype=np.int64)
    val = np.empty(nnz_max, dtype=np.float64)

    k = 0
    for ie in range(n_internal):
        e1, _, na1, na2, nb1, nb2 = (int(x) for x in mesh.internal_edges[ie])
        n_e = _edge_normal_for_tri(mesh, e1, na1, nb1)
        T = T_transformation(float(n_e[0]), float(n_e[1]))

        for ep_idx, (n_lhs, n_rhs) in enumerate(((na1, na2), (nb1, nb2))):
            base_row = 4 * ie + 2 * ep_idx
            for trow in range(2):
                r = base_row + trow
                for cl in range(3):
                    row_idx[k] = r; col_idx[k] = 3 * n_lhs + cl; val[k] = T[trow, cl]; k += 1
                    row_idx[k] = r; col_idx[k] = 3 * n_rhs + cl; val[k] = -T[trow, cl]; k += 1

    A = sp.csr_matrix(
        (val[:k], (row_idx[:k], col_idx[:k])),
        shape=(4 * n_internal, n_dof),
    )
    b = np.zeros(4 * n_internal)
    return A, b


def boundaries_condition(
    mesh: LBMesh,
    boundary_conditions: Sequence["TractionBC"] = (),
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    from optimizer import TractionBC
    bc_map = {bc.boundary: bc for bc in boundary_conditions if isinstance(bc, TractionBC)}
    n_dof = 3 * mesh.n_nodes

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_fixed_list: list[float] = []
    b_alpha_list: list[float] = []
    next_row = 0

    for ib in range(mesh.n_boundary_edges):
        e, na, nb = (int(x) for x in mesh.boundary_edges[ib])
        tag = mesh.boundary_edge_tag[ib]
        if tag not in bc_map:
            continue
        bc = bc_map[tag]
        n_e = _edge_normal_for_tri(mesh, e, na, nb)
        nx, ny = float(n_e[0]), float(n_e[1])
        tx_fix, ty_fix = float(bc.fixed[0]), float(bc.fixed[1])
        tx_scl, ty_scl = float(bc.scaled[0]), float(bc.scaled[1])

        if bc.constrain_x and bc.constrain_y:
            T = T_transformation(nx, ny)
            sigma_n_fix = nx * tx_fix + ny * ty_fix
            tau_fix = -ny * tx_fix + nx * ty_fix
            sigma_n_scl = nx * tx_scl + ny * ty_scl
            tau_scl = -ny * tx_scl + nx * ty_scl
            for nid in (na, nb):
                for cl in range(3):
                    rows.append(next_row); cols.append(3 * nid + cl); vals.append(T[0, cl])
                b_fixed_list.append(sigma_n_fix); b_alpha_list.append(sigma_n_scl)
                next_row += 1
                for cl in range(3):
                    rows.append(next_row); cols.append(3 * nid + cl); vals.append(T[1, cl])
                b_fixed_list.append(tau_fix); b_alpha_list.append(tau_scl)
                next_row += 1
        elif bc.constrain_x:
            for nid in (na, nb):
                rows.append(next_row); cols.append(3 * nid + 0); vals.append(nx)
                rows.append(next_row); cols.append(3 * nid + 2); vals.append(ny)
                b_fixed_list.append(tx_fix); b_alpha_list.append(tx_scl)
                next_row += 1
        elif bc.constrain_y:
            for nid in (na, nb):
                rows.append(next_row); cols.append(3 * nid + 2); vals.append(nx)
                rows.append(next_row); cols.append(3 * nid + 1); vals.append(ny)
                b_fixed_list.append(ty_fix); b_alpha_list.append(ty_scl)
                next_row += 1

    A = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(next_row, n_dof),
    )
    b_fixed = np.asarray(b_fixed_list, dtype=float)
    b_alpha = np.asarray(b_alpha_list, dtype=float)
    return A, b_fixed, b_alpha


# ─── (3) DEFINE yield criterion ───────────────────────────────────────────────

@dataclass
class YieldCriterion:
    name: str
    cohesion: float
    friction_angle_deg: float = 0.0

    @property
    def D(self) -> np.ndarray:
        phi = np.deg2rad(self.friction_angle_deg)
        return np.array([
            [-np.sin(phi), -np.sin(phi),  0.0],
            [         1.0,         -1.0,  0.0],
            [         0.0,          0.0,  2.0],
        ])

    @property
    def d(self) -> np.ndarray:
        phi = np.deg2rad(self.friction_angle_deg)
        return np.array([2.0 * self.cohesion * np.cos(phi), 0.0, 0.0])


def tresca(cohesion: float) -> YieldCriterion:
    return YieldCriterion(name="Tresca", cohesion=cohesion, friction_angle_deg=0.0)


def mohr_coulomb(cohesion: float, friction_angle_deg: float) -> YieldCriterion:
    return YieldCriterion(
        name="Mohr-Coulomb",
        cohesion=cohesion,
        friction_angle_deg=friction_angle_deg,
    )


def cohesion_from_param(param: dict, fi_deg: float, *, verbose: bool = True) -> float:
    raw = param.get("cohesion")
    if raw is None:
        raw = param.get("su")

    if isinstance(raw, str):
        if raw.strip().lower() != "computed":
            raise ValueError(
                f'unknown cohesion sentinel {raw!r}; use a number or "computed"'
            )
        attrac = param.get("attrac")
        if attrac is None:
            raise ValueError(
                'cohesion="computed" requires `attrac` to be defined in param'
            )
        if fi_deg == 0.0:
            raise ValueError(
                'cohesion="computed" requires fi > 0 '
                "(Tresca cannot derive cohesion from attraction)"
            )
        c = float(attrac) * np.tan(np.radians(fi_deg))
        if verbose:
            print(f"  cohesion (computed) = attrac * tan(fi) "
                  f"= {attrac} * tan({fi_deg}) = {c:.4f}")
        return c

    if raw is None:
        raise ValueError(
            "no strength in param; give a number under \"cohesion\" or \"su\", "
            "or the string \"computed\""
        )

    return float(raw)


def from_config(param: dict) -> YieldCriterion:
    name = param.get("yield_criterion", "tresca").lower()
    fi = float(param.get("fi", 0.0))
    c = cohesion_from_param(param, fi)
    if name == "tresca":
        return tresca(cohesion=c)
    if name == "mohr_coulomb":
        return mohr_coulomb(cohesion=c, friction_angle_deg=fi)
    raise ValueError(f"unknown yield_criterion {name!r}; use tresca or mohr_coulomb")


# ─── (4) ASSEMBLE full system + objective ─────────────────────────────────────

def assembly_of_equality_constraints(
    mesh: LBMesh,
    *,
    fixed_body_force: tuple[float, float] = (0.0, 0.0),
    scaled_body_force: tuple[float, float] = (0.0, 0.0),
    boundary_conditions: Sequence["TractionBC"] = (),
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    A_e, b_e_fix, b_e_alpha = A_equilibrium(
        mesh,
        fixed_body_force=fixed_body_force,
        scaled_body_force=scaled_body_force,
    )
    A_d, b_d = A_discontinuity(mesh)
    A_b, b_b_fix, b_b_alpha = boundaries_condition(mesh, boundary_conditions)

    A = sp.vstack([A_e, A_d, A_b], format="csr")
    b_fixed = np.concatenate([b_e_fix, b_d, b_b_fix])
    b_alpha = np.concatenate([b_e_alpha, np.zeros_like(b_d), b_b_alpha])
    return A, b_fixed, b_alpha


def objective_function(alpha: cp.Variable) -> cp.Maximize:
    return cp.Maximize(alpha)
