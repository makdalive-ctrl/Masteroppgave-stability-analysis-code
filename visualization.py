from __future__ import annotations

import atexit
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection
from amr import _element_gradient
from mesh import LBMesh
if TYPE_CHECKING:
    from amr import AdaptiveStep
    from optimizer import LBSolution




class _TeeStream:
    

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def start_log_capture(path: str) -> Path:
   
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fp = open(p, "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _TeeStream(original, fp)

    def _restore():
        sys.stdout = original
        try:
            fp.close()
        except Exception:
            pass

    atexit.register(_restore)
    print(f"  log     -> {p}")
    return p




def get_next_available_filename(directory: str, base_filename: str, extension: str) -> str:
    directory = str(directory)
    pattern = re.compile(rf"^{re.escape(base_filename)}_(\d+){re.escape(extension)}$")
    numbers: list[int] = []
    if os.path.isdir(directory):
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                numbers.append(int(m.group(1)))
    next_number = max(numbers) + 1 if numbers else 1
    return str(Path(directory) / f"{base_filename}_{next_number}{extension}")




_REAL_BOUNDARY_TAGS = frozenset({
    "footing", "free", "support_bottom", "support_left",
    "support_right", "symmetry",
})


def _continuous_nodal(mesh: LBMesh, values: np.ndarray) -> np.ndarray:
    
    flat_vals = np.asarray(values, dtype=float).reshape(-1)
    flat_dup = np.asarray(mesh.elements, dtype=int).reshape(-1)
    flat_orig = mesh.dup_to_orig[flat_dup]
    n_orig = mesh.nodes_orig.shape[0]
    nodal_sum = np.zeros(n_orig)
    nodal_count = np.zeros(n_orig)
    np.add.at(nodal_sum, flat_orig, flat_vals)
    np.add.at(nodal_count, flat_orig, 1)
    return nodal_sum / np.maximum(nodal_count, 1)


def _domain_outline(mesh: LBMesh) -> list[np.ndarray]:
    
    segs: list[np.ndarray] = []
    for ib in range(mesh.n_boundary_edges):
        if mesh.boundary_edge_tag[ib] not in _REAL_BOUNDARY_TAGS:
            continue
        _, na, nb = (int(x) for x in mesh.boundary_edges[ib])
        segs.append(np.stack([mesh.nodes[na], mesh.nodes[nb]]))
    return segs


def stress_localization(mesh: LBMesh, sigma: np.ndarray) -> np.ndarray:
    
    grad_sxx = _element_gradient(mesh, sigma[:, :, 0])
    grad_syy = _element_gradient(mesh, sigma[:, :, 1])
    grad_sxy = _element_gradient(mesh, sigma[:, :, 2])
    g2 = (grad_sxx[:, 0] ** 2 + grad_sxx[:, 1] ** 2
          + grad_syy[:, 0] ** 2 + grad_syy[:, 1] ** 2
          + 2.0 * (grad_sxy[:, 0] ** 2 + grad_sxy[:, 1] ** 2))
    return np.sqrt(g2)


def _stress_invariants(sxx, syy, sxy):
   
    half_sum = 0.5 * (sxx + syy)
    half_diff = 0.5 * (sxx - syy)
    radius = np.sqrt(half_diff ** 2 + sxy ** 2)
    return half_sum + radius, half_sum - radius, half_sum, radius


def _compressive_components(sigma: np.ndarray):
    
    return (
        np.maximum(-sigma[..., 0], 0.0),
        np.maximum(-sigma[..., 1], 0.0),
        np.abs(sigma[..., 2]),
    )


def _setup_axes(ax, mesh: LBMesh, *, pad_factor: float = 0.05,
                hide_chrome: bool = False) -> None:
    
    pad = pad_factor * (mesh.nodes.max() - mesh.nodes.min())
    ax.set_xlim(mesh.nodes[:, 0].min() - pad, mesh.nodes[:, 0].max() + pad)
    ax.set_ylim(mesh.nodes[:, 1].min() - pad, mesh.nodes[:, 1].max() + pad)
    ax.set_aspect("equal")
    if hide_chrome:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)




def plot_mesh(mesh: LBMesh, ax=None, edgecolor: str = "black",
              linewidth: float = 0.4, fill: bool = False):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    polys = [mesh.nodes[tri] for tri in mesh.elements]
    pc = PolyCollection(polys, facecolors="none" if not fill else "lightgray",
                        edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection(pc)
    _setup_axes(ax, mesh, hide_chrome=True)
    return ax


def plot_element_field(mesh: LBMesh, values: np.ndarray, *, ax=None,
                       cmap: str = "viridis", title: str | None = None,
                       draw_mesh: bool = True):
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    polys = [mesh.nodes[tri] for tri in mesh.elements]
    pc = PolyCollection(polys, array=np.asarray(values), cmap=cmap,
                        edgecolors="black" if draw_mesh else "none",
                        linewidths=0.2 if draw_mesh else 0.0)
    ax.add_collection(pc)
    _setup_axes(ax, mesh)
    plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
    if title:
        ax.set_title(title)
    return ax


def plot_nodal_field(mesh: LBMesh, values: np.ndarray, *, ax=None,
                     cmap: str = "viridis", title: str | None = None,
                     draw_mesh: bool = False, vmin: float | None = None,
                     vmax: float | None = None):
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    n_tri = mesh.n_tri
    coords_dup = mesh.nodes[mesh.elements].reshape(-1, 2)
    tris_dup = np.arange(3 * n_tri).reshape(n_tri, 3)
    triangulation = mtri.Triangulation(coords_dup[:, 0], coords_dup[:, 1], tris_dup)
    flat_values = np.asarray(values).reshape(-1)
    if vmin is None:
        vmin = float(flat_values.min())
    if vmax is None:
        vmax = float(flat_values.max())
    tpc = ax.tripcolor(triangulation, flat_values, shading="gouraud", cmap=cmap,
                       vmin=vmin, vmax=vmax)
    if draw_mesh:
        ax.triplot(mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements),
                   color="black", linewidth=0.15, alpha=0.4)
    _setup_axes(ax, mesh)
    plt.colorbar(tpc, ax=ax, fraction=0.04, pad=0.02)
    if title:
        ax.set_title(title)
    return ax


def plot_banded_contour(mesh: LBMesh, values: np.ndarray, *, ax=None,
                        cmap: str = "turbo", levels: int = 16,
                        title: str | None = None, units: str | None = None,
                        vmin: float | None = None, vmax: float | None = None):
   
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    nodal = _continuous_nodal(mesh, values)
    if vmin is None:
        vmin = float(nodal.min())
    if vmax is None:
        vmax = float(nodal.max())
    if vmax <= vmin:
        vmax = vmin + 1e-12

    triangulation = mtri.Triangulation(
        mesh.nodes_orig[:, 0], mesh.nodes_orig[:, 1], mesh.elements_orig
    )
    band_edges = np.linspace(vmin, vmax, levels + 1)
    cmap_obj = plt.get_cmap(cmap, levels)
    cf = ax.tricontourf(triangulation, nodal, levels=band_edges, cmap=cmap_obj,
                        vmin=vmin, vmax=vmax, extend="neither")

    for seg in _domain_outline(mesh):
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=0.9)

    _setup_axes(ax, mesh, pad_factor=0.03, hide_chrome=True)

    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, ticks=band_edges)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)
    if units:
        cb.set_label(units, fontsize=9)

    head = title or ""
    extras = f"min = {nodal.min():.3f}    max = {nodal.max():.3f}"
    full = f"{head}\n{extras}" if head else extras
    ax.set_title(full, fontsize=11)
    return ax




def plot_solution_summary(
    mesh: LBMesh,
    sol: "LBSolution",
    *,
    label: str,
    plastic_cmap: str = "turbo",
    alpha_fmt: str = ".3f",
    save_dir: str | None = None,
    fig_format: str = "png",
) -> None:
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_mesh(mesh, ax=axes[0, 0])
    axes[0, 0].set_title(f"mesh  (NE = {mesh.n_tri})")

    sxx_disp, syy_disp, sxy_abs = _compressive_components(sol.sigma)
    plot_banded_contour(mesh, sxx_disp, ax=axes[0, 1], cmap="turbo", vmin=0.0,
                        title=r"Horizontal stress  $\sigma_x$  (comp. +)")
    plot_banded_contour(mesh, syy_disp, ax=axes[0, 2], cmap="turbo", vmin=0.0,
                        title=r"Vertical stress  $\sigma_y$  (comp. +)")
    plot_banded_contour(mesh, sxy_abs, ax=axes[1, 0], cmap="turbo", vmin=0.0,
                        title=r"Shear stress  $|\tau_{xy}|$")
    plot_banded_contour(mesh, sol.lagrange_multiplier, ax=axes[1, 1],
                        cmap=plastic_cmap, vmin=0.0,
                        title=r"Lagrange multiplier  $L$")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.05, 0.85,
        f"{label}\n\nNE = {mesh.n_tri}\nalpha = {sol.alpha:{alpha_fmt}}\n"
        f"status: {sol.status}",
        transform=axes[1, 2].transAxes,
        fontsize=12, va="top", family="monospace",
    )

    fig.suptitle(label, fontsize=11)
    plt.tight_layout()

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{label}.{fig_format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  figure  -> {path}")

    plt.show()


def plot_amr_history(
    history: list[AdaptiveStep],
    *,
    label: str,
    reference_value: float,
    reference_label: str = "Nc",
    fmt: str = ".3f",
    adaptive_fan: bool = False,
    plastic_cmap: str = "turbo",
    gradient_cmap: str = "turbo",
    save_dir: str | None = None,
    fig_format: str = "png",
) -> None:
    
    n = len(history)
    if n == 0:
        return
    panel_w, panel_h = 7.0, 3.8

    def qc(step: AdaptiveStep) -> str:
        if reference_value > 0:
            return f"{reference_label} = {step.solution.alpha / reference_value:{fmt}}"
        return rf"$\alpha$ = {step.solution.alpha:{fmt}}"

    # Page 1: meshes
    fig1, axes1 = plt.subplots(1, n, figsize=(panel_w * n, panel_h + 1.0))
    if n == 1:
        axes1 = np.array([axes1])
    for k, step in enumerate(history):
        plot_mesh(step.mesh, ax=axes1[k])
        kind = "uniform" if step.iteration == 0 else f"AMR iter {step.iteration}"
        axes1[k].set_title(
            f"{kind}\nNE = {step.mesh.n_tri},  {qc(step)}", fontsize=11
        )
    fig1.suptitle(f"Page 1 - meshes\n{label}", fontsize=12, y=0.995)
    fig1.tight_layout(rect=(0, 0, 1, 0.92))

   
    if adaptive_fan:
        fig2, axes2 = plt.subplots(2, n, figsize=(panel_w * n, 2 * panel_h + 1.5))
        if n == 1:
            axes2 = axes2.reshape(2, 1)
        for k, step in enumerate(history):
            g_elem = stress_localization(step.mesh, step.solution.sigma)
            g_field = np.broadcast_to(g_elem[:, None], (g_elem.size, 3))
            plot_banded_contour(
                step.mesh, g_field, ax=axes2[0, k],
                cmap=gradient_cmap, vmin=0.0,
                title=f"Stress localization  (iter {step.iteration}, NE = {step.mesh.n_tri})",
            )
            plot_banded_contour(
                step.mesh, step.solution.lagrange_multiplier, ax=axes2[1, k],
                cmap=plastic_cmap, vmin=0.0,
                title=rf"Lagrange multiplier $L$  (iter {step.iteration})",
            )
        fig2.suptitle(
            f"Page 2 - Stress localization and Lagrange multiplier\n{label}",
            fontsize=12, y=0.995,
        )
        fig2.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig2, axes2 = plt.subplots(1, n, figsize=(panel_w * n, panel_h + 1.0))
        if n == 1:
            axes2 = np.array([axes2])
        for k, step in enumerate(history):
            plot_banded_contour(
                step.mesh, step.solution.lagrange_multiplier, ax=axes2[k],
                cmap=plastic_cmap, vmin=0.0,
                title=rf"Lagrange multiplier $L$  (iter {step.iteration}, NE = {step.mesh.n_tri})",
            )
        fig2.suptitle(f"Page 2 - Lagrange multiplier\n{label}", fontsize=12, y=0.995)
        fig2.tight_layout(rect=(0, 0, 1, 0.92))

    
    fig3, axes3 = plt.subplots(3, n, figsize=(panel_w * n, 3 * panel_h + 2.0))
    if n == 1:
        axes3 = axes3.reshape(3, 1)
    for k, step in enumerate(history):
        sxx_disp, syy_disp, sxy_abs = _compressive_components(step.solution.sigma)
        plot_banded_contour(step.mesh, sxx_disp, ax=axes3[0, k], cmap="turbo", vmin=0.0,
                            title=rf"Horizontal stress  $\sigma_x$  (comp. +)  iter {step.iteration}")
        plot_banded_contour(step.mesh, syy_disp, ax=axes3[1, k], cmap="turbo", vmin=0.0,
                            title=rf"Vertical stress  $\sigma_y$  (comp. +)  iter {step.iteration}")
        plot_banded_contour(step.mesh, sxy_abs, ax=axes3[2, k], cmap="turbo", vmin=0.0,
                            title=rf"Shear stress  $|\tau_{{xy}}|$  iter {step.iteration}")
    fig3.suptitle(
        f"Page 3 - stresses: $\\sigma_x$, $\\sigma_y$, $|\\tau_{{xy}}|$\n{label}",
        fontsize=12, y=0.995,
    )
    fig3.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k, fig in enumerate((fig1, fig2, fig3), start=1):
            path = out / f"{label}_amr_page{k}.{fig_format}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  figure  -> {path}")

    plt.show()




def make_VTK(
    mesh: LBMesh,
    solution: "LBSolution",
    *,
    path: str,
) -> None:
    p_out = Path(path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = mesh.n_nodes
    n_tri = mesh.n_tri

    sxx_node = solution.sigma[:, :, 0].reshape(-1)
    syy_node = solution.sigma[:, :, 1].reshape(-1)
    sxy_node = solution.sigma[:, :, 2].reshape(-1)
    lam_node = solution.lagrange_multiplier.reshape(-1)

    s1, s2, p_node, q_node = _stress_invariants(sxx_node, syy_node, sxy_node)

    coords3 = np.column_stack([mesh.nodes, np.zeros(n_nodes)])

    with p_out.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"lower_bound_lim_an  alpha={solution.alpha:.6f}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {n_nodes} float\n")
        for x, y, z in coords3:
            f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

        f.write(f"\nCELLS {n_tri} {4 * n_tri}\n")
        for tri in mesh.elements:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

        f.write(f"\nCELL_TYPES {n_tri}\n")
        for _ in range(n_tri):
            f.write("5\n")

        f.write(f"\nPOINT_DATA {n_nodes}\n")
        for name, arr in (
            ("sigma_x", sxx_node),
            ("sigma_y", syy_node),
            ("tau_xy", sxy_node),
            ("lagrange_multiplier", lam_node),
            ("sigma_1", s1),
            ("sigma_2", s2),
            ("p_mean", p_node),
            ("q_dev", q_node),
        ):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in arr:
                f.write(f"{float(v):.8e}\n")


def make_POS(
    mesh: LBMesh,
    solution: "LBSolution",
    *,
    path: str,
) -> None:
    p_out = Path(path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    sxx = solution.sigma[:, :, 0]
    syy = solution.sigma[:, :, 1]
    sxy = solution.sigma[:, :, 2]
    lam = solution.lagrange_multiplier

    s1, s2, p_field, q_field = _stress_invariants(sxx, syy, sxy)

    fields = (
        ("sigma_x",            sxx),
        ("sigma_y",            syy),
        ("tau_xy",            sxy),
        ("lagrange_multiplier",  lam),
        ("sigma_1",             s1),
        ("sigma_2",             s2),
        ("p_mean",              p_field),
        ("q_dev",               q_field),
    )

    coords = mesh.nodes
    tri = mesh.elements

    with p_out.open("w", encoding="utf-8") as f:
        for name, vals in fields:
            f.write(f'View "{name}" {{\n')
            for e in range(mesh.n_tri):
                ids = tri[e]
                p0 = coords[ids[0]]
                p1 = coords[ids[1]]
                p2 = coords[ids[2]]
                v0, v1, v2 = float(vals[e, 0]), float(vals[e, 1]), float(vals[e, 2])
                f.write(
                    f"ST({p0[0]:.8e},{p0[1]:.8e},0,"
                    f"{p1[0]:.8e},{p1[1]:.8e},0,"
                    f"{p2[0]:.8e},{p2[1]:.8e},0)"
                    f"{{{v0:.8e},{v1:.8e},{v2:.8e}}};\n"
                )
            f.write("};\n")
