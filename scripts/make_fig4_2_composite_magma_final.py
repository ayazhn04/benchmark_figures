#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.2 composite (magma, final)
Reference vs MicroLad vs SliceGAN -- three-phase NMC cathode 2D-to-3D reconstruction.

Manual fixed-position layout, same visual contract as Figure 4.1
(make_fig4_1_composite_magma_final.py): same cards, fonts, panel-label
style, and export block. No tight_layout / constrained_layout /
bbox_inches. Everything is derived from real volumes and real CSV tables,
with verified Table 4.5/4.6 values used only as an explicit, logged
fallback where the spec calls for it.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, to_hex
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG
# ============================================================================

# Project root is auto-detected from this file's location so the script keeps
# working no matter where the checkout lives, as long as it stays under
# <PROJECT>/scripts/make_fig4_2_composite_magma_final.py
PROJECT = Path(__file__).resolve().parents[1]

EVAL_DIR = PROJECT / "evaluation_4_2"
METRICS_DIR = PROJECT / "evaluation_4_2_results_all_metrics"
CLEAN_PACKAGE_DIR = PROJECT / "FINAL_SECTION_4_2_CLEAN_PACKAGE"

OUT = PROJECT / "paper_figures_4_2" / "figure_4_2_composite_magma_final"
TMP = OUT / "_render_cache"
LOGS = PROJECT / "logs"

STEM = "fig4_2_composite_magma_final"

# Internal folder / group keys stay unchanged; only the user-facing labels
# (LABELS, below) use the paper's model names.
GROUPS = ["real", "microlad", "slicegan"]

# Keywords used to *discover* each group's sample folder under EVAL_DIR.
# Folder names are searched robustly (case/format-insensitive substrings)
# rather than assumed exact, per the task's "search robustly" instruction.
GROUP_DIR_KEYWORDS = {
    "real": ["real"],
    "microlad": ["diffusion", "microlad"],
    "slicegan": ["slicegan", "gan"],
}

# Never use CBD-middle calibrated/sensitivity data for the main figure.
EXCLUDE_PATH_TOKENS = (
    "cbdmiddle", "cbd_middle", "calibrated", "comparisonwithcbdmiddle",
    "comparison_with_cbdmiddle",
)

VOLUME_EXTS = (".npy", ".tif", ".tiff")

# Cap on how many volumes per group are loaded for aggregate stats / curves.
# The Section 4.2 evaluation set is ~50 volumes per group; this cap keeps
# runtime bounded even if a folder holds more.
MAX_VOLUMES_PER_GROUP = 50

# Curve-computation fallback settings (used only when no official curve CSV
# can be resolved -- see CURVE SPECS below).
MAX_R_VOX = 30          # max lag/segment length for TPC and lineal-path curves
LOCAL_BLOCK = 8         # local active-fraction block size (8^3 non-overlapping)
LOCAL_HIST_BINS = 24

RENDER_PX = 1500        # off-screen render resolution
CANVAS_PX = 900         # final square canvas for each 3D cell

FIG_W, FIG_H = 17.8, 10.2

SHOW_CURVE_ERRORS = False  # panel-b relative-error corner notes: off by default

# ============================================================================
# 1b. SANITY-CHECK / FALLBACK CONSTANTS (verified Table 4.5 / 4.6 values)
# ============================================================================

# Table 4.5 -- phase fractions. Used only to sanity-check loaded data; the
# script raises if a group's measured mean is farther than PHASE_TOL from
# these in any phase.
TABLE45_PHASE_FRACTIONS = {
    "real":     {"pore": 0.4114, "active": 0.4593, "cbd": 0.1292},
    "microlad": {"pore": 0.4274, "active": 0.4682, "cbd": 0.1044},
    "slicegan": {"pore": 0.4013, "active": 0.4633, "cbd": 0.1354},
}
PHASE_TOL = 0.03

# Table 4.6 -- interface hierarchy densities. Used as the panel c-left
# fallback if no official per-group CSV can be resolved.
TABLE46_INTERFACE = {
    "real":     {"pore_active": 0.0000, "pore_cbd": 0.0677, "active_cbd": 0.0496, "tpc_proxy": 0.0448},
    "microlad": {"pore_active": 0.0331, "pore_cbd": 0.0185, "active_cbd": 0.0721, "tpc_proxy": 0.0942},
    "slicegan": {"pore_active": 0.0010, "pore_cbd": 0.0701, "active_cbd": 0.0500, "tpc_proxy": 0.0437},
}

# Table 4.6 -- active-domain continuity. CRITICAL: per the task's explicit
# warning, active connected-component count / Euler characteristic must
# never be recomputed ad hoc for panel c -- only official metrics or these
# verified table values are used. Euler characteristic is intentionally not
# plotted (kept in the written table only).
TABLE46_CONTINUITY = {
    "real":     {"components": 5.66,  "chord": 14.55},
    "microlad": {"components": 56.62, "chord": 7.91},
    "slicegan": {"components": 6.02,  "chord": 14.42},
}

INTERFACE_METRIC_ORDER = ["pore_active", "pore_cbd", "active_cbd", "tpc_proxy"]
INTERFACE_TICK_LABELS = ["Pore-active", "Pore-CBD", "Active-CBD", "TPC proxy"]

# ============================================================================
# 2. STYLE -- identical contract to Figure 4.1
# ============================================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BG = "#FBFAFE"
CARD_FACE = "#FFFFFF"
CARD_EDGE = "#E6DCEF"
GRID = "#EBE4F2"
SPINE = "#DCD2E8"
TEXT = "#1A0F2B"
SUBTEXT = "#59496A"

COLORS = {"real": "#8C93A1", "microlad": "#F2A93B", "slicegan": "#772A8E"}

# Visible model names used everywhere in the figure (legends, titles, labels).
# Internal keys ("real", "microlad", "slicegan") are unchanged and never
# shown; "diffusion_samples"/"gan_samples" folder names stay in logs only.
LABELS = {"real": "Reference", "microlad": "MicroLad", "slicegan": "SliceGAN"}

LINESTYLES = {"real": "-", "microlad": "-", "slicegan": (0, (5.5, 2.2))}
LINEWIDTHS = {"real": 1.85, "microlad": 1.85, "slicegan": 1.95}
TITLE_COLOR = {"real": TEXT, "microlad": COLORS["microlad"], "slicegan": COLORS["slicegan"]}
MARKERS = {"real": "o", "microlad": "D", "slicegan": "^"}

CARD_LW = 0.8
CELL_LW = 1.6

# Phase -> grayscale value (0=pore near-black, 1=active near-white, 2=CBD mid-gray)
PHASE_GRAY_VALUES = np.array([0.05, 0.94, 0.55])
PHASE_NAMES = ["Pore", "Active", "CBD"]

# ============================================================================
# 3. GEOMETRY (figure fractions) -- reused verbatim from Figure 4.1
# ============================================================================

card_a = [0.045, 0.365, 0.380, 0.570]
card_b = [0.445, 0.365, 0.510, 0.570]
card_c = [0.045, 0.012, 0.910, 0.290]

# Vertical clearance between a panel label's top-anchor and the top edge of
# the panel it names, identical for a), b), c) -- see Figure 4.1 for the
# reasoning (keeps each label visually attached to its own panel).
PANEL_LABEL_OFFSET = 0.030

assert abs(card_a[1] - card_b[1]) < 1e-12, "panel a/b y mismatch"
assert abs(card_a[3] - card_b[3]) < 1e-12, "panel a/b height mismatch"
assert abs(card_c[0] - card_a[0]) < 1e-12, "panel c left edge mismatch"
assert abs((card_c[0] + card_c[2]) - (card_b[0] + card_b[2])) < 1e-12, "panel c right edge mismatch"

# ============================================================================
# 4. GENERIC HELPERS -- reused verbatim from Figure 4.1
# ============================================================================


def log(*a):
    print(*a, flush=True)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_col(df: pd.DataFrame, candidates, what: str, required: bool = True):
    """Return the first candidate column present in df (case/format tolerant)."""
    lut = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in lut:
            return lut[key]
    msg = f"[cols] could not resolve '{what}' from {list(candidates)}"
    if required:
        log(msg)
        log("[cols] available columns:", list(df.columns))
        raise KeyError(msg)
    log(msg + "  -> skipped. Available:", list(df.columns))
    return None


def normalize_group(value) -> str | None:
    """Robust mapping of raw CSV/folder labels onto the internal group keys.

    real / reference / ground truth  -> real
    microlad / diffusion             -> microlad
    slicegan / gan                   -> slicegan  (checked before microlad so
                                         "slicegan" itself never matches "gan"
                                         incorrectly against microlad)
    """
    v = _norm(value)
    if not v:
        return None
    if "slicegan" in v or "gan" in v:
        return "slicegan"
    if "microlad" in v or "diffusion" in v or "diff" in v:
        return "microlad"
    if "real" in v or v.startswith("ref") or "ground" in v or v == "gt":
        return "real"
    return None


def _path_is_excluded(p: Path) -> bool:
    s = _norm(str(p))
    return any(tok in s for tok in EXCLUDE_PATH_TOKENS)


# ============================================================================
# 5. DATA DISCOVERY + THREE-PHASE VOLUME LOADING
# ============================================================================


def discover_sample_dir(group: str) -> Path:
    """Find the sample folder for a group under EVAL_DIR, never under an
    excluded (CBD-middle) path. Prefers a shallow folder that directly
    contains volume files matching the group's keywords."""
    if not EVAL_DIR.exists():
        raise FileNotFoundError(
            f"Expected sample root not found: {EVAL_DIR}\n"
            f"Checked because GROUP_DIR_KEYWORDS assumes samples live under evaluation_4_2/."
        )
    keywords = GROUP_DIR_KEYWORDS[group]
    candidates = []
    all_dirs = [p for p in EVAL_DIR.rglob("*") if p.is_dir()]
    for p in [EVAL_DIR] + all_dirs:
        if _path_is_excluded(p):
            continue
        name = _norm(p.name)
        if p is not EVAL_DIR and not any(k in name for k in keywords):
            continue
        try:
            files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VOLUME_EXTS]
        except OSError:
            continue
        if files:
            candidates.append((p, len(files)))
    if not candidates:
        found = sorted({str(p) for p in all_dirs if not _path_is_excluded(p)})
        raise FileNotFoundError(
            f"[data:{group}] could not find a sample folder under {EVAL_DIR} matching "
            f"keywords {keywords} with .npy/.tif/.tiff files inside.\n"
            f"Folders actually found under evaluation_4_2 (excluding CBD-middle paths):\n  "
            + "\n  ".join(found[:60])
        )
    # Prefer the shallowest folder with the most volume files.
    candidates.sort(key=lambda t: (len(t[0].parts), -t[1]))
    chosen = candidates[0][0]
    if _path_is_excluded(chosen):
        raise RuntimeError(f"[data:{group}] resolved folder {chosen} looks like a CBD-middle path -- aborting")
    return chosen


def load_raw_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        arr = tiff.imread(str(path))
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {arr.shape} in {path}")
    return arr


def remap_labels(arr: np.ndarray) -> np.ndarray:
    """Map an arbitrary 3-level label encoding (0/1/2, 0/0.5/1.0, 0/127/255,
    ...) onto 0=pore, 1=active, 2=CBD by ascending value order. Never
    label-swaps: the lowest raw value is always pore, middle is active,
    highest is CBD, per the task's phase convention."""
    vals = np.unique(arr)
    if vals.size != 3:
        raise ValueError(
            f"expected exactly 3 unique phase values (pore/active/CBD), got {vals.size}: "
            f"{vals[:10]}{' ...' if vals.size > 10 else ''}"
        )
    order = np.sort(vals)
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[arr == order[1]] = 1
    out[arr == order[2]] = 2
    return out


def phase_fractions(vol: np.ndarray) -> dict:
    n = vol.size
    return {
        "pore": float(np.sum(vol == 0)) / n,
        "active": float(np.sum(vol == 1)) / n,
        "cbd": float(np.sum(vol == 2)) / n,
    }


def load_group(group: str) -> dict:
    """Discover, load, remap, and cache every (capped) volume for a group,
    logging exactly what the task requires: folder, count, extensions,
    unique labels before/after conversion, and mean phase fractions."""
    d = discover_sample_dir(group)
    files = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VOLUME_EXTS)
    if not files:
        raise FileNotFoundError(f"[data:{group}] no volume files in {d}")

    ext_counts: dict = {}
    for f in files:
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

    log(f"[data:{group}] folder = {d}")
    log(f"[data:{group}] n_volumes = {len(files)}  extensions = {ext_counts}")

    use_files = files[:MAX_VOLUMES_PER_GROUP]
    if len(use_files) < len(files):
        log(f"[data:{group}] capping at {MAX_VOLUMES_PER_GROUP} of {len(files)} volumes "
            f"for aggregate stats / curves")

    volumes: dict = {}
    fracs = []
    for i, f in enumerate(use_files):
        raw = load_raw_array(f)
        if i < 2:
            log(f"[data:{group}] {f.name}: unique raw labels = {np.unique(raw)}")
        rem = remap_labels(raw)
        if i < 2:
            log(f"[data:{group}] {f.name}: unique remapped labels = {np.unique(rem)}")
        volumes[f] = rem
        fracs.append(phase_fractions(rem))

    mean_fracs = {
        k: float(np.mean([fr[k] for fr in fracs])) for k in ("pore", "active", "cbd")
    }
    log(f"[data:{group}] mean phase fractions: "
        f"pore={mean_fracs['pore']:.4f} active={mean_fracs['active']:.4f} cbd={mean_fracs['cbd']:.4f}")

    return {"dir": d, "files": files, "ext_counts": ext_counts,
            "volumes": volumes, "mean_fracs": mean_fracs}


def check_phase_fractions(group: str, mean_fracs: dict) -> None:
    target = TABLE45_PHASE_FRACTIONS[group]
    bad = [(k, mean_fracs[k], target[k]) for k in ("pore", "active", "cbd")
           if abs(mean_fracs[k] - target[k]) > PHASE_TOL]
    if bad:
        detail = ", ".join(f"{k}: got {g:.4f} vs Table 4.5 {t:.4f}" for k, g, t in bad)
        raise RuntimeError(
            f"[sanity-check] group '{group}' phase fractions deviate from Table 4.5 by more than "
            f"{PHASE_TOL} absolute ({detail}). This usually means the wrong sample folder was "
            f"resolved or the label mapping is off (e.g. label-swap). Stopping rather than "
            f"building a figure from suspect data."
        )
    log(f"[sanity-check] group '{group}' OK against Table 4.5 "
        f"(tolerance {PHASE_TOL}): {mean_fracs}")


def choose_representative(group: str, group_data: dict):
    """Sample whose phase fractions are closest to the group's median."""
    items = list(group_data["volumes"].items())
    fr = np.array([[phase_fractions(v)[k] for k in ("pore", "active", "cbd")] for _, v in items])
    med = np.median(fr, axis=0)
    d = np.linalg.norm(fr - med, axis=1)
    k = int(np.argmin(d))
    f, vol = items[k]
    log(f"[rep:{group}] {f.name} (closest to group median phase fractions)")
    return f, vol


# ============================================================================
# 6. CARD / LABEL / AXIS PRIMITIVES -- reused verbatim from Figure 4.1
# ============================================================================


def add_card(fig, xywh):
    x, y, w, h = xywh
    fig.add_artist(Rectangle(
        (x, y), w, h,
        transform=fig.transFigure,
        facecolor=CARD_FACE, edgecolor=CARD_EDGE,
        linewidth=CARD_LW, zorder=-50,
    ))


def add_panel_label(fig, x, y, label):
    fig.text(x, y, label, fontsize=14, fontweight="bold",
             color=TEXT, ha="left", va="top", zorder=100)


def style_axis(ax, grid=True):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(SPINE)
        sp.set_linewidth(0.7)
    ax.tick_params(colors=SUBTEXT, labelcolor=SUBTEXT)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.55, alpha=0.9)
        ax.set_axisbelow(True)


def image_cell(ax, color, lw=CELL_LW):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(color)


# ============================================================================
# 7. THREE-PHASE 3D ORTHOSLICE-CUBE RENDERING (panel a)
# ============================================================================


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _phase_gray_rgba(label_slice: np.ndarray) -> np.ndarray:
    g = PHASE_GRAY_VALUES[label_slice]
    return np.stack([g, g, g, np.ones_like(g)], axis=-1)


def render_orthoslice_cube_pyvista(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float):
    import pyvista as pv

    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    nz, ny, nx = vol.shape
    mx, my, mz = nx // 2, ny // 2, nz // 2

    grid = pv.ImageData()
    grid.dimensions = np.array([nx, ny, nz]) + 1
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)
    # pyvista/VTK expects Fortran (x-fastest) cell ordering; our array is
    # (z, y, x), so transpose to (x, y, z) before flattening.
    grid.cell_data["phase"] = np.transpose(vol, (2, 1, 0)).flatten(order="F")

    slices = grid.slice_orthogonal(x=mx, y=my, z=mz)
    phase_cmap = ListedColormap([to_hex((g, g, g)) for g in PHASE_GRAY_VALUES])

    pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX))
    pl.set_background("white")
    pl.add_mesh(slices, scalars="phase", cmap=phase_cmap, clim=[0, 2], show_scalar_bar=False)
    pl.add_mesh(pv.Box(bounds=(0, nx, 0, ny, 0, nz)), style="wireframe",
                color=COLORS[group], line_width=2.2, opacity=0.6)

    pl.enable_parallel_projection()
    center = np.array([nx / 2.0, ny / 2.0, nz / 2.0])
    direction = np.array([1.0, -1.30, 0.90])
    direction /= np.linalg.norm(direction)
    pl.camera.focal_point = tuple(center)
    pl.camera.position = tuple(center + direction * 4.0 * max(vol.shape))
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.parallel_scale = float(parallel_scale)

    pl.screenshot(str(out_raw), transparent_background=True)
    pl.close()


def render_orthoslice_cube_matplotlib(vol: np.ndarray, group: str, out_raw: Path):
    """Matplotlib Axes3D fallback: three orthogonal grayscale slice planes
    inside a group-colored wireframe box. Used automatically if PyVista is
    unavailable or its off-screen render fails."""
    nz, ny, nx = vol.shape
    mz, my, mx = nz // 2, ny // 2, nx // 2

    fig = plt.figure(figsize=(CANVAS_PX / 200.0, CANVAS_PX / 200.0), dpi=200)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    ax.set_proj_type("ortho")

    # XY plane at z = mz (varies x, y)
    X, Y = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    Z = np.full_like(X, mz, dtype=float)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                     facecolors=_phase_gray_rgba(vol[mz, :, :]).transpose(1, 0, 2),
                     shade=False, linewidth=0, antialiased=False)

    # XZ plane at y = my (varies x, z)
    X2, Z2 = np.meshgrid(np.arange(nx + 1), np.arange(nz + 1), indexing="ij")
    Y2 = np.full_like(X2, my, dtype=float)
    ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1,
                     facecolors=_phase_gray_rgba(vol[:, my, :]).transpose(1, 0, 2),
                     shade=False, linewidth=0, antialiased=False)

    # YZ plane at x = mx (varies y, z)
    Y3, Z3 = np.meshgrid(np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    X3 = np.full_like(Y3, mx, dtype=float)
    ax.plot_surface(X3, Y3, Z3, rstride=1, cstride=1,
                     facecolors=_phase_gray_rgba(vol[:, :, mx]).transpose(1, 0, 2),
                     shade=False, linewidth=0, antialiased=False)

    corners = np.array([[x, y, z] for x in (0, nx) for y in (0, ny) for z in (0, nz)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        p1, p2 = corners[i], corners[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=COLORS[group], linewidth=1.6, alpha=0.9)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)

    fig.savefig(out_raw, transparent=True, dpi=200)
    plt.close(fig)


def render_group_cube(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float) -> bool:
    """Try the preferred PyVista orthoslice-cube render; fall back to the
    Matplotlib Axes3D version on any failure. Returns True if PyVista path
    succeeded."""
    try:
        render_orthoslice_cube_pyvista(vol, group, out_raw, parallel_scale)
        return True
    except Exception as exc:
        log(f"[render:{group}] PyVista orthoslice-cube unavailable/failed -> Matplotlib fallback ({exc})")
        render_orthoslice_cube_matplotlib(vol, group, out_raw)
        return False


def finalize_renders(raw_paths, out_paths, pad_frac=0.06):
    """Common alpha crop across all groups -> identical scale and centring.
    Identical to Figure 4.1's version; works on any RGBA source."""
    ims = {g: Image.open(p).convert("RGBA") for g, p in raw_paths.items()}
    boxes = []
    for im in ims.values():
        a = np.asarray(im)[:, :, 3]
        m = a > 5
        if m.any():
            ys, xs = np.where(m)
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if not boxes:
        boxes = [(0, 0, list(ims.values())[0].size[0] - 1, list(ims.values())[0].size[1] - 1)]

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)

    pad = int(round(pad_frac * max(x1 - x0, y1 - y0))) + 6
    side = max(x1 - x0, y1 - y0) + 2 * pad
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    rs = _resample()
    for g, im in ims.items():
        W, H = im.size
        L = int(round(cx - side / 2.0))
        T = int(round(cy - side / 2.0))
        crop = Image.new("RGBA", (side, side), (255, 255, 255, 0))
        sx0, sy0 = max(L, 0), max(T, 0)
        sx1, sy1 = min(L + side, W), min(T + side, H)
        crop.alpha_composite(im.crop((sx0, sy0, sx1, sy1)), (sx0 - L, sy0 - T))
        crop = crop.resize((CANVAS_PX, CANVAS_PX), rs)
        canvas = Image.new("RGBA", (CANVAS_PX, CANVAS_PX), (255, 255, 255, 255))
        canvas.alpha_composite(crop)
        canvas.convert("RGB").save(out_paths[g])


# ============================================================================
# 8. CSV DISCOVERY (official metrics / curves)
# ============================================================================

CSV_TOPIC_KEYWORDS = ["tpc", "two_point", "lineal", "chord", "local", "interface",
                       "topology", "component", "distance", "diversity", "nearest",
                       "summary", "per_sample", "group", "curve"]


def list_metric_csvs(root: Path):
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if not _path_is_excluded(p))


def find_csv(keywords, roots):
    """Best-effort search for a CSV whose filename overlaps the given
    keywords, across the given roots (in order). Never returns a file under
    an excluded CBD-middle path."""
    cands = []
    for root in roots:
        for p in list_metric_csvs(root):
            score = sum(1 for k in keywords if k in _norm(p.stem))
            if score > 0:
                cands.append((score, len(str(p)), p))
    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], t[1]))
    return cands[0][2]


def log_available_csvs(label, roots):
    all_csv = []
    for root in roots:
        all_csv.extend(list_metric_csvs(root))
    log(f"[csv-search:{label}] {len(all_csv)} candidate CSV(s) available "
        f"(CBD-middle paths excluded): {[str(p) for p in all_csv][:30]}")


# ============================================================================
# 9. RECONSTRUCTION-SENSITIVE CURVES (panel b)
# ============================================================================


def _periodic_radius_grid(shape):
    axes = [np.fft.fftfreq(n) * n for n in shape]
    gz, gy, gx = np.meshgrid(*axes, indexing="ij")
    return np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)


def _radial_average(field, r, max_r, nbins):
    r_flat = r.ravel()
    f_flat = field.ravel()
    bins = np.linspace(0, max_r, nbins + 1)
    idx = np.digitize(r_flat, bins) - 1
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if np.any(m):
            xs.append(0.5 * (bins[b] + bins[b + 1]))
            ys.append(float(np.mean(f_flat[m])))
    return np.array(xs), np.array(ys)


def compute_tpc(mask_a: np.ndarray, mask_b: np.ndarray, max_r: int = MAX_R_VOX, nbins: int = 26):
    """Periodic-boundary two-point correlation S_AB(r) via FFT, radially
    binned. mask_a is mask_b for the same-phase case."""
    fa = np.fft.fftn(mask_a.astype(np.float64))
    fb = fa if mask_b is mask_a else np.fft.fftn(mask_b.astype(np.float64))
    corr = np.fft.ifftn(fa * np.conj(fb)).real / mask_a.size
    r = _periodic_radius_grid(mask_a.shape)
    return _radial_average(corr, r, max_r, nbins)


def _run_lengths_batch(bool2d: np.ndarray) -> np.ndarray:
    L, N = bool2d.shape
    padded = np.zeros((L, N + 2), dtype=np.int8)
    padded[:, 1:-1] = bool2d.astype(np.int8)
    d = np.diff(padded, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    return (ends[:, 1] - starts[:, 1]).astype(np.int64)


def lineal_path(mask: np.ndarray, max_r: int = MAX_R_VOX):
    """Lineal-path function L(r), averaged over the 3 principal axes, via
    vectorized run-length encoding (probability a length-r segment lies
    entirely within the phase)."""
    r_vals = np.arange(1, max_r + 1, dtype=np.float64)
    hits = np.zeros(max_r, dtype=np.float64)
    possible = np.zeros(max_r, dtype=np.float64)
    for axis in range(3):
        m = np.moveaxis(mask, axis, -1)
        N = m.shape[-1]
        lines = m.reshape(-1, N).astype(bool)
        lengths = _run_lengths_batch(lines)
        if lengths.size:
            contrib = np.clip(lengths[:, None] - r_vals[None, :] + 1, 0, None)
            hits += contrib.sum(axis=0)
        possible += lines.shape[0] * np.clip(N - r_vals + 1, 0, None)
    L = np.divide(hits, possible, out=np.zeros_like(hits), where=possible > 0)
    return r_vals, L


def local_active_fraction(mask_active: np.ndarray, block: int = LOCAL_BLOCK):
    shape = mask_active.shape
    trimmed = tuple((s // block) * block for s in shape)
    v = mask_active[:trimmed[0], :trimmed[1], :trimmed[2]]
    nz, ny, nx = (trimmed[0] // block, trimmed[1] // block, trimmed[2] // block)
    v = v.reshape(nz, block, ny, block, nx, block)
    return v.mean(axis=(1, 3, 5)).ravel()


def compute_curve_from_volumes(groups_data, func):
    """func(vol) -> (x, y); averaged per group, std kept for 'real' only."""
    out = {}
    for g in GROUPS:
        vols = list(groups_data[g]["volumes"].values())
        x_ref, ys = None, []
        for v in vols:
            x, y = func(v)
            if x_ref is None:
                x_ref = x
            ys.append(y)
        ys = np.asarray(ys)
        out[g] = {
            "x": x_ref,
            "y": ys.mean(axis=0),
            "std": ys.std(axis=0) if g == "real" else None,
        }
    return out


def local_heterogeneity_curves(groups_data):
    edges = np.linspace(0, 1, LOCAL_HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for g in GROUPS:
        vols = list(groups_data[g]["volumes"].values())
        frac = np.concatenate([local_active_fraction(v == 1) for v in vols]) if vols else np.array([])
        if frac.size:
            hist, _ = np.histogram(frac, bins=edges, density=True)
        else:
            hist = np.zeros(LOCAL_HIST_BINS)
        out[g] = {"x": centers, "y": hist, "std": None}
    return out


def try_load_official_curve(keywords, name_hints, roots):
    """Best-effort search for an official group-mean curve CSV with a
    Figure-4.1-style schema (curve/group/x/mean[/std]). Returns
    {group: {"x","y","std"}} or None if nothing safely resolves."""
    csv = find_csv(keywords, roots)
    if csv is None:
        return None
    try:
        df = pd.read_csv(csv)
    except Exception as exc:
        log(f"[curve] failed to read candidate CSV {csv}: {exc}")
        return None

    curve_col = pick_col(df, ["curve", "curve_name", "metric", "name", "descriptor"], "curve", required=False)
    group_col = pick_col(df, ["group", "set", "model", "category", "class"], "group", required=False)
    x_col = pick_col(df, ["x", "r", "radius_vox", "radius", "lag", "bin", "bin_center",
                           "length", "segment_length"], "x", required=False)
    y_col = pick_col(df, ["mean", "y_mean", "value_mean", "y", "value", "mean_value"], "y", required=False)
    std_col = pick_col(df, ["std", "y_std", "value_std", "sd", "stdev", "std_value"], "std", required=False)
    if group_col is None or x_col is None or y_col is None:
        log(f"[curve] candidate CSV {csv} missing required columns -> skip")
        return None

    sub = df
    if curve_col is not None and name_hints:
        mask = df[curve_col].astype(str).str.lower().apply(lambda s: any(h in s for h in name_hints))
        if not mask.any():
            log(f"[curve] no rows in {csv} match hints {name_hints} -> skip")
            return None
        sub = df[mask]

    sub = sub.copy()
    sub["_group"] = sub[group_col].map(normalize_group)
    out = {}
    for g in GROUPS:
        s = sub[sub["_group"] == g].sort_values(x_col)
        if s.empty:
            continue
        x = pd.to_numeric(s[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(s[y_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        std = None
        if std_col is not None:
            sd = np.nan_to_num(pd.to_numeric(s[std_col], errors="coerce").to_numpy(float)[ok])
            if np.any(sd > 0):
                std = sd
        out[g] = {"x": x, "y": y, "std": std}
    if len(out) < 2:
        return None
    log(f"[curve] using official CSV: {csv}")
    return out


def resolve_curve(kind, groups_data):
    """Try official CSV first; fall back to computing from volumes, always
    logging which path was used. Returns (curves, is_fallback)."""
    roots = [METRICS_DIR, CLEAN_PACKAGE_DIR]
    if kind == "active_tpc":
        off = try_load_official_curve(["tpc", "two_point", "twopoint"],
                                       ["11", "activeactive", "phase1phase1", "s11"], roots)
        if off:
            return off, False
        log("[curve:active_tpc] no official CSV resolved -> computing S_11(r) from volumes (FFT, periodic)")
        return compute_curve_from_volumes(groups_data, lambda v: compute_tpc(v == 1, v == 1)), True

    if kind == "cross_tpc":
        off = try_load_official_curve(["tpc", "two_point", "twopoint"],
                                       ["01", "10", "poreactive", "activepore", "cross", "s01"], roots)
        if off:
            return off, False
        log("[curve:cross_tpc] no official CSV resolved -> computing S_01(r) from volumes (FFT, periodic)")
        return compute_curve_from_volumes(groups_data, lambda v: compute_tpc(v == 0, v == 1)), True

    if kind == "lineal":
        off = try_load_official_curve(["lineal"], ["lineal"], roots)
        if off:
            return off, False
        log("[curve:lineal] no official CSV resolved -> computing active lineal-path from volumes (run-length)")
        return compute_curve_from_volumes(groups_data, lambda v: lineal_path(v == 1)), True

    if kind == "local_heterogeneity":
        off = try_load_official_curve(["local"], ["active", "heterogen"], roots)
        if off:
            return off, False
        log("[curve:local_heterogeneity] no official CSV resolved -> computing "
            f"{LOCAL_BLOCK}^3 local active-fraction histogram from volumes")
        return local_heterogeneity_curves(groups_data), True

    raise ValueError(kind)


def plot_group_curves(ax, curves, title, xlabel, ylabel, show_legend=False):
    style_axis(ax)
    ax.set_title(title, pad=4.5, color=TEXT, fontweight="bold")
    ax.set_xlabel(xlabel, color=SUBTEXT)
    ax.set_ylabel(ylabel, color=SUBTEXT)

    if not curves:
        ax.text(0.5, 0.5, "metric unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, color=SUBTEXT)
        return

    for g in GROUPS:
        if g not in curves:
            log(f"[curve-plot] '{title}': group '{g}' missing")
            continue
        d = curves[g]
        x, y, std = d["x"], d["y"], d.get("std")
        if g == "real" and std is not None and np.any(std > 0):
            ax.fill_between(x, y - std, y + std, color=COLORS[g], alpha=0.20, linewidth=0, zorder=1)
        ax.plot(x, y, color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                 solid_capstyle="round", label=LABELS[g], zorder=3 if g == "real" else 4)

    ax.margins(x=0.02, y=0.06)

    if show_legend:
        leg = ax.legend(loc="upper right", frameon=True, facecolor="white",
                         edgecolor=SPINE, framealpha=0.94, handlelength=2.3,
                         borderpad=0.45, labelspacing=0.34)
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(8)


# ============================================================================
# 10. PANEL C -- interface hierarchy fingerprint + active-domain continuity
# ============================================================================


def resolve_interface_hierarchy():
    roots = [METRICS_DIR, CLEAN_PACKAGE_DIR]
    csv = find_csv(["interface", "density"], roots)
    if csv is not None:
        try:
            df = pd.read_csv(csv)
            gcol = pick_col(df, ["group", "set", "model"], "group", required=False)
            if gcol is not None:
                df = df.copy()
                df["_group"] = df[gcol].map(normalize_group)
                col_candidates = {
                    "pore_active": ["pore_active_interface_density", "pore_active_density", "poreactive"],
                    "pore_cbd": ["pore_cbd_interface_density", "pore_cbd_density", "porecbd"],
                    "active_cbd": ["active_cbd_interface_density", "active_cbd_density", "activecbd"],
                    "tpc_proxy": ["tpc_proxy_density", "triple_phase_contact_density", "tpcproxy"],
                }
                resolved = {k: pick_col(df, v, k, required=False) for k, v in col_candidates.items()}
                if all(v is not None for v in resolved.values()):
                    out, ok = {}, True
                    for g in GROUPS:
                        row = df[df["_group"] == g]
                        if row.empty:
                            ok = False
                            break
                        out[g] = {k: float(pd.to_numeric(row.iloc[0][c], errors="coerce"))
                                  for k, c in resolved.items()}
                    if ok:
                        log(f"[panel-c-left] using official interface-hierarchy CSV: {csv}")
                        return out, False
        except Exception as exc:
            log(f"[panel-c-left] failed to parse candidate CSV {csv}: {exc}")
    log("[panel-c-left] official interface-hierarchy metrics not resolved "
        "-> using verified Table 4.5/4.6 fallback values")
    return TABLE46_INTERFACE, True


def resolve_active_continuity():
    """CRITICAL: never recomputes connected components / Euler / chord from
    volumes directly, per the task's explicit warning about mismatched
    conventions -- only an official CSV or the verified Table 4.6 fallback."""
    roots = [METRICS_DIR, CLEAN_PACKAGE_DIR]
    csv = find_csv(["component", "chord", "topology"], roots)
    if csv is not None:
        try:
            df = pd.read_csv(csv)
            gcol = pick_col(df, ["group", "set", "model"], "group", required=False)
            if gcol is not None:
                df = df.copy()
                df["_group"] = df[gcol].map(normalize_group)
                comp_col = pick_col(df, ["n_components_active", "active_connected_components",
                                          "components_active"], "components", required=False)
                chord_col = pick_col(df, ["chord_active_mean", "mean_active_chord_length",
                                           "active_chord_mean"], "chord", required=False)
                if comp_col is not None and chord_col is not None:
                    out, ok = {}, True
                    for g in GROUPS:
                        row = df[df["_group"] == g]
                        if row.empty:
                            ok = False
                            break
                        out[g] = {
                            "components": float(pd.to_numeric(row.iloc[0][comp_col], errors="coerce")),
                            "chord": float(pd.to_numeric(row.iloc[0][chord_col], errors="coerce")),
                        }
                    if ok:
                        log(f"[panel-c-right] using official active-domain continuity CSV: {csv}")
                        return out, False
        except Exception as exc:
            log(f"[panel-c-right] failed to parse candidate CSV {csv}: {exc}")
    log("[panel-c-right] official active-domain continuity metrics not resolved "
        "-> using verified Table 4.6 fallback group means "
        "(never recomputed ad hoc, per task instructions)")
    return TABLE46_CONTINUITY, True


def plot_interface_fingerprint(ax, data):
    style_axis(ax)
    ax.set_title("Interface hierarchy fingerprint", pad=4.5, color=TEXT, fontweight="bold")
    ax.set_ylabel("Density / proxy value", color=SUBTEXT)

    xs = np.arange(len(INTERFACE_METRIC_ORDER))
    for g in GROUPS:
        if g not in data:
            log(f"[panel-c-left] group '{g}' missing from interface-hierarchy data")
            continue
        vals = [data[g][m] for m in INTERFACE_METRIC_ORDER]
        ax.plot(xs, vals, color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                 marker="o", markersize=5.5, markerfacecolor=COLORS[g],
                 markeredgecolor="black", markeredgewidth=0.6,
                 label=LABELS[g], zorder=3 if g == "real" else 4)

    ax.set_xticks(xs)
    ax.set_xticklabels(INTERFACE_TICK_LABELS, fontsize=7.2)
    ax.set_xlim(-0.4, len(INTERFACE_METRIC_ORDER) - 0.6)
    ax.margins(y=0.16)

    leg = ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=SPINE,
                     framealpha=0.94, handlelength=2.0, borderpad=0.4,
                     labelspacing=0.3, fontsize=7.0)
    leg.get_frame().set_linewidth(0.6)


def plot_active_continuity(ax, data):
    style_axis(ax)
    ax.set_title("Active-domain continuity", pad=4.5, color=TEXT, fontweight="bold")
    ax.set_xlabel("Active connected components (log scale)", color=SUBTEXT)
    ax.set_ylabel("Mean active chord length (vox)", color=SUBTEXT)
    ax.set_xscale("log")

    pts = {g: (data[g]["components"], data[g]["chord"]) for g in GROUPS if g in data}
    missing = [g for g in GROUPS if g not in data]
    for g in missing:
        log(f"[panel-c-right] group '{g}' missing from active-continuity data")

    # Default label offset is up-right; any pair of points that would sit
    # nearly on top of each other (as Reference/SliceGAN typically do here)
    # gets pushed apart in opposite diagonal directions instead, so the two
    # labels never collide regardless of which data source populated `data`.
    offsets = {g: (9, 8) for g in pts}
    keys = list(pts.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            gi, gj = keys[i], keys[j]
            xi, yi = pts[gi]
            xj, yj = pts[gj]
            dx_log = abs(np.log10(max(xi, 1e-9)) - np.log10(max(xj, 1e-9)))
            dy = abs(yi - yj)
            if dx_log < 0.15 and dy < 2.0:
                offsets[gi] = (-12, 16)
                offsets[gj] = (12, -18)

    for g, (x, y) in pts.items():
        ax.scatter([x], [y], s=115, color=COLORS[g], marker=MARKERS[g],
                    edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(LABELS[g], (x, y), textcoords="offset points", xytext=offsets[g],
                    fontsize=7.4, color=TITLE_COLOR[g], fontweight="bold", zorder=6)

    ax.margins(x=0.35, y=0.22)


# ============================================================================
# 11. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT      = {PROJECT}")
    log(f"[paths] EVAL_DIR     = {EVAL_DIR}")
    log(f"[paths] METRICS_DIR  = {METRICS_DIR}")
    log(f"[paths] CLEAN_PKG    = {CLEAN_PACKAGE_DIR}")
    log(f"[paths] OUT          = {OUT}")
    log_available_csvs("metrics", [METRICS_DIR, CLEAN_PACKAGE_DIR])

    # ---- data: discover, load, sanity-check ----------------------------
    groups_data = {}
    for g in GROUPS:
        gd = load_group(g)
        check_phase_fractions(g, gd["mean_fracs"])
        groups_data[g] = gd

    rep = {}
    for g in GROUPS:
        f, vol = choose_representative(g, groups_data[g])
        rep[g] = {"file": f, "vol": vol, "fracs": phase_fractions(vol)}

    # ---- panel a: 3D orthoslice cubes -----------------------------------
    parallel_scale = 0.72 * max(max(rep[g]["vol"].shape) for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}
    for g in GROUPS:
        render_group_cube(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    finalize_renders(raw_paths, png_paths)

    # ---- panel b: curves --------------------------------------------------
    curves_active_tpc, fb1 = resolve_curve("active_tpc", groups_data)
    curves_cross_tpc, fb2 = resolve_curve("cross_tpc", groups_data)
    curves_lineal, fb3 = resolve_curve("lineal", groups_data)
    curves_local_het, fb4 = resolve_curve("local_heterogeneity", groups_data)

    # ---- panel c: interface hierarchy + active-domain continuity ---------
    interface_data, interface_is_fallback = resolve_interface_hierarchy()
    continuity_data, continuity_is_fallback = resolve_active_continuity()

    # ---- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.patch.set_facecolor(BG)

    add_card(fig, card_a)
    add_card(fig, card_b)
    add_card(fig, card_c)

    add_panel_label(fig, card_a[0] + 0.007, card_a[1] + card_a[3] + PANEL_LABEL_OFFSET, "a)")
    add_panel_label(fig, card_b[0] + 0.007, card_b[1] + card_b[3] + PANEL_LABEL_OFFSET, "b)")
    add_panel_label(fig, card_c[0] + 0.007, card_c[1] + card_c[3] + PANEL_LABEL_OFFSET, "c)")

    # =========================== PANEL A ====================================
    ax_, ay_, aw_, ah_ = card_a
    pad_x, pad_top, pad_bot = 0.016, 0.038, 0.034
    label_w, gap_x, gap_y = 0.062, 0.018, 0.009

    free_w_in = (aw_ - 2 * pad_x - label_w - 3 * gap_x) * FIG_W / 3.0
    free_h_in = (ah_ - pad_top - pad_bot - 3 * gap_y) * FIG_H / 4.0
    cell_in = min(free_w_in, free_h_in)
    cell_w, cell_h = cell_in / FIG_W, cell_in / FIG_H          # square cells

    grid_w = label_w + 3 * gap_x + 3 * cell_w
    grid_h = 4 * cell_h + 3 * gap_y
    gx0 = ax_ + (aw_ - grid_w) / 2.0
    gy_top = ay_ + ah_ - pad_top - max(0.0, (ah_ - pad_top - pad_bot - grid_h) / 2.0)

    def row_y(r):
        return gy_top - (r + 1) * cell_h - r * gap_y

    for r, name in enumerate(["3D volume", "X–Y slice", "X–Z slice", "Y–Z slice"]):
        a = fig.add_axes([gx0, row_y(r), label_w, cell_h])
        a.axis("off")
        a.text(0.98, 0.5, name, ha="right", va="center",
               fontsize=8.8, fontweight="bold", color=TEXT)

    for c, g in enumerate(GROUPS):
        x = gx0 + label_w + gap_x + c * (cell_w + gap_x)

        fig.text(x + cell_w / 2.0, gy_top + 0.010, LABELS[g],
                 ha="center", va="bottom", fontsize=9.6,
                 fontweight="bold", color=TITLE_COLOR[g])

        a = fig.add_axes([x, row_y(0), cell_w, cell_h])
        a.imshow(Image.open(png_paths[g]), interpolation="bilinear")
        image_cell(a, COLORS[g])
        fr = rep[g]["fracs"]
        a.text(0.035, 0.035, f"{fr['pore']:.2f}/{fr['active']:.2f}/{fr['cbd']:.2f}",
               transform=a.transAxes, fontsize=6.6, color=SUBTEXT,
               ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))

        v = rep[g]["vol"]
        slices = [v[v.shape[0] // 2, :, :], v[:, v.shape[1] // 2, :], v[:, :, v.shape[2] // 2]]
        for r, sl in enumerate(slices):
            a = fig.add_axes([x, row_y(r + 1), cell_w, cell_h])
            a.imshow(PHASE_GRAY_VALUES[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            image_cell(a, COLORS[g])

    # small phase legend (Pore | Active | CBD), centered under the grid
    legend_y = ay_ + 0.015
    phase_hex = [to_hex((v, v, v)) for v in PHASE_GRAY_VALUES]
    item_widths = [0.070, 0.082, 0.062]  # tuned to each label's rendered width
    total_w = sum(item_widths)
    lx = gx0 + label_w + gap_x + (grid_w - label_w - gap_x - total_w) / 2.0
    for name, color, w in zip(PHASE_NAMES, phase_hex, item_widths):
        fig.add_artist(Rectangle((lx, legend_y - 0.006), 0.013, 0.011,
                                  transform=fig.transFigure, facecolor=color,
                                  edgecolor=SPINE, linewidth=0.6, zorder=101))
        fig.text(lx + 0.019, legend_y, name, ha="left", va="center",
                 fontsize=7.4, color=SUBTEXT, zorder=101)
        lx += w

    # =========================== PANEL B ====================================
    bx_, by_, bw_, bh_ = card_b
    b_pl, b_pr, b_pt, b_pb = 0.045, 0.018, 0.040, 0.046
    b_gx, b_gy = 0.058, 0.076

    plot_w = (bw_ - b_pl - b_pr - b_gx) / 2.0
    plot_h = (bh_ - b_pt - b_pb - b_gy) / 2.0

    panel_b_specs = [
        (curves_active_tpc, "Active same-phase TPC", "Lag / radius (vox)", r"$S_{11}(r)$"),
        (curves_cross_tpc, "Pore-active cross TPC", "Lag / radius (vox)", r"$S_{01}(r)$"),
        (curves_lineal, "Active lineal path", "Segment length (vox)", "Probability"),
        (curves_local_het, "Local active-fraction heterogeneity",
         "Active fraction in local windows", "Density"),
    ]

    for i, (curves, title, xlab, ylab) in enumerate(panel_b_specs):
        r, c = divmod(i, 2)
        x = bx_ + b_pl + c * (plot_w + b_gx)
        y = by_ + bh_ - b_pt - (r + 1) * plot_h - r * b_gy
        ax = fig.add_axes([x, y, plot_w, plot_h])
        plot_group_curves(ax, curves, title, xlab, ylab, show_legend=(i == 1))

    # =========================== PANEL C ====================================
    # Left/right mechanism split (not the stacked-distribution layout used in
    # Figure 4.1's final panel c) -- per this figure's own spec. Card
    # geometry and header style are otherwise identical to Figure 4.1.
    cx_, cy_, cw_, ch_ = card_c

    header_title_y = cy_ + ch_ - 0.020
    header_subtitle_y = header_title_y - 0.020
    plot_top = header_subtitle_y - 0.028
    c_pl, c_pr, c_pb, c_gx = 0.045, 0.020, 0.046, 0.055
    plot_bottom = cy_ + c_pb
    plot_h_c = plot_top - plot_bottom
    plot_w_c = (cw_ - c_pl - c_pr - c_gx) / 2.0
    left_x = cx_ + c_pl
    right_x = left_x + plot_w_c + c_gx

    fig.text(cx_ + 0.014, header_title_y, "Mechanistic descriptors",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)
    fig.text(cx_ + 0.014, header_subtitle_y,
             "Interface hierarchy and active-domain continuity from the same official "
             "Section 4.2 evaluation used in Tables 4.5 and 4.6.",
             ha="left", va="top", fontsize=7.2, color=SUBTEXT, wrap=True)

    ax_left = fig.add_axes([left_x, plot_bottom, plot_w_c, plot_h_c])
    ax_right = fig.add_axes([right_x, plot_bottom, plot_w_c, plot_h_c])

    plot_interface_fingerprint(ax_left, interface_data)
    plot_active_continuity(ax_right, continuity_data)

    # =========================== SAVE ========================================
    png = OUT / f"{STEM}.png"
    pdf = OUT / f"{STEM}.pdf"
    svg = OUT / f"{STEM}.svg"
    tif_out = OUT / f"{STEM}.tiff"

    fig.savefig(png, dpi=450, facecolor=BG)     # no bbox_inches -> geometry preserved
    fig.savefig(pdf, facecolor=BG)
    fig.savefig(svg, facecolor=BG)
    Image.open(png).convert("RGB").save(tif_out, compression="tiff_lzw", dpi=(450, 450))
    plt.close(fig)

    log("\n[summary] fallbacks used:")
    log(f"  active_tpc curve         : {'Table/computed fallback' if fb1 else 'official CSV'}")
    log(f"  cross_tpc curve          : {'Table/computed fallback' if fb2 else 'official CSV'}")
    log(f"  lineal-path curve        : {'Table/computed fallback' if fb3 else 'official CSV'}")
    log(f"  local-heterogeneity curve: {'Table/computed fallback' if fb4 else 'official CSV'}")
    log(f"  panel-c interface data   : {'Table 4.5/4.6 fallback' if interface_is_fallback else 'official CSV'}")
    log(f"  panel-c continuity data  : {'Table 4.6 fallback' if continuity_is_fallback else 'official CSV'}")

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out):
        log(" ", p.resolve())


if __name__ == "__main__":
    sys.exit(main())
