#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.2 composite (magma, final)
Reference vs MicroLad vs SliceGAN -- three-phase NMC cathode 2D-to-3D reconstruction.

Same visual contract as Figure 4.1 (make_fig4_1_composite_magma_final.py):
same cards, fonts, panel-label style, and export block. No tight_layout /
constrained_layout / bbox_inches.

This revision uses ONLY the exact official standardized-input volume
folders and the exact official long-format metric CSVs / summary tables
named in the task -- it does not search broadly for sample folders or
metric files. Table 4.5/4.6 values are used only as an explicit, logged
fallback for panel c if an official summary table cannot be loaded, and
are never used to silently replace panel-b curves (those have no
fallback: if an official CSV is missing, the panel shows a clean
"metric unavailable" note instead).
"""

from __future__ import annotations

import re
import sys
import warnings
from collections import Counter
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

# Project root is auto-detected from this file's location -- resolves to
# /home/ra2/4.2/microlad_nmc_2d_to_3d when the script sits at
# <PROJECT>/scripts/make_fig4_2_composite_magma_final.py, exactly as
# specified.
PROJECT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT / "evaluation_4_2_results_all_metrics"

OUT = PROJECT / "paper_figures_4_2" / "figure_4_2_composite_magma_final"
TMP = OUT / "_render_cache"
LOGS = PROJECT / "logs"

STEM = "fig4_2_composite_magma_final"

GROUPS = ["real", "microlad", "slicegan"]

# Exact official standardized 64^3 metric-input volume folders. No broader
# search is performed under evaluation_4_2/* -- those folders (and any
# "_original_", "128"/"127"-sized, or CBD-middle variant) are intentionally
# never touched by this script.
GROUP_VOLUME_DIRS = {
    "real": PROJECT / "evaluation_4_2_all_metrics_inputs" / "real",
    "microlad": PROJECT / "evaluation_4_2_all_metrics_inputs" / "diffusion",
    "slicegan": PROJECT / "evaluation_4_2_all_metrics_inputs" / "gan",
}

# Group name strings as they literally appear in the official CSVs.
CSV_GROUP_MAP = {"real": "real", "diffusion": "microlad", "gan": "slicegan"}

EXCLUDE_PATH_TOKENS = ("cbdmiddle", "cbd_middle", "calibrated", "comparisonwithcbdmiddle",
                        "comparison_with_cbdmiddle", "original", "128", "127")

VOLUME_EXTS = (".npy", ".tif", ".tiff")
EXPECTED_SHAPE = (64, 64, 64)
MAX_VOLUMES_PER_GROUP = 50

# ---- exact official metric files (long-format curves + summary tables) ----
TPC_CSV = METRICS_DIR / "metrics" / "tpc_same_and_cross_curves.csv"
LINEAL_CSV = METRICS_DIR / "metrics" / "lineal_path_curves.csv"
LOCAL_FRAC_CSV = METRICS_DIR / "metrics" / "local_phase_fractions_blocks.csv"
INTERFACE_CSV = METRICS_DIR / "tables" / "summary_interface_density.csv"
TPC_PROXY_CSV = METRICS_DIR / "tables" / "summary_triple_phase_contact_proxy.csv"
CONNECTIVITY_CSV = METRICS_DIR / "tables" / "summary_connectivity_topology.csv"
CHORD_CSV = METRICS_DIR / "tables" / "summary_chord_lengths.csv"

LOCAL_HIST_BINS = 30  # np.linspace(0, 1, 31) -> 30 bins, per spec

RENDER_PX = 1500
CANVAS_PX = 900

FIG_W, FIG_H = 17.8, 10.2

SHOW_CURVE_ERRORS = False  # panel-b relative-error corner notes: off by default

# ============================================================================
# 1b. SANITY-CHECK / FALLBACK CONSTANTS (verified Table 4.5 / 4.6 values)
# ============================================================================

# Table 4.5 -- phase fractions from the exact official
# evaluation_4_2_all_metrics_inputs folders. Used only to sanity-check
# loaded data.
TABLE45_PHASE_FRACTIONS = {
    "real":     {"pore": 0.41143517, "active": 0.45933434, "cbd": 0.12923050},
    "microlad": {"pore": 0.42737579, "active": 0.46818214, "cbd": 0.10444206},
    "slicegan": {"pore": 0.40131248, "active": 0.46328995, "cbd": 0.13539757},
}
PHASE_TOL = 0.01

# Table 4.6 -- interface hierarchy densities + triple-phase-contact proxy.
# Fallback ONLY if summary_interface_density.csv / summary_triple_phase_
# contact_proxy.csv cannot be loaded.
TABLE46_INTERFACE = {
    "real":     {"pore_active": 0.0000, "pore_cbd": 0.0677, "active_cbd": 0.0496, "tpc_proxy": 0.044845},
    "microlad": {"pore_active": 0.033132, "pore_cbd": 0.018456, "active_cbd": 0.072058, "tpc_proxy": 0.094238},
    "slicegan": {"pore_active": 0.0010, "pore_cbd": 0.0701, "active_cbd": 0.0500, "tpc_proxy": 0.043716},
}

# Table 4.6 -- active-domain continuity. CRITICAL: per the task's explicit
# warning, active connected-component count / chord length must never be
# recomputed ad hoc for panel c -- only the official summary tables or
# these verified values are used. Active Euler characteristic is
# intentionally never plotted.
TABLE46_CONTINUITY = {
    "real":     {"components": 5.66,  "chord": 14.55},
    "microlad": {"components": 56.62, "chord": 7.91},
    "slicegan": {"components": 6.02,  "chord": 14.42},
}

INTERFACE_METRIC_ORDER = ["pore_active", "pore_cbd", "active_cbd", "tpc_proxy"]
INTERFACE_TICK_LABELS = ["Pore-active", "Pore-CBD", "Active-CBD", "TPC proxy"]
INTERFACE_PAIR_TO_NAME = {(0, 1): "pore_active", (0, 2): "pore_cbd", (1, 2): "active_cbd"}

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

# Visible model names used everywhere in the figure. Internal keys ("real",
# "microlad", "slicegan") and CSV group strings ("real", "diffusion", "gan")
# are never shown.
LABELS = {"real": "Reference", "microlad": "MicroLad", "slicegan": "SliceGAN"}

LINESTYLES = {"real": "-", "microlad": "-", "slicegan": (0, (5.5, 2.2))}
LINEWIDTHS = {"real": 1.85, "microlad": 1.85, "slicegan": 1.95}
TITLE_COLOR = {"real": TEXT, "microlad": COLORS["microlad"], "slicegan": COLORS["slicegan"]}
MARKERS = {"real": "o", "microlad": "D", "slicegan": "^"}

CARD_LW = 0.8
CELL_LW = 1.6

# Official phase convention -- never swapped: 0=pore, 1=active, 2=CBD.
# Phase -> grayscale value (0=pore near-black, 1=active near-white, 2=CBD mid-gray).
PHASE_GRAY_VALUES = np.array([0.05, 0.94, 0.55])
PHASE_NAMES = ["Pore", "Active", "CBD"]

# ============================================================================
# 3. GEOMETRY (figure fractions) -- reused verbatim from Figure 4.1
# ============================================================================

card_a = [0.045, 0.365, 0.380, 0.570]
card_b = [0.445, 0.365, 0.510, 0.570]
card_c = [0.045, 0.012, 0.910, 0.290]

PANEL_LABEL_OFFSET = 0.030

assert abs(card_a[1] - card_b[1]) < 1e-12, "panel a/b y mismatch"
assert abs(card_a[3] - card_b[3]) < 1e-12, "panel a/b height mismatch"
assert abs(card_c[0] - card_a[0]) < 1e-12, "panel c left edge mismatch"
assert abs((card_c[0] + card_c[2]) - (card_b[0] + card_b[2])) < 1e-12, "panel c right edge mismatch"

# ============================================================================
# 4. GENERIC HELPERS
# ============================================================================


def log(*a):
    print(*a, flush=True)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _path_is_excluded(p: Path) -> bool:
    s = _norm(str(p))
    return any(tok in s for tok in EXCLUDE_PATH_TOKENS)


def map_csv_group(value) -> str | None:
    """Exact mapping of the official CSV 'group' strings (real/diffusion/gan)
    onto internal keys. No fuzzy matching -- the task specifies these exact
    strings."""
    if value is None:
        return None
    return CSV_GROUP_MAP.get(str(value).strip().lower())


# ============================================================================
# 5. VOLUME LOADING (exact official standardized-input folders only)
# ============================================================================


def resolve_volume_dir(group: str) -> Path:
    d = GROUP_VOLUME_DIRS[group]
    if not d.exists():
        raise FileNotFoundError(
            f"[data:{group}] expected exact official folder not found: {d}\n"
            f"This script does not search broadly for sample folders -- only "
            f"evaluation_4_2_all_metrics_inputs/<real|diffusion|gan> is used, per task instructions."
        )
    if _path_is_excluded(d):
        raise RuntimeError(f"[data:{group}] resolved folder {d} matches an excluded pattern "
                            f"({EXCLUDE_PATH_TOKENS}) -- refusing to use it")
    return d


def load_raw_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        arr = tiff.imread(str(path))
    return np.squeeze(np.asarray(arr))


def remap_labels(arr: np.ndarray) -> np.ndarray:
    """Map onto 0=pore, 1=active, 2=CBD by ascending raw value -- the
    official convention. Never label-swaps."""
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
    d = resolve_volume_dir(group)
    files = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VOLUME_EXTS)
    if not files:
        raise FileNotFoundError(f"[data:{group}] no volume files (.npy/.tif/.tiff) in {d}")

    ext_counts: dict = {}
    for f in files:
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

    log(f"[data:{group}] folder = {d}")
    log(f"[data:{group}] n_volumes = {len(files)}  extensions = {ext_counts}")

    use_files = files[:MAX_VOLUMES_PER_GROUP]
    if len(use_files) < len(files):
        log(f"[data:{group}] capping at {MAX_VOLUMES_PER_GROUP} of {len(files)} volumes")

    volumes: dict = {}
    fracs = []
    shape_counter: Counter = Counter()
    for i, f in enumerate(use_files):
        raw = load_raw_array(f)
        shape_counter[tuple(raw.shape)] += 1
        if raw.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"[data:{group}] volume {f} has shape {raw.shape}, expected exactly {EXPECTED_SHAPE}. "
                f"Stopping WITHOUT saving the figure -- {d} should contain only standardized "
                f"64^3 metric-input volumes."
            )
        if i < 2:
            log(f"[data:{group}] {f.name}: unique raw labels = {np.unique(raw)}")
        rem = remap_labels(raw)
        if i < 2:
            log(f"[data:{group}] {f.name}: unique remapped labels = {np.unique(rem)}")
        volumes[f] = rem
        fracs.append(phase_fractions(rem))

    mean_fracs = {k: float(np.mean([fr[k] for fr in fracs])) for k in ("pore", "active", "cbd")}
    log(f"[data:{group}] mean phase fractions: "
        f"pore={mean_fracs['pore']:.6f} active={mean_fracs['active']:.6f} cbd={mean_fracs['cbd']:.6f}")

    return {"dir": d, "files": files, "ext_counts": ext_counts, "shape_counts": dict(shape_counter),
            "volumes": volumes, "mean_fracs": mean_fracs}


def check_phase_fractions(group: str, mean_fracs: dict) -> None:
    target = TABLE45_PHASE_FRACTIONS[group]
    bad = [(k, mean_fracs[k], target[k]) for k in ("pore", "active", "cbd")
           if abs(mean_fracs[k] - target[k]) > PHASE_TOL]
    if bad:
        detail = ", ".join(f"{k}: got {g:.6f} vs expected {t:.6f}" for k, g, t in bad)
        raise RuntimeError(
            f"[sanity-check] group '{group}' phase fractions deviate from the expected "
            f"evaluation_4_2_all_metrics_inputs means by more than {PHASE_TOL} absolute ({detail}). "
            f"This usually means the wrong folder was used or the label mapping is off. "
            f"Stopping rather than building a figure from suspect data."
        )
    log(f"[sanity-check] group '{group}' OK (tolerance {PHASE_TOL}): {mean_fracs}")


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
    grid.cell_data["phase"] = np.transpose(vol, (2, 1, 0)).flatten(order="F")

    slices = grid.slice_orthogonal(x=mx, y=my, z=mz)
    phase_cmap = ListedColormap([to_hex((g, g, g)) for g in PHASE_GRAY_VALUES])

    pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX))
    pl.set_background("white")
    pl.add_mesh(slices, scalars="phase", cmap=phase_cmap, clim=[0, 2], show_scalar_bar=False)
    pl.add_mesh(pv.Box(bounds=(0, nx, 0, ny, 0, nz)), style="wireframe",
                color=COLORS[group], line_width=2.4, opacity=0.65)

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
    inside a group-colored wireframe box, zoomed so the cube fills most of
    its cell (this was too small in an earlier revision)."""
    nz, ny, nx = vol.shape
    mz, my, mx = nz // 2, ny // 2, nx // 2

    fig = plt.figure(figsize=(CANVAS_PX / 200.0, CANVAS_PX / 200.0), dpi=200)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_proj_type("ortho")

    X, Y = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    Z = np.full_like(X, mz, dtype=float)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                     facecolors=_phase_gray_rgba(vol[mz, :, :]).transpose(1, 0, 2),
                     shade=False, linewidth=0, antialiased=False)

    X2, Z2 = np.meshgrid(np.arange(nx + 1), np.arange(nz + 1), indexing="ij")
    Y2 = np.full_like(X2, my, dtype=float)
    ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1,
                     facecolors=_phase_gray_rgba(vol[:, my, :]).transpose(1, 0, 2),
                     shade=False, linewidth=0, antialiased=False)

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
                color=COLORS[group], linewidth=1.8, alpha=0.9)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    try:
        # Matplotlib >= 3.6: zoom > 1 fills more of the axes with the cube --
        # this is the fix for the cube rendering too small.
        ax.set_box_aspect((nx, ny, nz), zoom=1.5)
    except TypeError:
        ax.set_box_aspect((nx, ny, nz))
        try:
            ax.dist = 7.0
        except Exception:
            pass
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)

    fig.savefig(out_raw, transparent=True, dpi=200)
    plt.close(fig)


def render_group_cube(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float) -> bool:
    try:
        render_orthoslice_cube_pyvista(vol, group, out_raw, parallel_scale)
        return True
    except Exception as exc:
        log(f"[render:{group}] PyVista orthoslice-cube unavailable/failed -> Matplotlib fallback ({exc})")
        render_orthoslice_cube_matplotlib(vol, group, out_raw)
        return False


def finalize_renders(raw_paths, out_paths, pad_frac=0.035):
    """Common alpha crop across all groups -> identical scale and centring.
    pad_frac tightened from Figure 4.1's 0.06 so the orthoslice cube fills
    more of its cell."""
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
# 8. OFFICIAL METRIC LOADING (panel b curves) -- no CSV search, no FFT
# fallback: these files are guaranteed to exist per task instructions. If
# one is genuinely missing, the corresponding panel shows "metric
# unavailable" -- it is never silently replaced by a recomputation.
# ============================================================================


def _require_columns(df: pd.DataFrame, required: set, path: Path, label: str) -> bool:
    missing = required - set(df.columns)
    if missing:
        log(f"[{label}] {path} missing expected columns: {sorted(missing)}; "
            f"available columns: {list(df.columns)}")
        return False
    return True


def load_tpc_curve(kind: str):
    """kind: 'same_active' (phase_i=1,phase_j=1) or 'cross_pore_active'
    (phase pairs 0-1 / 1-0 combined)."""
    if not TPC_CSV.exists():
        log(f"[curve:{kind}] MISSING official file: {TPC_CSV}")
        return None
    df = pd.read_csv(TPC_CSV)
    if not _require_columns(df, {"group", "sample_id", "descriptor", "axis",
                                  "phase_i", "phase_j", "r", "value"}, TPC_CSV, f"curve:{kind}"):
        return None

    df["_group"] = df["group"].map(map_csv_group)
    pi = pd.to_numeric(df["phase_i"], errors="coerce")
    pj = pd.to_numeric(df["phase_j"], errors="coerce")
    if kind == "same_active":
        sub = df[(pi == 1) & (pj == 1)]
    elif kind == "cross_pore_active":
        sub = df[((pi == 0) & (pj == 1)) | ((pi == 1) & (pj == 0))]
    else:
        raise ValueError(kind)
    sub = sub[sub["_group"].notna()]
    if sub.empty:
        log(f"[curve:{kind}] no rows matched the phase filter / group mapping in {TPC_CSV}")
        return None

    r = pd.to_numeric(sub["r"], errors="coerce")
    val = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.assign(_r=r, _val=val).dropna(subset=["_r", "_val"])
    agg = sub.groupby(["_group", "_r"])["_val"].agg(["mean", "std"]).reset_index()

    out = {}
    for g in GROUPS:
        s = agg[agg["_group"] == g].sort_values("_r")
        if s.empty:
            log(f"[curve:{kind}] group '{g}' has no rows in {TPC_CSV}")
            continue
        out[g] = {"x": s["_r"].to_numpy(float), "y": s["mean"].to_numpy(float),
                   "std": np.nan_to_num(s["std"].to_numpy(float)) if g == "real" else None}
    if len(out) < 2:
        return None
    log(f"[curve:{kind}] loaded from official CSV: {TPC_CSV} ({len(sub)} rows, "
        f"group-mean vs r, not raw per-sample rows)")
    return out


def load_lineal_curve():
    if not LINEAL_CSV.exists():
        log(f"[curve:lineal] MISSING official file: {LINEAL_CSV}")
        return None
    df = pd.read_csv(LINEAL_CSV)
    if not _require_columns(df, {"group", "sample_id", "descriptor", "axis",
                                  "phase", "r", "value"}, LINEAL_CSV, "curve:lineal"):
        return None

    df["_group"] = df["group"].map(map_csv_group)
    phase = pd.to_numeric(df["phase"], errors="coerce")
    sub = df[(phase == 1) & df["_group"].notna()]
    if sub.empty:
        log(f"[curve:lineal] no rows matched phase==1 / group mapping in {LINEAL_CSV}")
        return None

    r = pd.to_numeric(sub["r"], errors="coerce")
    val = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.assign(_r=r, _val=val).dropna(subset=["_r", "_val"])
    agg = sub.groupby(["_group", "_r"])["_val"].agg(["mean", "std"]).reset_index()

    out = {}
    for g in GROUPS:
        s = agg[agg["_group"] == g].sort_values("_r")
        if s.empty:
            log(f"[curve:lineal] group '{g}' has no rows in {LINEAL_CSV}")
            continue
        out[g] = {"x": s["_r"].to_numpy(float), "y": s["mean"].to_numpy(float),
                   "std": np.nan_to_num(s["std"].to_numpy(float)) if g == "real" else None}
    if len(out) < 2:
        return None
    log(f"[curve:lineal] loaded from official CSV: {LINEAL_CSV} ({len(sub)} rows, "
        f"group-mean vs r, not raw per-sample rows)")
    return out


def _active_phase_mask(df: pd.DataFrame) -> pd.Series:
    """True where a row is the active phase, accepting either an integer
    'phase' column (==1) or a 'phase_name' column (=='active')."""
    mask = pd.Series(False, index=df.index)
    if "phase" in df.columns:
        mask = mask | (pd.to_numeric(df["phase"], errors="coerce") == 1)
    if "phase_name" in df.columns:
        mask = mask | (df["phase_name"].astype(str).str.lower() == "active")
    return mask


def load_local_heterogeneity_hist():
    if not LOCAL_FRAC_CSV.exists():
        log(f"[curve:local_heterogeneity] MISSING official file: {LOCAL_FRAC_CSV}")
        return None
    df = pd.read_csv(LOCAL_FRAC_CSV)
    if not _require_columns(df, {"group", "sample_id", "block_size", "block_id", "z", "y", "x",
                                  "phase", "phase_name", "local_fraction"},
                             LOCAL_FRAC_CSV, "curve:local_heterogeneity"):
        return None

    df["_group"] = df["group"].map(map_csv_group)
    sub = df[_active_phase_mask(df) & df["_group"].notna()]
    if sub.empty:
        log(f"[curve:local_heterogeneity] no rows matched phase==1/'active' / group mapping "
            f"in {LOCAL_FRAC_CSV}")
        return None

    edges = np.linspace(0, 1, LOCAL_HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for g in GROUPS:
        vals = pd.to_numeric(sub.loc[sub["_group"] == g, "local_fraction"],
                              errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            log(f"[curve:local_heterogeneity] group '{g}' has no rows in {LOCAL_FRAC_CSV}")
            continue
        hist, _ = np.histogram(vals, bins=edges, density=True)
        out[g] = {"x": centers, "y": hist, "std": None}
    if len(out) < 2:
        return None
    log(f"[curve:local_heterogeneity] loaded from official CSV: {LOCAL_FRAC_CSV} "
        f"({len(sub)} block rows pooled per group, {LOCAL_HIST_BINS}-bin density histogram)")
    return out


def resolve_curve(kind):
    """No computed fallback for panel-b curves: if the official CSV is
    missing/unparseable, the panel shows 'metric unavailable'."""
    if kind == "active_tpc":
        return load_tpc_curve("same_active"), f"official {TPC_CSV.name}, phase_i=1 & phase_j=1"
    if kind == "cross_tpc":
        return load_tpc_curve("cross_pore_active"), f"official {TPC_CSV.name}, phase pairs 0-1/1-0 combined"
    if kind == "lineal":
        return load_lineal_curve(), f"official {LINEAL_CSV.name}, phase=1"
    if kind == "local_heterogeneity":
        return load_local_heterogeneity_hist(), f"official {LOCAL_FRAC_CSV.name} histogram, phase=1"
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
# 9. PANEL C -- interface hierarchy fingerprint + active-domain continuity
# (official summary tables only; Table 4.5/4.6 fallback if genuinely missing)
# ============================================================================


def load_interface_hierarchy():
    if not INTERFACE_CSV.exists() or not TPC_PROXY_CSV.exists():
        for p in (INTERFACE_CSV, TPC_PROXY_CSV):
            if not p.exists():
                log(f"[panel-c-left] MISSING official file: {p}")
        return None

    idf = pd.read_csv(INTERFACE_CSV)
    tdf = pd.read_csv(TPC_PROXY_CSV)
    if not _require_columns(idf, {"group", "axis", "phase_i", "phase_j", "interface_density_mean",
                                   "interface_density_std", "interface_density_median"},
                             INTERFACE_CSV, "panel-c-left"):
        return None
    if not _require_columns(tdf, {"group", "n_samples", "triple_phase_contact_count_mean",
                                   "triple_phase_contact_count_std", "triple_phase_contact_count_median",
                                   "triple_phase_contact_density_mean", "triple_phase_contact_density_std",
                                   "triple_phase_contact_density_median"}, TPC_PROXY_CSV, "panel-c-left"):
        return None

    idf = idf[idf["axis"].astype(str).str.lower() == "mean3d"].copy()
    if idf.empty:
        log(f"[panel-c-left] no axis=='mean3d' rows in {INTERFACE_CSV}")
        return None
    idf["_group"] = idf["group"].map(map_csv_group)
    idf["_pair"] = [tuple(sorted((int(a), int(b))))
                    for a, b in zip(pd.to_numeric(idf["phase_i"], errors="coerce"),
                                     pd.to_numeric(idf["phase_j"], errors="coerce"))]
    tdf = tdf.copy()
    tdf["_group"] = tdf["group"].map(map_csv_group)

    out = {}
    for g in GROUPS:
        row_g = idf[idf["_group"] == g]
        rec = {}
        for pair, key in INTERFACE_PAIR_TO_NAME.items():
            r = row_g[row_g["_pair"] == pair]
            if r.empty:
                log(f"[panel-c-left] missing interface pair {pair} for group '{g}' in {INTERFACE_CSV}")
                return None
            rec[key] = float(pd.to_numeric(r.iloc[0]["interface_density_mean"], errors="coerce"))
        tr = tdf[tdf["_group"] == g]
        if tr.empty:
            log(f"[panel-c-left] missing group '{g}' in {TPC_PROXY_CSV}")
            return None
        rec["tpc_proxy"] = float(pd.to_numeric(tr.iloc[0]["triple_phase_contact_density_mean"], errors="coerce"))
        out[g] = rec

    log(f"[panel-c-left] loaded from official CSVs: {INTERFACE_CSV.name} (axis=mean3d) + {TPC_PROXY_CSV.name}")
    return out


def load_active_continuity():
    """CRITICAL: never recomputes connected components / chord length from
    volumes -- only these official summary tables or the Table 4.6
    fallback."""
    if not CONNECTIVITY_CSV.exists() or not CHORD_CSV.exists():
        for p in (CONNECTIVITY_CSV, CHORD_CSV):
            if not p.exists():
                log(f"[panel-c-right] MISSING official file: {p}")
        return None

    cdf = pd.read_csv(CONNECTIVITY_CSV)
    ldf = pd.read_csv(CHORD_CSV)
    if "group" not in cdf.columns or "component_count_mean" not in cdf.columns:
        log(f"[panel-c-right] {CONNECTIVITY_CSV} missing required columns; available: {list(cdf.columns)}")
        return None
    if "group" not in ldf.columns or "chord_mean_mean" not in ldf.columns:
        log(f"[panel-c-right] {CHORD_CSV} missing required columns; available: {list(ldf.columns)}")
        return None

    cdf = cdf[_active_phase_mask(cdf)].copy()
    cdf["_group"] = cdf["group"].map(map_csv_group)
    ldf = ldf[_active_phase_mask(ldf)].copy()
    ldf["_group"] = ldf["group"].map(map_csv_group)

    out = {}
    for g in GROUPS:
        cr = cdf[cdf["_group"] == g]
        if cr.empty:
            log(f"[panel-c-right] missing group '{g}' (phase=active) in {CONNECTIVITY_CSV}")
            return None
        comp = float(pd.to_numeric(cr.iloc[0]["component_count_mean"], errors="coerce"))

        lr = ldf[ldf["_group"] == g]
        if lr.empty:
            log(f"[panel-c-right] missing group '{g}' (phase=active) in {CHORD_CSV}")
            return None
        # CHORD_CSV has per-axis (x/y/z) rows, not mean3d -> average across axes.
        chord = float(pd.to_numeric(lr["chord_mean_mean"], errors="coerce").mean())

        out[g] = {"components": comp, "chord": chord}

    log(f"[panel-c-right] loaded from official CSVs: {CONNECTIVITY_CSV.name} (phase=active) + "
        f"{CHORD_CSV.name} (phase=active, averaged across axis rows)")
    return out


def resolve_interface_hierarchy():
    data = load_interface_hierarchy()
    if data is not None:
        return data, False, f"official {INTERFACE_CSV.name} (axis=mean3d) + {TPC_PROXY_CSV.name}"
    log("[panel-c-left] FALLBACK: using verified Table 4.5/4.6 values because the official "
        f"summary table(s) could not be loaded/parsed ({INTERFACE_CSV.name}, {TPC_PROXY_CSV.name})")
    return TABLE46_INTERFACE, True, "Table 4.5/4.6 fallback"


def resolve_active_continuity():
    data = load_active_continuity()
    if data is not None:
        return data, False, f"official {CONNECTIVITY_CSV.name} + {CHORD_CSV.name} (phase=active)"
    log("[panel-c-right] FALLBACK: using verified Table 4.6 group means because the official "
        f"summary table(s) could not be loaded/parsed ({CONNECTIVITY_CSV.name}, {CHORD_CSV.name}); "
        "never recomputed ad hoc from volumes, per task instructions")
    return TABLE46_CONTINUITY, True, "Table 4.6 fallback"


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
    for g in GROUPS:
        if g not in data:
            log(f"[panel-c-right] group '{g}' missing from active-continuity data")

    # Default label offset is up-right; a pair of near-coincident points
    # (Reference/SliceGAN are typically very close here) gets pushed apart
    # in opposite diagonal directions so the labels never overlap.
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
# 10. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT      = {PROJECT}")
    log(f"[paths] METRICS_DIR  = {METRICS_DIR}")
    log(f"[paths] OUT          = {OUT}")
    for g in GROUPS:
        log(f"[paths] volume dir[{g}] = {GROUP_VOLUME_DIRS[g]}")

    # ---- data: load exact official volumes, assert shape, sanity-check ----
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
    parallel_scale = 0.60 * max(max(rep[g]["vol"].shape) for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}
    for g in GROUPS:
        render_group_cube(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    finalize_renders(raw_paths, png_paths)

    # ---- panel b: curves (official CSVs only, no computed fallback) -------
    curves_active_tpc, src_active_tpc = resolve_curve("active_tpc")
    curves_cross_tpc, src_cross_tpc = resolve_curve("cross_tpc")
    curves_lineal, src_lineal = resolve_curve("lineal")
    curves_local_het, src_local_het = resolve_curve("local_heterogeneity")

    # ---- panel c: interface hierarchy + active-domain continuity ---------
    interface_data, interface_is_fallback, src_interface = resolve_interface_hierarchy()
    continuity_data, continuity_is_fallback, src_continuity = resolve_active_continuity()

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
    item_widths = [0.070, 0.082, 0.062]
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

    # =========================== SUMMARY ========================================
    log("\nVolume folders:")
    for g in GROUPS:
        log(f"  {g}: {GROUP_VOLUME_DIRS[g]}")

    log("\nShapes:")
    for g in GROUPS:
        log(f"  {g}: {groups_data[g]['shape_counts']}")

    log("\nPhase means:")
    for g in GROUPS:
        fr = groups_data[g]["mean_fracs"]
        log(f"  {g}: pore={fr['pore']:.6f} active={fr['active']:.6f} cbd={fr['cbd']:.6f}")

    log("\nMetric sources:")
    log(f"  active TPC              : {src_active_tpc if curves_active_tpc else 'MISSING -> metric unavailable'}")
    log(f"  pore-active cross TPC    : {src_cross_tpc if curves_cross_tpc else 'MISSING -> metric unavailable'}")
    log(f"  active lineal            : {src_lineal if curves_lineal else 'MISSING -> metric unavailable'}")
    log(f"  local active heterogeneity: {src_local_het if curves_local_het else 'MISSING -> metric unavailable'}")
    log(f"  interface hierarchy      : {src_interface}")
    log(f"  active continuity        : {src_continuity}")

    fallback_notes = []
    if not curves_active_tpc:
        fallback_notes.append(f"active TPC unavailable: {TPC_CSV} missing/unparseable -> panel shows "
                               f"'metric unavailable' (no computed fallback used)")
    if not curves_cross_tpc:
        fallback_notes.append(f"cross TPC unavailable: {TPC_CSV} missing/unparseable -> panel shows "
                               f"'metric unavailable' (no computed fallback used)")
    if not curves_lineal:
        fallback_notes.append(f"lineal path unavailable: {LINEAL_CSV} missing/unparseable -> panel shows "
                               f"'metric unavailable' (no computed fallback used)")
    if not curves_local_het:
        fallback_notes.append(f"local heterogeneity unavailable: {LOCAL_FRAC_CSV} missing/unparseable -> "
                               f"panel shows 'metric unavailable' (no computed fallback used)")
    if interface_is_fallback:
        fallback_notes.append(f"panel-c-left used Table 4.5/4.6 fallback: {INTERFACE_CSV} and/or "
                               f"{TPC_PROXY_CSV} missing/unparseable")
    if continuity_is_fallback:
        fallback_notes.append(f"panel-c-right used Table 4.6 fallback: {CONNECTIVITY_CSV} and/or "
                               f"{CHORD_CSV} missing/unparseable")

    log("\nFallbacks used:")
    if fallback_notes:
        for note in fallback_notes:
            log(f"  - {note}")
    else:
        log("  none -- all panels used the official metric files.")

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out):
        log(" ", p.resolve())


if __name__ == "__main__":
    sys.exit(main())
