#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.4 composite (magma, final)
Reference vs AB-CDM vs SurVol -- anisotropy preservation on the COMMON100
glass benchmark.

Same visual contract as Figures 4.1-4.3 (make_fig4_1_composite_magma_final.py,
make_fig4_2_composite_magma_final.py, make_fig4_3_composite_magma_final.py):
same cards, fonts, panel-label style, export block. No tight_layout /
constrained_layout / bbox_inches.

Scientific message: statistical/orientational anisotropy is NOT equivalent
to connected/transport-sensitive anisotropy. AB-CDM preserves dominant
directional morphology well but loses transverse connectivity; SurVol
preserves transverse connectivity but weakens directional morphology
magnitude and over-connects phase 1.

Only the exact official COMMON100 standardized-input volume folders and the
exact official COMMON100 metric CSVs are used -- no broad path discovery,
no NATIVE128, no abcdm_real_check, no calibration/exploratory/candidate
cohorts. If an official source cannot be loaded or interpreted
unambiguously, the script raises and does not save a figure.
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
from matplotlib.patches import Rectangle, Patch
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG
# ============================================================================

# Exact project root -- per task instructions this is hardcoded rather than
# auto-detected, and the script never searches /home/ra2 broadly for an
# alternative Section 4.4 project.
PROJECT = Path("/home/ra2/4_4_survol_glass_anisotropy_clean")

STANDARDIZED = PROJECT / "outputs" / "section44_full_metrics" / "00_standardized_inputs" / "COMMON100"
REAL_DIR = STANDARDIZED / "real"
DIFFUSION_DIR = STANDARDIZED / "diffusion"
GAN_DIR = STANDARDIZED / "gan"

CORE_DIR = PROJECT / "outputs" / "section44_full_metrics" / "01_core_anisotropy" / "COMMON100"
SCALAR_GROUP_SUMMARY = CORE_DIR / "SCALAR_GROUP_SUMMARY.csv"
SCALAR_MODEL_ERRORS = CORE_DIR / "SCALAR_MODEL_ERRORS_VS_REAL.csv"
DIRECTIONAL_SUMMARY = CORE_DIR / "GROUP_DIRECTIONAL_ANISOTROPY_SUMMARY.csv"
CURVE_GROUP_MEANS = CORE_DIR / "CURVE_GROUP_MEANS.csv"
CURVE_MODEL_ERRORS = CORE_DIR / "CURVE_MODEL_ERRORS_VS_REAL.csv"

FD_TRANSPORT = PROJECT / "outputs" / "section44_full_metrics" / "02_fd_transport" / "COMMON100_FD_TRANSPORT_PER_AXIS.csv"
TENSOR_ERRORS = PROJECT / "outputs" / "section44_full_metrics" / "04_aggregated" / "TENSOR_AND_PRINCIPAL_DIRECTION_ERRORS.csv"

OUT = PROJECT / "paper_figures_4_4" / "figure_4_4_composite_magma_final"
TMP = OUT / "_render_cache"
LOGS = PROJECT / "logs"
STEM = "fig4_4_composite_magma_final"

GROUPS = ["real", "diffusion", "gan"]
GROUP_VOLUME_DIRS = {"real": REAL_DIR, "diffusion": DIFFUSION_DIR, "gan": GAN_DIR}

EXPECTED_SHAPE = (100, 100, 100)
EXPECTED_N = 50

# Defense-in-depth: even though every path above is hardcoded exactly per
# task instructions (no broad search is ever performed), this guards
# against an accidental future edit resolving into one of the explicitly
# forbidden cohorts.
EXCLUDE_PATH_TOKENS = ("abcdmrealcheck", "native128", "final128", "microlad",
                       "ddim", "true2dto3drecon", "calibration", "smoketest",
                       "preflight", "debug", "temporary", "candidate",
                       "sourcebundle", "10sample")

RENDER_PX = 1500
CANVAS_PX = 900

FIG_W, FIG_H = 17.8, 10.2

# ============================================================================
# 1b. SANITY-CHECK CONSTANTS (verified reference values; used only to
# cross-check independently loaded data -- never as the plotted source)
# ============================================================================

PHASE_FRAC_TOL = 0.02
TRANSITION_REL_TOL = 0.15
TRANSITION_ABS_FLOOR = 0.0015
NONPERC_DSTAR_TOL = 1e-6
# FD group-axis means vary with the underlying model checkpoint / sampling
# more than a single-descriptor mean would, so this sanity band is
# deliberately generous (roughly factor-of-2): it exists to catch a wrong
# folder / wrong phase filter / wrong axis mapping, not to reproduce the
# manuscript numbers exactly.
FD_SANITY_REL_TOL = 1.0
FD_SANITY_ABS_FLOOR = 0.05

EXPECTED_PHASE_FRACTION_1 = {"real": 0.559, "diffusion": 0.489, "gan": 0.552}

EXPECTED_TRANSITIONS = {
    "real":      {"x": 0.090708, "y": 0.093561, "z": 0.009729},
    "diffusion": {"x": 0.092157, "y": 0.095021, "z": 0.016363},
    "gan":       {"x": 0.089684, "y": 0.094044, "z": 0.018043},
}

EXPECTED_FD = {
    "real":      {"perc": {"x": 0.78, "y": 0.74, "z": 1.00},
                  "dstar": {"x": 0.028781, "y": 0.044450, "z": 0.542944}},
    "diffusion": {"perc": {"x": 0.14, "y": 0.06, "z": 1.00},
                  "dstar": {"x": 0.000975, "y": 0.001006, "z": 0.464835}},
    "gan":       {"perc": {"x": 1.00, "y": 1.00, "z": 1.00},
                  "dstar": {"x": 0.066027, "y": 0.054630, "z": 0.481973}},
}

EXPECTED_TENSOR_ERR_DEG = {"diffusion": 0.083276, "gan": 1.890547}

# ============================================================================
# 2. STYLE -- identical contract to Figures 4.1-4.3
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

COLORS = {"real": "#8C93A1", "diffusion": "#F2A93B", "gan": "#772A8E"}
LABELS = {"real": "Reference", "diffusion": "AB-CDM", "gan": "SurVol"}
LINESTYLES = {"real": "-", "diffusion": "-", "gan": (0, (5.5, 2.2))}
LINEWIDTHS = {"real": 1.85, "diffusion": 1.85, "gan": 1.95}
MARKERS = {"real": "o", "diffusion": "D", "gan": "^"}
TITLE_COLOR = {"real": TEXT, "diffusion": COLORS["diffusion"], "gan": COLORS["gan"]}

CARD_LW = 0.8
CELL_LW = 1.6

# ============================================================================
# 3. GEOMETRY (figure fractions)
#
# Panel a (3 groups x {3D, XY, XZ, YZ}) and panel b (a 2x2 plot block) are
# structurally the same shapes as Figure 4.2's panel a / panel b, so the
# proven card_a / card_b geometry is reused directly. Panel c keeps the
# card_c footprint but widens its bottom padding relative to 4.1-4.3,
# since the transport dot-plot needs room for a small log-scale caption
# under the axis in addition to the X/Y/Z tick labels.
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
    s = _norm(p.name)
    return any(tok in s for tok in EXCLUDE_PATH_TOKENS)


def _coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer, float, np.floating)):
        if x in (0, 1):
            return bool(x)
        return np.nan
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return np.nan


def _status_is_failure(s) -> bool:
    s = str(s).lower()
    return any(tok in s for tok in ("fail", "error", "invalid", "exception", "timeout"))


# ============================================================================
# 5. VOLUME LOADING + VALIDATION (exact official COMMON100 folders only)
# ============================================================================


def load_group_volumes(group: str) -> dict:
    d = GROUP_VOLUME_DIRS[group]
    if not d.exists():
        raise FileNotFoundError(
            f"[data:{group}] expected exact official COMMON100 folder not found: {d}\n"
            f"This script never searches broadly for alternative Section 4.4 sample folders."
        )
    if _path_is_excluded(d):
        raise RuntimeError(f"[data:{group}] resolved folder {d} matches a forbidden pattern "
                            f"({EXCLUDE_PATH_TOKENS}) -- refusing to use it")

    files = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff"))
    if len(files) != EXPECTED_N:
        raise RuntimeError(f"[data:{group}] expected exactly {EXPECTED_N} TIFF volumes in {d}, "
                            f"found {len(files)}")

    volumes = {}
    for f in files:
        arr = np.squeeze(tiff.imread(str(f)))
        if arr.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"[data:{group}] {f.name}: shape {arr.shape} != expected "
                                f"{EXPECTED_SHAPE}. Stopping WITHOUT saving the figure.")
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"[data:{group}] {f.name}: contains NaN/Inf values")
        vals = set(np.unique(arr).tolist())
        if vals != {0, 1}:
            raise RuntimeError(f"[data:{group}] {f.name}: unique labels {sorted(vals)} -- "
                                f"expected exactly {{0,1}} with both phases present")
        volumes[f] = arr.astype(np.uint8)

    log(f"[data:{group}] folder = {d}")
    log(f"[data:{group}] n_volumes = {len(files)}  shape={EXPECTED_SHAPE}  "
        f"labels={{0,1}} (validated, both phases present, finite)")
    return volumes


# Physical axis mapping for v(z,y,x): x = numpy axis 2, y = numpy axis 1,
# z = numpy axis 0. Never changed.


def transition_rates(v: np.ndarray):
    tx = float(np.mean(v[:, :, 1:] != v[:, :, :-1]))
    ty = float(np.mean(v[:, 1:, :] != v[:, :-1, :]))
    tz = float(np.mean(v[1:, :, :] != v[:-1, :, :]))
    return tx, ty, tz


def phase1_fraction(v: np.ndarray) -> float:
    return float(np.mean(v == 1))


def compute_group_descriptors(volumes: dict) -> list:
    rows = []
    for f, v in volumes.items():
        tx, ty, tz = transition_rates(v)
        rows.append({"file": f, "phi1": phase1_fraction(v), "tx": tx, "ty": ty, "tz": tz})
    return rows


# ============================================================================
# 6. SANITY CHECKS AGAINST OFFICIAL CSVs (direct-TIFF vs. official values)
# ============================================================================


def check_phase_fraction(group: str, direct_mean_phi1: float, scalar_df: pd.DataFrame):
    row = scalar_df[(scalar_df["group"] == group) & (scalar_df["metric"] == "phase_fraction_1")]
    if row.empty:
        raise RuntimeError(f"[sanity:phase_fraction] no 'phase_fraction_1' row for group "
                            f"'{group}' in {SCALAR_GROUP_SUMMARY}")
    official = float(row.iloc[0]["mean"])
    if abs(direct_mean_phi1 - official) > PHASE_FRAC_TOL:
        raise RuntimeError(
            f"[sanity:phase_fraction] group '{group}': direct TIFF mean={direct_mean_phi1:.6f} "
            f"vs official mean={official:.6f} (tol {PHASE_FRAC_TOL}) -- stopping without saving")
    exp = EXPECTED_PHASE_FRACTION_1[group]
    log(f"[sanity:phase_fraction] group '{group}' OK: direct={direct_mean_phi1:.6f} "
        f"official={official:.6f} (manuscript reference ~{exp})")


def check_transitions(group: str, direct_mean_txyz, dir_df: pd.DataFrame):
    row = dir_df[dir_df["group"] == group]
    if row.empty:
        raise RuntimeError(f"[sanity:transitions] no row for group '{group}' in {DIRECTIONAL_SUMMARY}")
    row = row.iloc[0]
    official = (float(row["mean_transition_x"]), float(row["mean_transition_y"]),
                float(row["mean_transition_z"]))
    for axis, d_val, o_val in zip("xyz", direct_mean_txyz, official):
        tol = max(TRANSITION_REL_TOL * abs(o_val), TRANSITION_ABS_FLOOR)
        if abs(d_val - o_val) > tol:
            raise RuntimeError(
                f"[sanity:transitions] group '{group}' axis {axis}: direct={d_val:.6f} vs "
                f"official={o_val:.6f} (tol {tol:.6f}) -- axis mapping or data source appears "
                f"wrong, stopping without saving")
    log(f"[sanity:transitions] group '{group}' OK: direct=(tx={direct_mean_txyz[0]:.6f}, "
        f"ty={direct_mean_txyz[1]:.6f}, tz={direct_mean_txyz[2]:.6f})  official=(tx={official[0]:.6f}, "
        f"ty={official[1]:.6f}, tz={official[2]:.6f})")


# ============================================================================
# 7. REPRESENTATIVE-SAMPLE SELECTION (independent per group, no pairing
# assumption, deterministic)
# ============================================================================


def choose_representative(group: str, rows: list):
    files = [r["file"] for r in rows]
    M = np.array([[r["phi1"], r["tx"], r["ty"], r["tz"]] for r in rows], dtype=float)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = (M - mu) / sd_safe
    target = np.median(Z, axis=0)
    d = np.linalg.norm(Z - target, axis=1)
    dmin = float(d.min())
    tol = 1e-9 * max(1.0, abs(dmin))
    tied = sorted((i for i in range(len(rows)) if d[i] <= dmin + tol), key=lambda i: files[i].name)
    k = tied[0]
    r = rows[k]
    log(f"[rep:{group}] selected {r['file'].name}  phi1={r['phi1']:.6f}  tx={r['tx']:.6f} "
        f"ty={r['ty']:.6f} tz={r['tz']:.6f}  distance-to-median={d[k]:.6f}"
        + ("  (tie-break: lexicographically first filename)" if len(tied) > 1 else ""))
    return r["file"], float(d[k])


# ============================================================================
# 8. CARD / LABEL / AXIS PRIMITIVES -- reused verbatim from Figures 4.1-4.3
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


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


# ============================================================================
# 9. 3D RENDERING (panel a) -- true marching-cubes isosurface of phase 1
# from the raw, unsmoothed representative volume. No MIP / voxel fallback:
# if PyVista + scikit-image marching_cubes cannot run, the script fails
# loudly rather than silently substituting another representation.
# ============================================================================


def render_isosurface(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float):
    import pyvista as pv
    from skimage import measure

    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    v = (vol == 1).astype(np.float32)
    if v.min() >= 0.5 or v.max() <= 0.5:
        raise RuntimeError(f"[render:{group}] degenerate iso-level for phase 1 -- volume is "
                            f"single-phase after binarization, cannot extract a level=0.5 isosurface")

    verts, faces, _, _ = measure.marching_cubes(v, level=0.5)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, np.int64), faces.astype(np.int64)])
    mesh = pv.PolyData(verts, faces_pv)

    nz, ny, nx = vol.shape
    center = np.array([nz / 2.0, ny / 2.0, nx / 2.0])

    pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX))
    pl.set_background("white")
    pl.add_mesh(mesh, color=COLORS[group], opacity=0.90,
                smooth_shading=True, specular=0.18, specular_power=14,
                ambient=0.24, diffuse=0.82, show_scalar_bar=False)
    pl.add_mesh(pv.Box(bounds=(0, nz, 0, ny, 0, nx)), style="wireframe",
                color=COLORS[group], line_width=2.2, opacity=0.55)

    # identical camera + identical parallel scale across groups => comparable
    # size, no cropping; same convention as Figures 4.1/4.2.
    pl.enable_parallel_projection()
    direction = np.array([1.0, -1.30, 0.90])
    direction /= np.linalg.norm(direction)
    pl.camera.focal_point = tuple(center)
    pl.camera.position = tuple(center + direction * 4.0 * max(vol.shape))
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.parallel_scale = float(parallel_scale)

    pl.screenshot(str(out_raw), transparent_background=True)
    pl.close()


def render_group_isosurface(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float):
    try:
        render_isosurface(vol, group, out_raw, parallel_scale)
    except Exception as exc:
        raise RuntimeError(
            f"[render:{group}] true 3D isosurface rendering failed and no fallback is permitted "
            f"for this figure (PyVista + scikit-image marching_cubes are required): {exc}"
        ) from exc


def finalize_renders(raw_paths, out_paths, pad_frac=0.06):
    """Common alpha crop across all groups -> identical scale and centring."""
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
# 10. OFFICIAL METRIC LOADING -- exact official COMMON100 CSVs only. No CSV
# search, no hardcoded plotted values (the EXPECTED_* constants above are
# used only for sanity-check comparisons against what is loaded here).
# ============================================================================


def load_scalar_group_summary() -> pd.DataFrame:
    if not SCALAR_GROUP_SUMMARY.exists():
        raise FileNotFoundError(f"[scalar] missing official file: {SCALAR_GROUP_SUMMARY}")
    df = pd.read_csv(SCALAR_GROUP_SUMMARY)
    required = {"benchmark", "group", "metric", "n", "mean", "std"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[scalar] {SCALAR_GROUP_SUMMARY} missing expected columns: {sorted(missing)}")
    sub = df[(df["benchmark"] == "COMMON100") & (df["group"].isin(GROUPS))].copy()
    if sub.empty:
        raise RuntimeError(f"[scalar] no COMMON100 rows for real/diffusion/gan in {SCALAR_GROUP_SUMMARY}")
    log(f"[scalar] loaded {SCALAR_GROUP_SUMMARY.name}: {len(sub)} COMMON100 rows "
        f"(abcdm_real_check excluded)")
    return sub


def load_directional_summary() -> pd.DataFrame:
    if not DIRECTIONAL_SUMMARY.exists():
        raise FileNotFoundError(f"[directional] missing official file: {DIRECTIONAL_SUMMARY}")
    df = pd.read_csv(DIRECTIONAL_SUMMARY)
    required = {"benchmark", "group", "mean_transition_x", "mean_transition_y", "mean_transition_z"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[directional] {DIRECTIONAL_SUMMARY} missing expected columns: "
                            f"{sorted(missing)}")
    sub = df[(df["benchmark"] == "COMMON100") & (df["group"].isin(GROUPS))].copy()
    if sub.empty:
        raise RuntimeError(f"[directional] no COMMON100 rows for real/diffusion/gan in "
                            f"{DIRECTIONAL_SUMMARY}")
    log(f"[directional] loaded {DIRECTIONAL_SUMMARY.name}: {len(sub)} COMMON100 rows")
    return sub


AUTOCOV_METRIC_NAMES = {axis: f"normalized_autocovariance_phase1_{axis}" for axis in ("x", "y", "z")}


def load_autocovariance_curves() -> dict:
    if not CURVE_GROUP_MEANS.exists():
        raise FileNotFoundError(f"[curves] missing official file: {CURVE_GROUP_MEANS}")
    df = pd.read_csv(CURVE_GROUP_MEANS)
    required = {"benchmark", "group", "metric_family", "metric_name", "phase", "axis",
                "coord", "mean", "std", "count"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[curves] {CURVE_GROUP_MEANS} missing expected columns: {sorted(missing)}")

    base = df[(df["benchmark"] == "COMMON100") & (df["group"].isin(GROUPS))].copy()
    if base.empty:
        raise RuntimeError(f"[curves] no COMMON100 rows for real/diffusion/gan in {CURVE_GROUP_MEANS}")

    out = {}
    for axis, metric_name in AUTOCOV_METRIC_NAMES.items():
        sub = base[base["metric_name"] == metric_name].copy()
        if sub.empty:
            raise RuntimeError(f"[curves] metric_name '{metric_name}' not found in {CURVE_GROUP_MEANS}")

        bad_family = sub.loc[sub["metric_family"] != "normalized_autocovariance", "metric_family"].unique()
        if len(bad_family):
            raise RuntimeError(f"[curves] '{metric_name}' rows with unexpected metric_family: "
                                f"{bad_family.tolist()}")
        bad_phase = sub.loc[sub["phase"] != "phase1", "phase"].unique()
        if len(bad_phase):
            raise RuntimeError(f"[curves] '{metric_name}' rows with unexpected phase: {bad_phase.tolist()}")
        bad_axis = sub.loc[sub["axis"] != axis, "axis"].unique()
        if len(bad_axis):
            raise RuntimeError(f"[curves] '{metric_name}' rows with unexpected axis field: "
                                f"{bad_axis.tolist()} (expected '{axis}')")

        per_group, grid_by_group = {}, {}
        for g in GROUPS:
            gsub = sub[sub["group"] == g].sort_values("coord")
            if gsub.empty:
                raise RuntimeError(f"[curves] '{metric_name}': group '{g}' missing")
            coord = pd.to_numeric(gsub["coord"], errors="coerce").to_numpy(float)
            mean = pd.to_numeric(gsub["mean"], errors="coerce").to_numpy(float)
            std = pd.to_numeric(gsub["std"], errors="coerce").to_numpy(float)
            count = pd.to_numeric(gsub["count"], errors="coerce").to_numpy(float)
            if not np.all(np.isfinite(coord) & np.isfinite(mean) & np.isfinite(std)):
                raise RuntimeError(f"[curves] '{metric_name}' group '{g}': non-finite coord/mean/std")
            if not np.all(count == EXPECTED_N):
                raise RuntimeError(f"[curves] '{metric_name}' group '{g}': count != {EXPECTED_N} "
                                    f"at some coordinate(s)")
            per_group[g] = {"x": coord, "y": mean, "std": std}
            grid_by_group[g] = coord

        ref_grid = grid_by_group["real"]
        for g in GROUPS:
            if grid_by_group[g].shape != ref_grid.shape or not np.allclose(grid_by_group[g], ref_grid):
                raise RuntimeError(
                    f"[curves] '{metric_name}': coordinate grid for group '{g}' differs from "
                    f"'real' -- refusing to interpolate, stopping without saving")

        out[axis] = per_group
        log(f"[curves] resolved metric_name='{metric_name}'  n_coords={ref_grid.size}  "
            f"groups={list(per_group.keys())}  count=={EXPECTED_N} at every coordinate")
    return out


def load_curve_model_errors_optional():
    """Validation/logging only -- never a substitute for the group-mean
    curves loaded above."""
    if not CURVE_MODEL_ERRORS.exists():
        log(f"[curves:errors] optional validation file not found (skipping): {CURVE_MODEL_ERRORS}")
        return
    try:
        df = pd.read_csv(CURVE_MODEL_ERRORS)
    except Exception as exc:
        log(f"[curves:errors] optional validation file could not be read ({exc}) -- skipping")
        return
    if "metric_family" not in df.columns or "phase" not in df.columns:
        return
    sub = df[(df["metric_family"] == "normalized_autocovariance") & (df["phase"] == "phase1")]
    for _, r in sub.iterrows():
        log(f"[curves:errors] model={r.get('model')} axis={r.get('axis')} "
            f"rmse={r.get('rmse')} mae={r.get('mae')}")


def load_fd_transport():
    if not FD_TRANSPORT.exists():
        raise FileNotFoundError(f"[fd] missing official file: {FD_TRANSPORT}")
    df = pd.read_csv(FD_TRANSPORT)
    required = {"group", "sample_index", "file", "axis", "axis_index", "phase",
                "percolates_6conn", "effective_diffusivity_proxy", "status"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[fd] {FD_TRANSPORT} missing expected columns: {sorted(missing)}")

    sub = df[(df["group"].isin(GROUPS)) & (df["phase"].astype(str) == "phase1") &
             (df["axis"].astype(str).isin(["x", "y", "z"]))].copy()
    if len(sub) != 450:
        raise RuntimeError(f"[fd] expected exactly 450 filtered rows (3 groups x 50 samples x "
                            f"3 axes), got {len(sub)}")
    for g in GROUPS:
        for axis in ("x", "y", "z"):
            n = len(sub[(sub["group"] == g) & (sub["axis"] == axis)])
            if n != 50:
                raise RuntimeError(f"[fd] group '{g}' axis '{axis}': expected 50 rows, got {n}")

    perc_bool = sub["percolates_6conn"].map(_coerce_bool)
    if perc_bool.isna().any():
        bad = sub.loc[perc_bool.isna(), "percolates_6conn"].unique()
        raise RuntimeError(f"[fd] percolates_6conn has non-Boolean-interpretable values: {bad.tolist()}")
    sub = sub.assign(_percolates=perc_bool.astype(bool))

    dstar = pd.to_numeric(sub["effective_diffusivity_proxy"], errors="coerce")
    if dstar.isna().any() or not np.isfinite(dstar.to_numpy()).all():
        raise RuntimeError("[fd] effective_diffusivity_proxy has non-numeric/non-finite values")
    if (dstar < 0).any():
        raise RuntimeError("[fd] effective_diffusivity_proxy has negative values")
    sub = sub.assign(_dstar=dstar)

    bad_status = sub.loc[sub["status"].map(_status_is_failure), "status"].unique()
    if len(bad_status):
        raise RuntimeError(f"[fd] {len(bad_status)} distinct solver-failure status value(s) present: "
                            f"{sorted(bad_status.tolist())} -- stopping")
    log(f"[fd] status values observed: {sorted(sub['status'].astype(str).unique().tolist())}")

    nonperc = sub[~sub["_percolates"]]
    bad_nonperc = nonperc[nonperc["_dstar"].abs() > NONPERC_DSTAR_TOL]
    if len(bad_nonperc):
        raise RuntimeError(f"[fd] {len(bad_nonperc)} nonpercolating row(s) have "
                            f"effective_diffusivity_proxy != 0 beyond tolerance {NONPERC_DSTAR_TOL}")

    perc_agg = sub.groupby(["group", "axis"])["_percolates"].mean()
    dstar_agg = sub.groupby(["group", "axis"])["_dstar"].mean()
    log(f"[fd] loaded {FD_TRANSPORT.name}: {len(sub)} rows (phase1, groups real/diffusion/gan, "
        f"axes x/y/z)")
    return perc_agg, dstar_agg


def check_fd_sanity(perc_agg, dstar_agg):
    for g in GROUPS:
        for axis in ("x", "y", "z"):
            got_p = float(perc_agg[(g, axis)])
            exp_p = EXPECTED_FD[g]["perc"][axis]
            tol_p = max(FD_SANITY_REL_TOL * exp_p, FD_SANITY_ABS_FLOOR)
            if abs(got_p - exp_p) > tol_p:
                raise RuntimeError(f"[sanity:fd] group '{g}' axis '{axis}' percolating fraction "
                                    f"{got_p:.4f} far from expected ~{exp_p:.2f} (tol {tol_p:.3f})")
            got_d = float(dstar_agg[(g, axis)])
            exp_d = EXPECTED_FD[g]["dstar"][axis]
            tol_d = max(FD_SANITY_REL_TOL * exp_d, 1e-4)
            if abs(got_d - exp_d) > tol_d:
                raise RuntimeError(f"[sanity:fd] group '{g}' axis '{axis}' D* {got_d:.6f} far from "
                                    f"expected ~{exp_d:.6f} (tol {tol_d:.6f})")
    log("[sanity:fd] percolation / D* group-axis means within the sanity band of the manuscript "
        "reference values for all 9 group-axis pairs")


def load_tensor_errors() -> dict:
    if not TENSOR_ERRORS.exists():
        raise FileNotFoundError(f"[tensor] missing official file: {TENSOR_ERRORS}")
    df = pd.read_csv(TENSOR_ERRORS)
    required = {"tensor", "model", "frobenius_error", "principal_direction_error_deg"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[tensor] {TENSOR_ERRORS} missing expected columns: {sorted(missing)}")

    sub = df[df["tensor"] == "acf3d_second_moment_tensor"]
    out = {}
    for model in ("diffusion", "gan"):
        r = sub[sub["model"] == model]
        if len(r) != 1:
            raise RuntimeError(f"[tensor] expected exactly 1 row for tensor="
                                f"'acf3d_second_moment_tensor' model='{model}', got {len(r)} -- "
                                f"ambiguous, stopping")
        val = float(r.iloc[0]["principal_direction_error_deg"])
        if not np.isfinite(val):
            raise RuntimeError(f"[tensor] model='{model}' principal_direction_error_deg is not finite")
        out[model] = val

    log(f"[tensor] ACF principal-direction error: AB-CDM={out['diffusion']:.6f} deg "
        f"(manuscript ref ~{EXPECTED_TENSOR_ERR_DEG['diffusion']}) | SurVol={out['gan']:.6f} deg "
        f"(manuscript ref ~{EXPECTED_TENSOR_ERR_DEG['gan']})")
    if not (out["diffusion"] < out["gan"]):
        log("[sanity:tensor] WARNING: expected AB-CDM ACF orientation error < SurVol's -- got the "
            "opposite; verify the source table (angle metrics are sensitive, so this warns rather "
            "than stopping)")
    return out


# ============================================================================
# 11. PANEL A -- directional morphology (3D volume + XY/XZ/YZ sections)
# ============================================================================

ROW_NAMES = ["3D volume", "X–Y slice", "X–Z slice", "Y–Z slice"]

# Small, quiet direction cues -- shown only in the leftmost (Reference)
# column so the convention reads once instead of being repeated 9 times.
DIRECTION_LABELS = {
    "X–Y slice": [("→ X", (0.90, 0.07), "right", "bottom"), ("↑ Y", (0.07, 0.90), "left", "top")],
    "X–Z slice": [("→ X", (0.90, 0.07), "right", "bottom"), ("↑ Z", (0.07, 0.90), "left", "top")],
    "Y–Z slice": [("→ Y", (0.90, 0.07), "right", "bottom"), ("↑ Z", (0.07, 0.90), "left", "top")],
}


def build_panel_a(fig, card, rep, png_paths):
    ax_, ay_, aw_, ah_ = card
    pad_x, pad_top, pad_bot = 0.016, 0.038, 0.034
    label_w, gap_x, gap_y = 0.062, 0.018, 0.009

    free_w_in = (aw_ - 2 * pad_x - label_w - 3 * gap_x) * FIG_W / 3.0
    free_h_in = (ah_ - pad_top - pad_bot - 3 * gap_y) * FIG_H / 4.0
    cell_in = min(free_w_in, free_h_in)
    cell_w, cell_h = cell_in / FIG_W, cell_in / FIG_H

    grid_w = label_w + 3 * gap_x + 3 * cell_w
    grid_h = 4 * cell_h + 3 * gap_y
    gx0 = ax_ + (aw_ - grid_w) / 2.0
    gy_top = ay_ + ah_ - pad_top - max(0.0, (ah_ - pad_top - pad_bot - grid_h) / 2.0)

    def row_y(r):
        return gy_top - (r + 1) * cell_h - r * gap_y

    for r, name in enumerate(ROW_NAMES):
        a = fig.add_axes([gx0, row_y(r), label_w, cell_h])
        a.axis("off")
        a.text(0.98, 0.5, name, ha="right", va="center", fontsize=8.8, fontweight="bold", color=TEXT)

    for c, g in enumerate(GROUPS):
        x = gx0 + label_w + gap_x + c * (cell_w + gap_x)
        fig.text(x + cell_w / 2.0, gy_top + 0.010, LABELS[g], ha="center", va="bottom",
                 fontsize=9.6, fontweight="bold", color=TITLE_COLOR[g])

        a = fig.add_axes([x, row_y(0), cell_w, cell_h])
        a.imshow(Image.open(png_paths[g]), interpolation="bilinear")
        image_cell(a, COLORS[g])
        a.text(0.035, 0.035, f"$\\varphi_1$ = {rep[g]['phi1']:.3f}",
               transform=a.transAxes, fontsize=6.8, color=SUBTEXT,
               ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))

        v = rep[g]["vol"]
        zc, yc, xc = v.shape[0] // 2, v.shape[1] // 2, v.shape[2] // 2
        # XY: v[zc,:,:] -> horizontal=X, vertical=Y. XZ: v[:,yc,:] -> horizontal=X,
        # vertical=Z. YZ: v[:,:,xc] -> horizontal=Y, vertical=Z. origin="lower"
        # makes each vertical axis increase upward -- no transpose needed.
        slices = [v[zc, :, :], v[:, yc, :], v[:, :, xc]]
        for r, sl in enumerate(slices):
            a = fig.add_axes([x, row_y(r + 1), cell_w, cell_h])
            a.imshow(sl, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            image_cell(a, COLORS[g])
            if c == 0:
                for txt, (tx, ty), ha, va in DIRECTION_LABELS[ROW_NAMES[r + 1]]:
                    a.text(tx, ty, txt, transform=a.transAxes, fontsize=6.9, fontweight="bold",
                           color="white", ha=ha, va=va,
                           bbox=dict(facecolor=TEXT, edgecolor="none", alpha=0.55, pad=1.2))

    # small neutral-grayscale phase legend, centered under the grid
    legend_y = ay_ + 0.015
    item_widths = [0.078, 0.078]
    total_w = sum(item_widths)
    lx = gx0 + label_w + gap_x + (grid_w - label_w - gap_x - total_w) / 2.0
    for name, color, w in zip(["phase 0", "phase 1"], ["#000000", "#FFFFFF"], item_widths):
        fig.add_artist(Rectangle((lx, legend_y - 0.006), 0.013, 0.011,
                                  transform=fig.transFigure, facecolor=color,
                                  edgecolor=SPINE, linewidth=0.6, zorder=101))
        fig.text(lx + 0.019, legend_y, name, ha="left", va="center",
                 fontsize=7.4, color=SUBTEXT, zorder=101)
        lx += w


# ============================================================================
# 12. PANEL B -- statistical / orientational anisotropy (2x2 block)
# ============================================================================

TRANSITION_AXES = ["x", "y", "z"]


def plot_directional_transitions(ax, scalar_df):
    style_axis(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title("Directional transitions", pad=4.5, color=TEXT, fontweight="bold")
    ax.set_ylabel("Transition rate", color=SUBTEXT)

    n_groups = len(GROUPS)
    group_w = 0.72
    bar_w = group_w / n_groups
    xs = np.arange(len(TRANSITION_AXES))

    max_top = 0.0
    for gi, g in enumerate(GROUPS):
        means, stds = [], []
        for axis in TRANSITION_AXES:
            row = scalar_df[(scalar_df["group"] == g) & (scalar_df["metric"] == f"transition_rate_{axis}")]
            if row.empty:
                raise RuntimeError(f"[panel-b-bars] missing transition_rate_{axis} for group '{g}' "
                                    f"in {SCALAR_GROUP_SUMMARY}")
            means.append(float(row.iloc[0]["mean"]))
            stds.append(float(row.iloc[0]["std"]))
        offset = (gi - (n_groups - 1) / 2.0) * bar_w
        ax.bar(xs + offset, means, yerr=stds, width=bar_w * 0.88, color=COLORS[g],
               edgecolor="black", linewidth=0.7, label=LABELS[g],
               error_kw=dict(ecolor=TEXT, elinewidth=1.0, capsize=2.2))
        max_top = max(max_top, max(m + s for m, s in zip(means, stds)))

    ax.set_xticks(xs)
    ax.set_xticklabels(["X", "Y", "Z"], fontsize=7.6)
    # Transition rates are non-negative -- start the axis at a true zero
    # baseline, with modest headroom above the tallest mean+std error bar.
    ax.set_ylim(0.0, max_top * 1.18)

    # Single model legend for all of panel b -- placed here, upper-right,
    # so the 2x2 block carries exactly one legend (see plot_autocovariance).
    leg = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=SPINE,
                     framealpha=0.94, handlelength=1.6, borderpad=0.4, labelspacing=0.3, fontsize=6.8)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(8)


def plot_autocovariance(ax, curves, title, ylim, xlim, show_ylabel, show_legend):
    style_axis(ax)
    ax.set_title(title, pad=4.5, color=TEXT, fontweight="bold")
    ax.set_xlabel("Lag (vox)", color=SUBTEXT)
    if show_ylabel:
        ax.set_ylabel("Normalized autocovariance", color=SUBTEXT)

    for g in GROUPS:
        d = curves[g]
        x, y, std = d["x"], d["y"], d["std"]
        if g == "real" and np.any(std > 0):
            ax.fill_between(x, y - std, y + std, color=COLORS[g], alpha=0.20, linewidth=0, zorder=1)
        ax.plot(x, y, color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                 solid_capstyle="round", label=LABELS[g], zorder=3 if g == "real" else 4)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.margins(x=0.02)
    ax.set_ylim(*ylim)

    if show_legend:
        leg = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=SPINE,
                         framealpha=0.94, handlelength=2.3, borderpad=0.45, labelspacing=0.34)
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(8)


def build_panel_b(fig, card, scalar_df, autocov):
    bx_, by_, bw_, bh_ = card
    b_pl, b_pr, b_pt, b_pb = 0.045, 0.018, 0.040, 0.046
    b_gx, b_gy = 0.058, 0.076

    plot_w = (bw_ - b_pl - b_pr - b_gx) / 2.0
    plot_h = (bh_ - b_pt - b_pb - b_gy) / 2.0

    axes = {}
    for i, key in enumerate(["bars", "x", "y", "z"]):
        r, c = divmod(i, 2)
        x = bx_ + b_pl + c * (plot_w + b_gx)
        y = by_ + bh_ - b_pt - (r + 1) * plot_h - r * b_gy
        axes[key] = fig.add_axes([x, y, plot_w, plot_h])

    plot_directional_transitions(axes["bars"], scalar_df)

    # Identical y-limits (and x-limits, since the coordinate grids match by
    # construction -- see load_autocovariance_curves) across all three
    # normalized-autocovariance subplots, so magnitudes are visually comparable.
    all_y = np.concatenate([autocov[axis][g]["y"] for axis in ("x", "y", "z") for g in GROUPS])
    y_span = all_y.max() - all_y.min()
    y_pad = 0.06 * (y_span if y_span > 0 else 1.0)
    ylim = (all_y.min() - y_pad, all_y.max() + y_pad)
    # A shared x-limit across all three subplots is only meaningful if the
    # x/y/z coordinate grids actually coincide (they are only guaranteed
    # identical *within* an axis, across the three groups -- see
    # load_autocovariance_curves). If the grids differ across axes, each
    # subplot keeps its own natural x-extent instead of a forced shared one.
    grids_match_across_axes = (
        autocov["x"]["real"]["x"].shape == autocov["y"]["real"]["x"].shape == autocov["z"]["real"]["x"].shape
        and np.allclose(autocov["x"]["real"]["x"], autocov["y"]["real"]["x"])
        and np.allclose(autocov["x"]["real"]["x"], autocov["z"]["real"]["x"])
    )
    if grids_match_across_axes:
        all_x = autocov["x"]["real"]["x"]
        x_span = all_x.max() - all_x.min()
        x_pad = 0.02 * (x_span if x_span > 0 else 1.0)
        xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
        log("[panel-b-curves] x/y/z coordinate grids coincide -> using identical x-limits "
            "across all three normalized-autocovariance subplots")
    else:
        xlim = None
        log("[panel-b-curves] x/y/z coordinate grids differ -> each normalized-autocovariance "
            "subplot keeps its own x-extent")

    # No per-subplot legend here -- the single panel-b model legend now lives
    # in the Directional transitions subplot (upper-right).
    plot_autocovariance(axes["x"], autocov["x"], "Normalized autocovariance — X", ylim, xlim,
                         show_ylabel=False, show_legend=False)
    plot_autocovariance(axes["y"], autocov["y"], "Normalized autocovariance — Y", ylim, xlim,
                         show_ylabel=True, show_legend=False)
    plot_autocovariance(axes["z"], autocov["z"], "Normalized autocovariance — Z", ylim, xlim,
                         show_ylabel=False, show_legend=False)


# ============================================================================
# 13. PANEL C -- connected / transport-sensitive anisotropy
# ============================================================================


def plot_percolation(ax, perc_agg):
    style_axis(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title("Phase-1 directional percolation", pad=4.5, color=TEXT, fontweight="bold")
    ax.set_ylabel("Percolating fraction", color=SUBTEXT)

    n_groups = len(GROUPS)
    group_w = 0.72
    bar_w = group_w / n_groups
    xs = np.arange(3)

    for gi, g in enumerate(GROUPS):
        vals = [float(perc_agg[(g, axis)]) for axis in ("x", "y", "z")]
        offset = (gi - (n_groups - 1) / 2.0) * bar_w
        ax.bar(xs + offset, vals, width=bar_w * 0.88, color=COLORS[g],
               edgecolor="black", linewidth=0.7, label=LABELS[g])
        for bx, v in zip(xs + offset, vals):
            ax.text(bx, v + 0.03, f"{100 * v:.0f}%", ha="center", va="bottom",
                    fontsize=6.4, color=SUBTEXT)

    ax.set_xticks(xs)
    ax.set_xticklabels(["X", "Y", "Z"], fontsize=7.6)
    # Extra headroom above the tallest (100%) bars so every percentage label
    # sits fully inside the plotting area with visible whitespace above it.
    ax.set_ylim(0, 1.15)


DSTAR_MARKER_OFFSET = {"real": -0.16, "diffusion": 0.0, "gan": 0.16}


def plot_transport(ax, dstar_agg):
    style_axis(ax)
    ax.set_title("Phase-1 structural transport", pad=4.5, color=TEXT, fontweight="bold")
    ax.set_ylabel(r"Effective diffusivity proxy $D^{*}$", color=SUBTEXT)
    ax.set_yscale("log")

    xs = np.arange(3)
    all_vals = np.array([float(dstar_agg[(g, axis)]) for g in GROUPS for axis in ("x", "y", "z")])
    positive = all_vals[all_vals > 0]

    # Data-driven log-space padding: the smallest positive group mean sits
    # well above the bottom spine and the largest sits well below the top
    # spine, so no marker (including the AB-CDM X/Y diamonds) reads as
    # touching the frame. Values/data are never altered -- only the axis
    # range is chosen from them.
    if positive.size:
        lower_y = float(positive.min()) / 1.9
        upper_y = float(positive.max()) * 1.5
    else:
        lower_y, upper_y = 1e-4, 1.0
    ax.set_ylim(lower_y, upper_y)
    # Display-only floor for exact-zero D* markers (see below) -- kept
    # safely inside the axis range set above rather than at its edge.
    floor = lower_y * 1.2

    for g in GROUPS:
        vals = np.array([float(dstar_agg[(g, axis)]) for axis in ("x", "y", "z")])
        xpos = xs + DSTAR_MARKER_OFFSET[g]
        is_zero = vals <= 0
        y_plot = np.where(is_zero, floor, vals)
        if (~is_zero).any():
            ax.scatter(xpos[~is_zero], y_plot[~is_zero], s=70, marker=MARKERS[g], color=COLORS[g],
                       edgecolors="black", linewidths=0.9, zorder=5)
        if is_zero.any():
            # True zeros are not representable on a log axis -- shown as an
            # open marker pinned to a display-only floor line, explicitly
            # annotated "0" so it is never mistaken for a small positive value.
            ax.scatter(xpos[is_zero], y_plot[is_zero], s=70, marker=MARKERS[g],
                       facecolors="none", edgecolors=COLORS[g], linewidths=1.3, zorder=5)
            for xp in xpos[is_zero]:
                ax.annotate("0", (xp, floor), textcoords="offset points", xytext=(0, -8),
                            fontsize=6.2, color=SUBTEXT, ha="center", va="top")

    ax.set_xticks(xs)
    ax.set_xticklabels(["X", "Y", "Z"], fontsize=7.6)
    ax.set_xlim(-0.55, 2.55)
    # The "log scale; D*=0 ..." note is drawn by build_panel_c, positioned
    # just below this axis, so its vertical placement can be tuned together
    # with the rest of panel c's geometry.


def build_panel_c(fig, card, perc_agg, dstar_agg):
    cx_, cy_, cw_, ch_ = card
    header_title_y = cy_ + ch_ - 0.020
    # The card heading and the subplot titles must read as two clearly
    # separate hierarchy levels (not a two-line title), but the previous gap
    # (header_title_y - 0.058) left slightly too much empty space above the
    # plots. Both axes are shifted upward by AXES_SHIFT_UP (~1.5% of the
    # total figure height) -- the gap shrinks a bit while the plot height
    # (plot_top - plot_bottom) is unchanged -- landing between that overly
    # roomy version and the original cramped one (gap was 0.030 there).
    AXES_SHIFT_UP = 0.016
    plot_top = header_title_y - 0.058 + AXES_SHIFT_UP
    c_pl, c_pr, c_pb, c_gx = 0.045, 0.020, 0.044, 0.055
    plot_bottom = cy_ + c_pb + AXES_SHIFT_UP
    plot_h_c = plot_top - plot_bottom
    plot_w_c = (cw_ - c_pl - c_pr - c_gx) / 2.0
    left_x = cx_ + c_pl
    right_x = left_x + plot_w_c + c_gx

    fig.text(cx_ + 0.014, header_title_y, "Connected and transport-sensitive anisotropy",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    legend_handles = [Patch(facecolor=COLORS[g], edgecolor=COLORS[g], alpha=0.85, label=LABELS[g])
                      for g in GROUPS]
    fig.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(cx_ + cw_ - 0.010, header_title_y + 0.006),
              bbox_transform=fig.transFigure, ncol=3, frameon=True,
              facecolor="white", edgecolor=SPINE, framealpha=0.94,
              borderpad=0.45, labelspacing=0.3, columnspacing=1.1,
              handlelength=1.4, handletextpad=0.5, fontsize=7.4).get_frame().set_linewidth(0.6)

    ax_left = fig.add_axes([left_x, plot_bottom, plot_w_c, plot_h_c])
    ax_right = fig.add_axes([right_x, plot_bottom, plot_w_c, plot_h_c])

    plot_percolation(ax_left, perc_agg)
    plot_transport(ax_right, dstar_agg)

    # D*=0 explanatory note -- kept, but moved up close under the right
    # (structural-transport) subplot's x-axis, inside the card, instead of
    # sitting near the card's bottom border.
    fig.text(right_x + plot_w_c / 2.0, plot_bottom - 0.018,
             "log scale; nonpercolating samples contribute D*=0 to the ensemble mean",
             ha="center", va="top", fontsize=6.5, color=SUBTEXT)


# ============================================================================
# 14. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT           = {PROJECT}")
    log(f"[paths] REAL_DIR          = {REAL_DIR}")
    log(f"[paths] DIFFUSION_DIR     = {DIFFUSION_DIR}")
    log(f"[paths] GAN_DIR           = {GAN_DIR}")
    log(f"[paths] SCALAR_GROUP_SUMMARY = {SCALAR_GROUP_SUMMARY}")
    log(f"[paths] SCALAR_MODEL_ERRORS  = {SCALAR_MODEL_ERRORS}")
    log(f"[paths] DIRECTIONAL_SUMMARY  = {DIRECTIONAL_SUMMARY}")
    log(f"[paths] CURVE_GROUP_MEANS    = {CURVE_GROUP_MEANS}")
    log(f"[paths] CURVE_MODEL_ERRORS   = {CURVE_MODEL_ERRORS}")
    log(f"[paths] FD_TRANSPORT         = {FD_TRANSPORT}")
    log(f"[paths] TENSOR_ERRORS        = {TENSOR_ERRORS}")

    # ---- volumes: load, validate, compute descriptors, sanity-check -------
    scalar_df = load_scalar_group_summary()
    directional_df = load_directional_summary()

    volumes_by_group, descriptors_by_group, rep = {}, {}, {}
    for g in GROUPS:
        volumes = load_group_volumes(g)
        volumes_by_group[g] = volumes
        rows = compute_group_descriptors(volumes)
        descriptors_by_group[g] = rows

        direct_phi1 = float(np.mean([r["phi1"] for r in rows]))
        direct_txyz = (float(np.mean([r["tx"] for r in rows])),
                       float(np.mean([r["ty"] for r in rows])),
                       float(np.mean([r["tz"] for r in rows])))
        check_phase_fraction(g, direct_phi1, scalar_df)
        check_transitions(g, direct_txyz, directional_df)

        rep_file, rep_dist = choose_representative(g, rows)
        rep_row = next(r for r in rows if r["file"] == rep_file)
        rep[g] = {"file": rep_file, "vol": volumes[rep_file], "phi1": rep_row["phi1"],
                  "tx": rep_row["tx"], "ty": rep_row["ty"], "tz": rep_row["tz"], "dist": rep_dist,
                  "direct_phi1_mean": direct_phi1, "direct_txyz_mean": direct_txyz}

    # ---- panel a: 3D isosurfaces -------------------------------------------
    parallel_scale = 0.72 * max(max(rep[g]["vol"].shape) for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}
    for g in GROUPS:
        render_group_isosurface(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    finalize_renders(raw_paths, png_paths)

    # ---- panel b: official curves + scalar summary + tensor errors --------
    autocov = load_autocovariance_curves()
    load_curve_model_errors_optional()
    tensor_err = load_tensor_errors()

    # ---- panel c: official FD transport ------------------------------------
    perc_agg, dstar_agg = load_fd_transport()
    check_fd_sanity(perc_agg, dstar_agg)

    # ---- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.patch.set_facecolor(BG)

    add_card(fig, card_a)
    add_card(fig, card_b)
    add_card(fig, card_c)

    add_panel_label(fig, card_a[0] + 0.007, card_a[1] + card_a[3] + PANEL_LABEL_OFFSET, "a)")
    add_panel_label(fig, card_b[0] + 0.007, card_b[1] + card_b[3] + PANEL_LABEL_OFFSET, "b)")
    add_panel_label(fig, card_c[0] + 0.007, card_c[1] + card_c[3] + PANEL_LABEL_OFFSET, "c)")

    build_panel_a(fig, card_a, rep, png_paths)
    build_panel_b(fig, card_b, scalar_df, autocov)
    build_panel_c(fig, card_c, perc_agg, dstar_agg)

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
        log(f"  {g}: {GROUP_VOLUME_DIRS[g]}  ({EXPECTED_N} TIFFs, shape {EXPECTED_SHAPE})")

    log("\nDirect-TIFF vs. official descriptors:")
    for g in GROUPS:
        r = rep[g]
        log(f"  {g}: direct phi1_mean={r['direct_phi1_mean']:.6f}  "
            f"direct (tx,ty,tz)_mean=({r['direct_txyz_mean'][0]:.6f}, "
            f"{r['direct_txyz_mean'][1]:.6f}, {r['direct_txyz_mean'][2]:.6f})")

    log("\nRepresentative samples (independent per-group median-distance selection):")
    for g in GROUPS:
        r = rep[g]
        log(f"  {g}: {r['file'].name}  phi1={r['phi1']:.6f}  tx={r['tx']:.6f}  ty={r['ty']:.6f}  "
            f"tz={r['tz']:.6f}  distance-to-median={r['dist']:.6f}")

    log("\nNormalized-autocovariance curves resolved:")
    for axis in ("x", "y", "z"):
        n_coords = autocov[axis]["real"]["x"].size
        log(f"  axis={axis}: metric_name='{AUTOCOV_METRIC_NAMES[axis]}'  n_coords={n_coords}  "
            f"groups={GROUPS}")

    log("\nACF principal-direction error (tensor='acf3d_second_moment_tensor'):")
    log(f"  AB-CDM: {tensor_err['diffusion']:.6f} deg")
    log(f"  SurVol: {tensor_err['gan']:.6f} deg")

    log("\nPhase-1 directional percolation (mean of percolates_6conn):")
    for g in GROUPS:
        log(f"  {g}: X={float(perc_agg[(g,'x')]):.4f}  Y={float(perc_agg[(g,'y')]):.4f}  "
            f"Z={float(perc_agg[(g,'z')]):.4f}")

    log("\nPhase-1 structural transport D* (mean of effective_diffusivity_proxy, zeros included):")
    for g in GROUPS:
        log(f"  {g}: X={float(dstar_agg[(g,'x')]):.6f}  Y={float(dstar_agg[(g,'y')]):.6f}  "
            f"Z={float(dstar_agg[(g,'z')]):.6f}")

    log("\nFallbacks used: none")

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out):
        log(" ", p.resolve())


if __name__ == "__main__":
    sys.exit(main())
