#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.1 composite (magma, final)
Reference vs PoreGen/DiffSci vs True2Dto3Drecon.

Manual fixed-position layout. No tight_layout / constrained_layout / bbox_inches.
Everything is derived from real TIFF volumes and real CSV tables.
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

# Project root is auto-detected from this file's location so the script keeps
# working no matter where the checkout lives, as long as it stays under
# <PROJECT>/scripts/make_fig4_1_composite_magma_final.py
PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "4.1_final_all_metrics_gan_best_vs_poregen"
RESULTS = ROOT / "results"

OUT = ROOT / "figures_final" / "figure_4_1_composite_magma_final"
TMP = OUT / "_render_cache"
LOGS = ROOT / "logs"

STEM = "fig4_1_composite_magma_final"

# Internal folder / group keys stay unchanged; only the user-facing labels
# (LABELS, below) use the new model names.
GROUPS = ["real", "poregen", "gan_best"]

GROUP_DIRS = {
    "real": ROOT / "real_samples",
    "poregen": ROOT / "poregen_samples",
    "gan_best": ROOT / "gan_best_samples",
}

STAGE1_METRICS = RESULTS / "stage1_core_morphology_topology" / "stage1_per_sample_metrics.csv"
STAGE2_CURVES = RESULTS / "stage2_multiscale_curves" / "stage2_group_mean_curves.csv"
STAGE6_CURVES = RESULTS / "stage6_capillary_porosimetry" / "stage6b_capillary_porosimetry_group_mean_curves.csv"

FORCE_RERENDER = True          # rebuild the 3D renders from scratch
# The required model names (PoreGen/DiffSci, True2Dto3Drecon) are long enough
# that the small relative-error corner notes tend to overlap curve data in
# panel b, so they default to off; flip to True only after checking for
# clutter against the real curves.
SHOW_CURVE_ERRORS = False      # small relative-error notes inside panel-b subplots
RENDER_PX = 1500               # off-screen render resolution
CANVAS_PX = 900                # final square canvas for each 3D cell
DOWNSAMPLE = 2                 # marching-cubes decimation factor

FIG_W, FIG_H = 17.8, 10.2

# ============================================================================
# 2. STYLE
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

COLORS = {"real": "#8C93A1", "poregen": "#F2A93B", "gan_best": "#772A8E"}

# Visible model names used everywhere in the figure (legends, titles, labels).
# Internal folder / dict keys ("real", "poregen", "gan_best") are unchanged.
LABELS = {"real": "Reference", "poregen": "PoreGen/DiffSci", "gan_best": "True2Dto3Drecon"}

LINESTYLES = {"real": "-", "poregen": "-", "gan_best": (0, (5.5, 2.2))}
LINEWIDTHS = {"real": 1.85, "poregen": 1.85, "gan_best": 1.95}
TITLE_COLOR = {"real": TEXT, "poregen": COLORS["poregen"], "gan_best": COLORS["gan_best"]}
MARKERS = {"real": "o", "poregen": "D", "gan_best": "^"}

CARD_LW = 0.8
CELL_LW = 1.6

# ============================================================================
# 3. GEOMETRY (figure fractions) — the alignment contract
# ============================================================================

card_a = [0.045, 0.365, 0.380, 0.570]
card_b = [0.445, 0.365, 0.510, 0.570]
card_c = [0.045, 0.012, 0.910, 0.290]

# Vertical clearance between a panel label's top-anchor and the top edge of
# the panel it names, held identical for a), b), and c) so all three labels
# sit the same visual distance above their own panel. Kept small and, since
# card positions above are fixed, this pulls each label closer to its own
# panel than to whatever sits above it (avoids c) reading as if it belongs
# to panel a).
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

    real / reference / ground truth        -> real
    poregen / diffusion / latent / LDM      -> poregen
    gan / GAN / True2Dto3Drecon             -> gan_best
    """
    v = _norm(value)
    if not v:
        return None
    if "gan" in v or "true2d" in v or "true2dto3d" in v:
        return "gan_best"
    if "poregen" in v or "diff" in v or "ldm" in v or "latent" in v:
        return "poregen"
    if "real" in v or v.startswith("ref") or "ground" in v or "gt" == v:
        return "real"
    return None


def load_volume(path: Path) -> np.ndarray:
    """Binary pore mask, axes (z, y, x). Non-zero == pore."""
    arr = np.squeeze(tiff.imread(str(path)))
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {arr.shape} in {path}")
    return (arr > 0).astype(np.uint8)


def list_tiffs(group: str):
    d = GROUP_DIRS[group]
    files = sorted([p for p in d.glob("*") if p.suffix.lower() in (".tif", ".tiff")])
    if not files:
        raise FileNotFoundError(f"no TIFF files in {d}")
    return files


def match_file(sample_id, files):
    """Resolve a CSV sample identifier to one of the TIFF paths."""
    if sample_id is None:
        return None
    raw = str(sample_id)
    stem = Path(raw).stem
    key = _norm(stem)

    for f in files:                                   # exact
        if f.stem == stem:
            return f
    for f in files:                                   # normalised
        if _norm(f.stem) == key:
            return f
    for f in files:                                   # substring
        fk = _norm(f.stem)
        if key and (key in fk or fk in key):
            return f

    m = re.findall(r"\d+", stem)                      # trailing index
    if m:
        idx = m[-1]
        for f in files:
            fm = re.findall(r"\d+", f.stem)
            if fm and fm[-1].lstrip("0") == idx.lstrip("0"):
                return f
    return None


# ============================================================================
# 5. CARD / LABEL / AXIS PRIMITIVES
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
# 6. REPRESENTATIVE SAMPLE SELECTION
# ============================================================================

REP_METRICS = [
    "porosity",
    "pore_diameter_vox_mean",
    "interface_density_total_faces",
    "largest_component_fraction_pore",
    "disconnected_fraction_pore",
    "local_porosity_w32_std",
]


def load_stage1():
    if not STAGE1_METRICS.exists():
        log(f"[stage1] missing: {STAGE1_METRICS}")
        return None, None, None
    df = pd.read_csv(STAGE1_METRICS)
    log(f"[stage1] {STAGE1_METRICS.name}: {df.shape[0]} rows")
    log("[stage1] columns:", list(df.columns))

    gcol = pick_col(df, ["group", "set", "model", "category", "class", "source"],
                    "group", required=False)
    scol = pick_col(df, ["sample", "sample_id", "file", "filename", "name", "stem", "path", "id"],
                    "sample id", required=False)
    if gcol is None:
        log("[stage1] no group column -> stage1 cannot be used")
        return None, None, None

    df["_group"] = df[gcol].map(normalize_group)
    unknown = df.loc[df["_group"].isna(), gcol].unique()
    if len(unknown):
        log("[stage1] unmapped group values:", list(unknown))
    df = df[df["_group"].notna()].copy()
    return df, scol, gcol


def choose_representative_file(group, stage1, scol):
    """Sample closest to the group median in standardised metric space."""
    files = list_tiffs(group)

    if stage1 is not None:
        sub = stage1[stage1["_group"] == group].copy()
        avail = [m for m in REP_METRICS if m in sub.columns]
        missing = [m for m in REP_METRICS if m not in sub.columns]
        if missing:
            log(f"[rep:{group}] metrics not in CSV: {missing}")

        if len(sub) and avail:
            M = sub[avail].apply(pd.to_numeric, errors="coerce")
            keep = M.notna().all(axis=1)
            sub, M = sub[keep], M[keep]
            if len(sub):
                mu, sd = M.mean(axis=0), M.std(axis=0).replace(0, np.nan)
                Z = ((M - mu) / sd).fillna(0.0).to_numpy(float)
                target = np.median(Z, axis=0)
                d = np.linalg.norm(Z - target, axis=1)
                order = np.argsort(d)
                if scol is not None:
                    for k in order:                       # first row that resolves
                        f = match_file(sub.iloc[k][scol], files)
                        if f is not None:
                            log(f"[rep:{group}] {f.name}  (median-distance, {len(avail)} metrics)")
                            return f
                    log(f"[rep:{group}] no CSV sample id matched a TIFF -> fallback")
                else:
                    log(f"[rep:{group}] no sample-id column -> fallback")

        # fallback A: median porosity row from the CSV
        if len(sub) and "porosity" in sub.columns and scol is not None:
            p = pd.to_numeric(sub["porosity"], errors="coerce")
            k = int(np.nanargmin(np.abs(p - np.nanmedian(p))))
            f = match_file(sub.iloc[k][scol], files)
            if f is not None:
                log(f"[rep:{group}] {f.name}  (fallback: median porosity from CSV)")
                return f

    # fallback B: median porosity measured directly from the TIFFs
    por = np.array([load_volume(f).mean() for f in files])
    k = int(np.argmin(np.abs(por - np.median(por))))
    log(f"[rep:{group}] {files[k].name}  (fallback: median measured porosity)")
    return files[k]


# ============================================================================
# 7. 3D RENDERING
# ============================================================================


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def render_volume(vol, group, out_raw, parallel_scale):
    """Off-screen isosurface render to a transparent PNG. Returns True on success."""
    try:
        import pyvista as pv
        from skimage import measure
        from scipy.ndimage import gaussian_filter

        try:
            pv.start_xvfb(wait=0.2)
        except Exception:
            pass

        step = DOWNSAMPLE
        v = vol[::step, ::step, ::step].astype(np.float32)
        v = gaussian_filter(v, sigma=0.6)
        if v.min() >= 0.45 or v.max() <= 0.45:
            raise RuntimeError("degenerate iso-level")

        verts, faces, _, _ = measure.marching_cubes(v, level=0.45)
        verts = verts * step                                   # back to voxel units
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, np.int64),
                              faces.astype(np.int64)])
        mesh = pv.PolyData(verts, faces_pv)

        nz, ny, nx = vol.shape
        center = np.array([nz / 2.0, ny / 2.0, nx / 2.0])

        pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX))
        pl.set_background("white")
        pl.add_mesh(mesh, color=COLORS[group], opacity=0.85,
                    smooth_shading=True, specular=0.18, specular_power=14,
                    ambient=0.24, diffuse=0.82, show_scalar_bar=False)
        pl.add_mesh(pv.Box(bounds=(0, nz, 0, ny, 0, nx)), style="wireframe",
                    color=COLORS[group], line_width=2.2, opacity=0.55)

        # identical camera + identical parallel scale => comparable size, no cropping
        pl.enable_parallel_projection()
        direction = np.array([1.0, -1.30, 0.90])
        direction /= np.linalg.norm(direction)
        pl.camera.focal_point = tuple(center)
        pl.camera.position = tuple(center + direction * 4.0 * max(vol.shape))
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.camera.parallel_scale = float(parallel_scale)

        pl.screenshot(str(out_raw), transparent_background=True)
        pl.close()
        return True

    except Exception as exc:
        log(f"[render:{group}] PyVista unavailable/failed -> MIP fallback ({exc})")
        proj = vol.max(axis=1).astype(float)
        f = plt.figure(figsize=(4, 4), dpi=200)
        a = f.add_axes([0, 0, 1, 1])
        a.imshow(proj, cmap="magma", interpolation="bilinear")
        a.axis("off")
        f.savefig(out_raw, transparent=True, dpi=200)
        plt.close(f)
        return False


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
# 8. CURVE PLOTTING (panel b)
# ============================================================================


def curve_schema(df, name):
    return {
        "curve": pick_col(df, ["curve", "curve_name", "metric", "name", "descriptor"], f"{name}.curve"),
        "group": pick_col(df, ["group", "set", "model", "category", "class"], f"{name}.group"),
        "x": pick_col(df, ["x", "r", "radius_vox", "radius", "lag", "bin", "bin_center",
                           "chord_length", "freq", "frequency"], f"{name}.x"),
        "mean": pick_col(df, ["mean", "y_mean", "value_mean", "y", "value", "mean_value"], f"{name}.mean"),
        "std": pick_col(df, ["std", "y_std", "value_std", "sd", "stdev", "std_value"],
                        f"{name}.std", required=False),
    }


def _corner_note(ax, text, series, blocked=()):
    """Place a small note in the emptiest corner of the axes."""
    regions = {
        "ur": ((0.55, 1.00, 0.58, 1.00), (0.975, 0.955, "right", "top")),
        "ul": ((0.00, 0.45, 0.58, 1.00), (0.025, 0.955, "left", "top")),
        "lr": ((0.55, 1.00, 0.00, 0.42), (0.975, 0.045, "right", "bottom")),
        "ll": ((0.00, 0.45, 0.00, 0.42), (0.025, 0.045, "left", "bottom")),
    }
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    if xr == xl or yt == yb:
        return
    counts = {}
    for k, ((fx0, fx1, fy0, fy1), _) in regions.items():
        if k in blocked:
            counts[k] = np.inf
            continue
        n = 0
        for x, y in series:
            fx = (np.asarray(x, float) - xl) / (xr - xl)
            fy = (np.asarray(y, float) - yb) / (yt - yb)
            n += int(np.sum((fx >= fx0) & (fx <= fx1) & (fy >= fy0) & (fy <= fy1)))
        counts[k] = n
    best = min(counts, key=counts.get)
    px, py, ha, va = regions[best][1]
    ax.text(px, py, text, transform=ax.transAxes, fontsize=6.3, color=SUBTEXT,
            ha=ha, va=va, linespacing=1.35, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.4))


def plot_curve(ax, df, sch, curve, title, xlabel, ylabel, show_legend=False,
               legend_corner="ur"):
    sub = df[df[sch["curve"]].astype(str).str.strip() == curve].copy()
    style_axis(ax)
    ax.set_title(title, pad=4.5, color=TEXT, fontweight="bold")
    ax.set_xlabel(xlabel, color=SUBTEXT)
    ax.set_ylabel(ylabel, color=SUBTEXT)

    if sub.empty:
        avail = sorted(df[sch["curve"]].astype(str).unique())[:25]
        log(f"[curve] '{curve}' not found. Available (first 25): {avail}")
        ax.text(0.5, 0.5, "curve not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, color=SUBTEXT)
        return

    sub["_group"] = sub[sch["group"]].map(normalize_group)
    series, ref_xy, model_err = [], None, {}

    for g in GROUPS:
        s = sub[sub["_group"] == g].copy()
        if s.empty:
            log(f"[curve] '{curve}': group '{g}' missing")
            continue
        s = s.sort_values(sch["x"])
        x = pd.to_numeric(s[sch["x"]], errors="coerce").to_numpy(float)
        y = pd.to_numeric(s[sch["mean"]], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size == 0:
            continue

        if g == "real":
            ref_xy = (x, y)
            if sch["std"] is not None:
                sd = pd.to_numeric(s[sch["std"]], errors="coerce").to_numpy(float)[ok]
                sd = np.nan_to_num(sd)
                if np.any(sd > 0):
                    ax.fill_between(x, y - sd, y + sd, color=COLORS[g],
                                    alpha=0.20, linewidth=0, zorder=1)
                    series.append((np.r_[x, x], np.r_[y - sd, y + sd]))

        ax.plot(x, y, color=COLORS[g], linestyle=LINESTYLES[g],
                linewidth=LINEWIDTHS[g], solid_capstyle="round",
                label=LABELS[g], zorder=3 if g == "real" else 4)
        series.append((x, y))

        if g != "real" and ref_xy is not None:
            rx, ry = ref_xy
            yi = np.interp(rx, x, y)
            denom = np.mean(np.abs(ry))
            if denom > 0:
                model_err[g] = 100.0 * np.mean(np.abs(yi - ry)) / denom

    ax.margins(x=0.02, y=0.06)

    if show_legend:
        leg = ax.legend(loc="upper right", frameon=True, facecolor="white",
                        edgecolor=SPINE, framealpha=0.94, handlelength=2.3,
                        borderpad=0.45, labelspacing=0.34)
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(8)

    if SHOW_CURVE_ERRORS and model_err:
        txt = "\n".join(f"{LABELS[g]} rel. err {model_err[g]:.1f}%"
                        for g in GROUPS if g in model_err)
        blocked = (legend_corner,) if show_legend else ()
        _corner_note(ax, txt, series, blocked=blocked)


# ============================================================================
# 9. PANEL C — connectivity-sensitive distributions
# ============================================================================

# Top-to-bottom row order requested for panel c.
CONNECTIVITY_ORDER = ["real", "poregen", "gan_best"]
CONNECTIVITY_POS = {"real": 2, "poregen": 1, "gan_best": 0}


def plot_connectivity_distribution(ax, stage1, col, xlabel, rng):
    """One horizontal box + jittered-strip row per group for a single metric."""
    style_axis(ax, grid=False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8.0, labelpad=4.0)
    ax.tick_params(axis="x", labelsize=7.4)

    def _finalize_yaxis():
        # ax.boxplot() resets the y-tick locator/formatter on every call, so
        # the custom group labels must be applied last, after all boxplots.
        ax.set_yticks([CONNECTIVITY_POS[g] for g in CONNECTIVITY_ORDER])
        ax.set_yticklabels([LABELS[g] for g in CONNECTIVITY_ORDER], fontsize=7.6)
        ax.set_ylim(-0.66, 2.66)

    if stage1 is None or col not in stage1.columns:
        if stage1 is not None:
            log(f"[panel c] '{col}' not in stage1 columns")
            log("[panel c] available columns:", list(stage1.columns))
        ax.set_xticks([])
        ax.text(0.5, 0.5, f"metric unavailable: '{col}'", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.0, color=SUBTEXT)
        _finalize_yaxis()
        return

    ref_vals = pd.to_numeric(
        stage1.loc[stage1["_group"] == "real", col], errors="coerce").dropna().to_numpy(float)
    if ref_vals.size:
        ax.axvline(float(np.mean(ref_vals)), color=COLORS["real"], linestyle=(0, (1.4, 1.6)),
                   linewidth=1.2, zorder=1)

    any_data = False
    for g in CONNECTIVITY_ORDER:
        vals = pd.to_numeric(
            stage1.loc[stage1["_group"] == g, col], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            log(f"[panel c] '{col}': group '{g}' has no valid values")
            continue
        any_data = True
        p = CONNECTIVITY_POS[g]

        bp = ax.boxplot([vals], positions=[p], vert=False, widths=0.50,
                        patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color=TEXT, linewidth=1.2),
                        whiskerprops=dict(color=COLORS[g], linewidth=1.0),
                        capprops=dict(color=COLORS[g], linewidth=1.0),
                        boxprops=dict(facecolor=COLORS[g], edgecolor=COLORS[g],
                                      alpha=0.28, linewidth=1.0))
        for b in bp["boxes"]:
            b.set_zorder(2)

        jitter = p + (rng.random(vals.size) - 0.5) * 0.30
        ax.scatter(vals, jitter, s=10.0, color=COLORS[g], alpha=0.60,
                   edgecolors="none", zorder=3)

        ax.scatter([np.mean(vals)], [p], marker=MARKERS[g], s=95, color=COLORS[g],
                   edgecolors="black", linewidths=1.0, zorder=5)

    _finalize_yaxis()

    if not any_data:
        ax.set_xticks([])
        ax.text(0.5, 0.5, f"no valid values for '{col}'", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.0, color=SUBTEXT)


# ============================================================================
# 10. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT = {PROJECT}")
    log(f"[paths] ROOT    = {ROOT}")

    # ---- data ---------------------------------------------------------------
    stage1, scol, _ = load_stage1()

    rep = {}
    for g in GROUPS:
        f = choose_representative_file(g, stage1, scol)
        vol = load_volume(f)
        rep[g] = {"file": f, "vol": vol, "porosity": float(vol.mean())}
        log(f"[vol:{g}] {f.name}  shape={vol.shape}  phi={rep[g]['porosity']:.4f}")

    parallel_scale = 0.72 * max(max(rep[g]["vol"].shape) for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}

    need = FORCE_RERENDER or not all(p.exists() for p in png_paths.values())
    if need:
        for g in GROUPS:
            render_volume(rep[g]["vol"], g, raw_paths[g], parallel_scale)
        finalize_renders(raw_paths, png_paths)
    else:
        log("[render] using cache")

    stage2 = pd.read_csv(STAGE2_CURVES) if STAGE2_CURVES.exists() else pd.DataFrame()
    stage6 = pd.read_csv(STAGE6_CURVES) if STAGE6_CURVES.exists() else pd.DataFrame()
    for nm, d, p in (("stage2", stage2, STAGE2_CURVES), ("stage6", stage6, STAGE6_CURVES)):
        if d.empty:
            log(f"[{nm}] missing or empty: {p}")
        else:
            log(f"[{nm}] columns:", list(d.columns))
    sch2 = curve_schema(stage2, "stage2") if not stage2.empty else None
    sch6 = curve_schema(stage6, "stage6") if not stage6.empty else None

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
    pad_x, pad_top, pad_bot = 0.016, 0.038, 0.014
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
        a.text(0.035, 0.035, f"$\\varphi$ = {rep[g]['porosity']:.3f}",
               transform=a.transAxes, fontsize=6.8, color=SUBTEXT,
               ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))

        v = rep[g]["vol"]
        slices = [v[v.shape[0] // 2, :, :], v[:, v.shape[1] // 2, :], v[:, :, v.shape[2] // 2]]
        for r, sl in enumerate(slices):
            a = fig.add_axes([x, row_y(r + 1), cell_w, cell_h])
            a.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            image_cell(a, COLORS[g])

    # =========================== PANEL B ====================================
    bx_, by_, bw_, bh_ = card_b
    b_pl, b_pr, b_pt, b_pb = 0.045, 0.018, 0.040, 0.046
    b_gx, b_gy = 0.058, 0.076

    plot_w = (bw_ - b_pl - b_pr - b_gx) / 2.0
    plot_h = (bh_ - b_pt - b_pb - b_gy) / 2.0

    specs = [
        ("tpcf_radial_periodic_pore", "Radial TPCF", "Lag / radius (vox)", r"$S_2(r)$", sch2, stage2),
        ("psd_radial_centered_pore", "Radial PSD", "Frequency bin", "Normalized power", sch2, stage2),
        ("chord_hist_pore_all_axes", "Pore chord lengths", "Chord length (vox)", "Probability", sch2, stage2),
        ("geom_intrusion_saturation_vs_radius_vox", "Geometric intrusion",
         "Radius (vox)", "Saturation proxy", sch6, stage6),
    ]

    for i, (curve, title, xlab, ylab, sch, df) in enumerate(specs):
        r, c = divmod(i, 2)
        x = bx_ + b_pl + c * (plot_w + b_gx)
        y = by_ + bh_ - b_pt - (r + 1) * plot_h - r * b_gy
        ax = fig.add_axes([x, y, plot_w, plot_h])
        if sch is None or df.empty:
            style_axis(ax)
            ax.set_title(title, pad=4.5, color=TEXT, fontweight="bold")
            ax.set_xlabel(xlab, color=SUBTEXT)
            ax.set_ylabel(ylab, color=SUBTEXT)
            ax.text(0.5, 0.5, "source table not available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color=SUBTEXT)
            continue
        plot_curve(ax, df, sch, curve, title, xlab, ylab, show_legend=(i == 1))

    # =========================== PANEL C ====================================
    # Full-width connectivity-sensitive distributions. Positions below are
    # tuned against card_c = [0.045, 0.012, 0.910, 0.290] and FIG_H = 10.2 in;
    # if either changes, re-check these constants for overlap. Generous
    # breathing room is deliberate at three spots that previously felt
    # cramped: card-top-to-title, the gap between the two plot blocks, and
    # last-xlabel-to-card-bottom.
    cx_, cy_, cw_, ch_ = card_c

    header_x = cx_ + 0.014
    title_y = cy_ + ch_ - 0.016
    subtitle_y = title_y - 0.0173

    p1_title_y = subtitle_y - 0.0155
    p1_ax_h = 0.0527
    p1_ax_y = p1_title_y - 0.0157 - p1_ax_h
    # (tick labels + xlabel render in the space below the axes, via
    # ax.set_xlabel with labelpad in plot_connectivity_distribution)

    p2_title_y = p1_ax_y - 0.0557
    p2_ax_h = 0.0527
    p2_ax_y = p2_title_y - 0.0157 - p2_ax_h
    # (p2_ax_y - 0.0437 lands close to the card bottom cy_, leaving a clear
    # margin below the last xlabel + tick labels)

    plot_x0 = cx_ + 0.075
    plot_w2 = (cx_ + cw_ - 0.020) - plot_x0
    plot_xc = plot_x0 + plot_w2 / 2.0

    fig.text(header_x, title_y, "Connectivity-sensitive distributions",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)
    legend_handles = [Patch(facecolor=COLORS[g], edgecolor=COLORS[g], alpha=0.85, label=LABELS[g])
                      for g in CONNECTIVITY_ORDER]
    fig.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(cx_ + cw_ - 0.010, title_y + 0.006),
              bbox_transform=fig.transFigure, ncol=3, frameon=True,
              facecolor="white", edgecolor=SPINE, framealpha=0.94,
              borderpad=0.45, labelspacing=0.3, columnspacing=1.1,
              handlelength=1.4, handletextpad=0.5, fontsize=7.4).get_frame().set_linewidth(0.6)

    fig.text(plot_xc, p1_title_y, "Largest connected pore-component fraction",
             ha="center", va="top", fontsize=9.0, fontweight="bold", color=TEXT)
    fig.text(plot_xc, p2_title_y, "Disconnected pore fraction",
             ha="center", va="top", fontsize=9.0, fontweight="bold", color=TEXT)

    ax_top = fig.add_axes([plot_x0, p1_ax_y, plot_w2, p1_ax_h])
    ax_bot = fig.add_axes([plot_x0, p2_ax_y, plot_w2, p2_ax_h])

    rng = np.random.default_rng(11)
    plot_connectivity_distribution(
        ax_top, stage1, "largest_component_fraction_pore",
        "Fraction of pore voxels in the largest connected component", rng)
    plot_connectivity_distribution(
        ax_bot, stage1, "disconnected_fraction_pore",
        "Fraction of pore voxels outside the largest connected component", rng)

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

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out):
        log(" ", p.resolve())


if __name__ == "__main__":
    sys.exit(main())
