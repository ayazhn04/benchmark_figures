#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.7 composite (magma, final)
Reference vs PoreDiT vs SurVol -- IBM 18A Liver Rock sandstone, large-volume
(512^3) generative benchmark.

Same visual contract as Figures 4.1/4.4 (make_fig4_1_composite_magma_final.py,
make_fig4_4_composite_magma_final.py): same cards, fonts, panel-label style,
export block. No tight_layout / constrained_layout / bbox_inches.

RUNTIME NOTE: this script is written and reviewed without access to the
Section 4.7 runtime filesystem. Every data path below is the exact,
authoritative path supplied for the Section 4.7 transfer package -- the
script never searches /home/ra2 broadly, never substitutes an alternative
path, and never invents a fallback location. Panel-a source-volume failures
raise loudly (no MIP/fallback render). Quantitative-panel (b/c/d/e) CSV
failures log a WARNING and draw a restrained "metric unavailable" note in
that subplot instead of stopping the whole run or silently substituting a
different metric/curve/model.

Physical phase convention in the transferred NPY files: 0 = pore, 1 = solid.
This script never rewrites that convention and never writes back into the
transfer package. For visualization only, pore_mask = (volume == 0) is used
so the pore phase is the rendered/displayed foreground.

Run on ra2 (after the script is manually copied there):

    cd /home/ra2/benchmark_figures
    conda activate poregen_nmc128

    mkdir -p /home/ra2/section_4_7_transfer/paper_figures_4_7/figure_4_7_composite_magma_final

    python -u scripts/make_fig4_7_composite_magma_final.py \
      2>&1 | tee /home/ra2/section_4_7_transfer/paper_figures_4_7/figure_4_7_composite_magma_final/fig4_7_composite_magma_final_run.log
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG -- authoritative Section 4.7 transfer package paths
# ============================================================================

# Exact, authoritative package root. Never searched broadly, never
# substituted. Every scientific-data path below resolves under this root.
PACKAGE_ROOT = Path("/home/ra2/section_4_7_transfer/section_4_7_transfer_package")

DO_NOT_USE_TXT = PACKAGE_ROOT / "DO_NOT_USE_FOR_FIGURE_4_7.txt"
MANIFEST_CSV = PACKAGE_ROOT / "MANIFEST_4_7.csv"
SHA256SUMS_TXT = PACKAGE_ROOT / "SHA256SUMS.txt"
QC_MANIFEST_CSV = PACKAGE_ROOT / "provenance" / "visual_qc_manifest.csv"

GROUPS = ["real", "poredit", "survol"]

# Transferred visual sample folders (four approved Stage-13 candidates per
# cohort, ordinal indices 0/16/32/49 -- NOT the full 50-sample evaluation
# cohort). Only the 512 folders are actually used by this figure (it focuses
# on the 512^3 case); the 256 folders are listed for completeness/provenance
# but are never opened by any panel here.
SAMPLE_DIRS = {
    ("real", 256): PACKAGE_ROOT / "samples/real/256",
    ("real", 512): PACKAGE_ROOT / "samples/real/512",
    ("poredit", 256): PACKAGE_ROOT / "samples/poredit/256",
    ("poredit", 512): PACKAGE_ROOT / "samples/poredit/512",
    ("survol", 256): PACKAGE_ROOT / "samples/survol/256",
    ("survol", 512): PACKAGE_ROOT / "samples/survol/512",
}
SAMPLE_DIRS_512 = {g: SAMPLE_DIRS[(g, 512)] for g in GROUPS}
EXPECTED_VIZ_SAMPLES_PER_FOLDER = 4
APPROVED_ORDINALS = (0, 16, 32, 49)

# Deterministic, pre-approved panel-a representative sample (ordinal 32).
# Never chosen by visually inspecting the four candidates, never changed
# automatically.
REP_ORDINAL = 32
REP_FILES = {
    "real": PACKAGE_ROOT / "samples/real/512/real_512cube_032.npy",
    "poredit": PACKAGE_ROOT / "samples/poredit/512/poredit_512cube_032_seed57532.npy",
    "survol": PACKAGE_ROOT / "samples/survol/512/survol_512_seed_57532.npy",
}

EXPECTED_SHAPE_512 = (512, 512, 512)
CENTRAL_INDEX = 256
VALID_LABELS = {0, 1}

# ---- panel b: long-range structure curves (official CSVs only) -----------
S2_CSV = PACKAGE_ROOT / "metrics/04_curves/spatial_statistics/stage02_group_mean_curves.csv"
C2_CSV = PACKAGE_ROOT / "metrics/04_curves/c2_pair_connectedness/stage03B_c2_group_mean_curves.csv"

# ---- panel c: scale-dependent heterogeneity / REV convergence ------------
REV_CSV = PACKAGE_ROOT / "metrics/05_rev_scale/rev_scale_convergence_curves.csv"
REV_BLOCK_SIZES = (32, 64, 128, 256)

# ---- panel d: physical trade-off group-summary CSVs -----------------------
COORD_CSV = PACKAGE_ROOT / "metrics/08_network/stage07_snow_graph/stage07_group_summary.csv"
TRANSPORT_CSV = PACKAGE_ROOT / "metrics/09_transport/stage08_network_transport/stage08_group_summary.csv"
CAPILLARY_CSV = PACKAGE_ROOT / "metrics/10_large_volume_integrity/capillary_accessibility/stage09_group_summary.csv"
BETTI_CSV = PACKAGE_ROOT / "metrics/07_topology/stage10_betti_persistent_homology/stage10_group_summary.csv"

# (metric_key, row label, source CSV, log label) -- exact order required.
# Row labels are display text only -- the metric_key (first element) used
# for CSV lookup is unchanged.
PANEL_D_ROWS = [
    ("mean_coordination", "Coordination", COORD_CSV, "stage07-coordination"),
    ("mean_permeability_vox2", "PNM permeability", TRANSPORT_CSV, "stage08-permeability"),
    ("mean_tortuosity_factor", "PNM tortuosity", TRANSPORT_CSV, "stage08-tortuosity"),
    ("critical_radius_weighted_p50_vox", "Critical radius $r_{50}$", CAPILLARY_CSV, "stage09-r50"),
    ("betti1_fullres", "$\\beta_1$", BETTI_CSV, "stage10-betti1"),
]

# ---- panel e: ensemble representativeness ---------------------------------
DIVERSITY_CSV = PACKAGE_ROOT / "metrics/11_diversity/diversity_descriptor_comparison_full300_core.csv"
CORRELATION_CSV = PACKAGE_ROOT / "metrics/11_diversity/final_refresh/stage11_cross_property_correlation_discrepancy.csv"

COHORT_TO_GROUP = {"real_512": "real", "poredit_512": "poredit", "survol_512": "survol"}

# Defense-in-depth only: every path above is hardcoded exactly per task
# instructions, so this should never trigger, but it guards against an
# accidental future edit resolving into one of the explicitly forbidden
# cohorts/folders.
EXCLUDE_PATH_TOKENS = ("poregen", "section47finalbenchmark", "roi2", "checkpointscreen",
                       "smoketest", "smoke", "pilot", "debug", "tmpcache", "cache",
                       "aborted", "failedrun", "survol1c", "checkpointpreview",
                       "archived", "backup", "ablation")

RENDER_PX = 1500
CANVAS_PX = 900
DOWNSAMPLE = 4          # fixed, identical factor for all three groups' 512^3 render

FIG_W, FIG_H = 17.8, 10.2

OUT = Path("/home/ra2/section_4_7_transfer/paper_figures_4_7/figure_4_7_composite_magma_final")
TMP = OUT / "_render_cache"
STEM = "fig4_7_composite_magma_final"
AUDIT_JSON = OUT / "figure_4_7_audit.json"

# ============================================================================
# 1b. SANITY-CHECK CONSTANTS -- approximate manuscript values, used ONLY to
# log a soft warning if loaded data are grossly inconsistent. NEVER plotted,
# never used as a substitute for the loaded CSV values.
# ============================================================================

SANITY_D = {
    "mean_coordination": {"real": 3.067, "poredit": 3.285, "survol": 3.143},
    "mean_permeability_vox2": {"real": 0.06956, "poredit": 0.05570, "survol": 0.04198},
    "mean_tortuosity_factor": {"real": 6.933, "poredit": 9.824, "survol": 7.781},
    "critical_radius_weighted_p50_vox": {"real": 6.845, "poredit": 6.774, "survol": 6.392},
    "betti1_fullres": {"real": 1270.2, "poredit": 559.1, "survol": 1424.6},
}
SANITY_D_REL_TOL = 0.5  # generous: sanity check only, never a hard stop

SANITY_E_MEDIAN = {"poredit": 0.945, "survol": 0.309}
SANITY_E_CORR = {"poredit": 0.2685, "survol": 0.3817}

# ============================================================================
# 2. STYLE -- identical contract to Figures 4.1/4.4
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

COLORS = {"real": "#8C93A1", "poredit": "#F2A93B", "survol": "#772A8E"}
LABELS = {"real": "Reference", "poredit": "PoreDiT", "survol": "SurVol"}
LINESTYLES = {"real": "-", "poredit": "-", "survol": (0, (5.5, 2.2))}
LINEWIDTHS = {"real": 1.85, "poredit": 1.85, "survol": 1.95}
MARKERS = {"real": "o", "poredit": "D", "survol": "^"}
TITLE_COLOR = {"real": TEXT, "poredit": COLORS["poredit"], "survol": COLORS["survol"]}

CARD_LW = 0.8
CELL_LW = 1.6

# Small, quiet direction cues -- shown only in the leftmost (Reference)
# column, same convention as Figures 4.1/4.4.
ROW_NAMES = ["3D pore volume", "X–Y slice", "X–Z slice", "Y–Z slice"]
DIRECTION_LABELS = {
    "X–Y slice": [("→ X", (0.90, 0.07), "right", "bottom"), ("↑ Y", (0.07, 0.90), "left", "top")],
    "X–Z slice": [("→ X", (0.90, 0.07), "right", "bottom"), ("↑ Z", (0.07, 0.90), "left", "top")],
    "Y–Z slice": [("→ Y", (0.90, 0.07), "right", "bottom"), ("↑ Z", (0.07, 0.90), "left", "top")],
}

# ============================================================================
# 3. GEOMETRY (figure fractions) -- five-card layout, per task instructions
# ============================================================================

card_a = [0.045, 0.365, 0.380, 0.570]
card_b = [0.445, 0.655, 0.510, 0.280]
card_c = [0.445, 0.365, 0.510, 0.240]
card_d = [0.045, 0.012, 0.445, 0.290]
card_e = [0.510, 0.012, 0.445, 0.290]

PANEL_LABEL_OFFSET = 0.030

# ============================================================================
# 4. GENERIC HELPERS
# ============================================================================

WARNINGS: list = []


def log(*a):
    print(*a, flush=True)


def warn(msg: str):
    WARNINGS.append(msg)
    log(f"[WARNING] {msg}")


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _path_is_excluded(p: Path) -> bool:
    s = _norm(str(p))
    return any(tok in s for tok in EXCLUDE_PATH_TOKENS)


def pick_col(df: pd.DataFrame, candidates, what: str, required: bool = True):
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
    return None


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


def draw_unavailable(ax, text="metric unavailable"):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.0, color=SUBTEXT)


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


# ============================================================================
# 5. PACKAGE / SAMPLE VALIDATION (fails loudly -- panel a has no fallback)
# ============================================================================


def validate_package_root():
    if not PACKAGE_ROOT.exists():
        raise FileNotFoundError(
            f"[package] PACKAGE_ROOT does not exist: {PACKAGE_ROOT}\n"
            f"This script only reads from the exact authoritative Section 4.7 transfer "
            f"package -- it never searches for or invents a substitute location."
        )
    log(f"[package] PACKAGE_ROOT OK: {PACKAGE_ROOT}")

    if DO_NOT_USE_TXT.exists():
        log(f"[package] exclusion-policy marker found: {DO_NOT_USE_TXT.name} -- noted; this "
            f"script still only ever uses the exact paths specified for Figure 4.7")
    else:
        log(f"[package] exclusion-policy marker not found (informational only): {DO_NOT_USE_TXT}")

    for p in (MANIFEST_CSV, SHA256SUMS_TXT):
        log(f"[package] {'found' if p.exists() else 'NOT FOUND (informational only)'}: {p}")


def validate_visualization_folder(group: str, path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"[viz-folder:{group}] expected transferred folder not found: {path}")
    if _path_is_excluded(path):
        raise RuntimeError(f"[viz-folder:{group}] resolved folder {path} matches a forbidden "
                            f"pattern ({EXCLUDE_PATH_TOKENS}) -- refusing to use it")
    files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".npy")
    if len(files) != EXPECTED_VIZ_SAMPLES_PER_FOLDER:
        raise RuntimeError(f"[viz-folder:{group}] {path}: expected exactly "
                            f"{EXPECTED_VIZ_SAMPLES_PER_FOLDER} .npy files, found {len(files)}: "
                            f"{[f.name for f in files]}")
    log(f"[viz-folder:{group}] {path}: {len(files)} NPY files OK "
        f"({[f.name for f in files]})")
    return files


def load_and_validate_representative(group: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"[rep:{group}] expected representative file not found: {path}")
    if _path_is_excluded(path):
        raise RuntimeError(f"[rep:{group}] resolved file {path} matches a forbidden pattern "
                            f"({EXCLUDE_PATH_TOKENS}) -- refusing to use it")

    arr = np.load(str(path))
    arr = np.squeeze(arr)
    if arr.shape != EXPECTED_SHAPE_512:
        raise RuntimeError(f"[rep:{group}] {path.name}: shape {arr.shape} != expected "
                            f"{EXPECTED_SHAPE_512}. Stopping WITHOUT saving the figure.")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"[rep:{group}] {path.name}: contains non-finite values")
    if arr.dtype != np.uint8:
        log(f"[rep:{group}] note: on-disk dtype is {arr.dtype} (expected uint8) -- proceeding "
            f"only because the exact {{0,1}} label check below still passes")
    vals = set(np.unique(arr).tolist())
    if vals != VALID_LABELS:
        raise RuntimeError(f"[rep:{group}] {path.name}: unique labels {sorted(vals)} -- expected "
                            f"exactly {sorted(VALID_LABELS)} (0=pore, 1=solid)")
    arr = arr.astype(np.uint8)
    phi = float(np.mean(arr == 0))
    log(f"[rep:{group}] {path.name}  shape={arr.shape}  on-disk dtype checked  labels={{0,1}}  "
        f"phi(pore)={phi:.4f}")
    return arr, phi


def validate_qc_manifest(rep_files: dict):
    if not QC_MANIFEST_CSV.exists():
        raise FileNotFoundError(f"[qc-manifest] missing official file: {QC_MANIFEST_CSV}")
    df = pd.read_csv(QC_MANIFEST_CSV)

    filename_col = pick_col(df, ["filename", "file", "file_name", "sample_filename"], "qc.filename")
    size_col = pick_col(df, ["size", "volume_size", "cube_size"], "qc.size")
    ordinal_col = pick_col(df, ["ordinal_index", "ordinal", "stage13_ordinal_index"], "qc.ordinal_index")
    central_col = pick_col(df, ["central_index", "central_slice_index", "center_index"], "qc.central_index")

    for group, path in rep_files.items():
        row = df[df[filename_col].astype(str) == path.name]
        if row.empty:
            raise RuntimeError(f"[qc-manifest] filename '{path.name}' (group '{group}') not found "
                                f"in {QC_MANIFEST_CSV}")
        if len(row) > 1:
            raise RuntimeError(f"[qc-manifest] filename '{path.name}' (group '{group}') matched "
                                f"{len(row)} rows in {QC_MANIFEST_CSV} -- ambiguous")
        row = row.iloc[0]
        size = int(pd.to_numeric(row[size_col], errors="coerce"))
        ordinal = int(pd.to_numeric(row[ordinal_col], errors="coerce"))
        central = int(pd.to_numeric(row[central_col], errors="coerce"))
        if size != 512 or ordinal != REP_ORDINAL or central != CENTRAL_INDEX:
            raise RuntimeError(
                f"[qc-manifest] '{path.name}' (group '{group}'): size={size} ordinal_index={ordinal} "
                f"central_index={central} -- expected size=512 ordinal_index={REP_ORDINAL} "
                f"central_index={CENTRAL_INDEX}")
        log(f"[qc-manifest] '{path.name}' (group '{group}') OK: size=512 "
            f"ordinal_index={REP_ORDINAL} central_index={CENTRAL_INDEX}")


# ============================================================================
# 6. PANEL A -- 3D pore-phase isosurface + XY/XZ/YZ slices (512^3)
#
# True marching-cubes isosurface of the PORE phase (pore_mask = volume==0)
# from the representative volume, downsampled by a single fixed, deterministic
# block-occupancy-averaging factor (DOWNSAMPLE) identical across all three
# groups purely for render tractability -- no cosmetic smoothing, no
# content-dependent cropping, no MIP fallback. If PyVista/scikit-image cannot
# run, this raises loudly.
# ============================================================================


def render_isosurface_pore(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float):
    import pyvista as pv
    from skimage import measure

    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    pore_mask = (vol == 0).astype(np.float32)
    step = DOWNSAMPLE
    if any(n % step != 0 for n in pore_mask.shape):
        raise RuntimeError(f"[render:{group}] volume shape {pore_mask.shape} is not evenly "
                            f"divisible by DOWNSAMPLE={step} -- cannot block-average")

    nz, ny, nx = pore_mask.shape
    # Fixed, deterministic block-occupancy averaging (no Gaussian/cosmetic
    # smoothing): each output cell is the exact mean pore occupancy of its
    # step^3 input block.
    v = pore_mask.reshape(
        nz // step, step,
        ny // step, step,
        nx // step, step,
    ).mean(axis=(1, 3, 5))
    if v.min() >= 0.5 or v.max() <= 0.5:
        raise RuntimeError(f"[render:{group}] degenerate iso-level for the pore phase after "
                            f"block-occupancy downsampling -- cannot extract a level=0.5 isosurface")

    # spacing=(step, step, step) restores vertex coordinates directly to
    # original voxel units, so no separate verts * step rescale is needed.
    verts, faces, _, _ = measure.marching_cubes(v, level=0.5, spacing=(step, step, step))
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
    # size, no cropping; same convention as Figures 4.1/4.4.
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
        render_isosurface_pore(vol, group, out_raw, parallel_scale)
    except Exception as exc:
        raise RuntimeError(
            f"[render:{group}] true 3D pore-phase isosurface rendering failed and no fallback "
            f"(e.g. a MIP projection) is permitted for this figure -- PyVista + scikit-image "
            f"marching_cubes are required: {exc}"
        ) from exc


def finalize_renders(raw_paths: dict, out_paths: dict, pad_frac: float = 0.06):
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
        a.text(0.035, 0.035, f"$\\varphi$ = {rep[g]['phi']:.3f}",
               transform=a.transAxes, fontsize=6.8, color=SUBTEXT,
               ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))

        v = rep[g]["vol"]
        pore_mask = (v == 0).astype(np.uint8)
        i = CENTRAL_INDEX
        # (z, y, x) convention: XY = v[z=i,:,:], XZ = v[:,y=i,:], YZ = v[:,:,x=i].
        slices = [pore_mask[i, :, :], pore_mask[:, i, :], pore_mask[:, :, i]]
        slice_row_names = ["X–Y slice", "X–Z slice", "Y–Z slice"]
        for ri, sl in enumerate(slices):
            a = fig.add_axes([x, row_y(ri + 1), cell_w, cell_h])
            # pore=1 -> bright/white foreground, solid=0 -> dark background.
            a.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            image_cell(a, COLORS[g])
            if c == 0:
                for txt, (tx, ty), ha, va in DIRECTION_LABELS[slice_row_names[ri]]:
                    a.text(tx, ty, txt, transform=a.transAxes, fontsize=6.9, fontweight="bold",
                           color="white", ha=ha, va=va,
                           bbox=dict(facecolor=TEXT, edgecolor="none", alpha=0.55, pad=1.2))

    # small neutral pore/solid legend, centered under the grid
    legend_y = ay_ + 0.015
    item_widths = [0.072, 0.078]
    total_w = sum(item_widths)
    lx = gx0 + label_w + gap_x + (grid_w - label_w - gap_x - total_w) / 2.0
    for name, color, w in zip(["Pore", "Solid"], ["#FFFFFF", "#000000"], item_widths):
        fig.add_artist(Rectangle((lx, legend_y - 0.006), 0.013, 0.011,
                                  transform=fig.transFigure, facecolor=color,
                                  edgecolor=SPINE, linewidth=0.6, zorder=101))
        fig.text(lx + 0.019, legend_y, name, ha="left", va="center",
                 fontsize=7.4, color=SUBTEXT, zorder=101)
        lx += w


# ============================================================================
# 7. PANEL B -- long-range structure at 512^3 (S2 left, pore C2 right)
# Official CSVs only. No recomputation from the transferred NPY volumes.
# ============================================================================


def _read_csv_checked(path: Path, required_cols: set, label: str):
    if not path.exists():
        warn(f"[{label}] MISSING official file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warn(f"[{label}] failed to read {path}: {exc}")
        return None
    missing = required_cols - set(df.columns)
    if missing:
        warn(f"[{label}] {path} missing required columns: {sorted(missing)}; "
             f"available: {list(df.columns)}")
        return None
    log(f"[{label}] loaded {path.name}: {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_curve(csv_path: Path, curve_key: str, x_col: str, label: str):
    df = _read_csv_checked(csv_path, {"cohort", "volume_size", "curve", x_col, "mean", "std"}, label)
    if df is None:
        return None

    vsize = pd.to_numeric(df["volume_size"], errors="coerce")
    sub = df[(vsize == 512) & (df["curve"].astype(str) == curve_key)].copy()
    if sub.empty:
        warn(f"[{label}] no rows matched volume_size==512 & curve=='{curve_key}' in {csv_path}")
        return None

    out = {}
    for cohort, g in COHORT_TO_GROUP.items():
        s = sub[sub["cohort"].astype(str) == cohort].copy()
        if s.empty:
            warn(f"[{label}] cohort '{cohort}' missing for curve '{curve_key}' in {csv_path}")
            return None
        s[x_col] = pd.to_numeric(s[x_col], errors="coerce")
        s = s.sort_values(x_col)
        x = s[x_col].to_numpy(float)
        y = pd.to_numeric(s["mean"], errors="coerce").to_numpy(float)
        sd = pd.to_numeric(s["std"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(x) & np.isfinite(y)):
            warn(f"[{label}] cohort '{cohort}': non-finite x/y values for curve '{curve_key}'")
            return None
        out[g] = {"x": x, "y": y, "std": np.nan_to_num(sd)}

    log(f"[{label}] resolved curve='{curve_key}' (volume_size=512): {len(sub)} rows, "
        f"groups={list(out.keys())}")
    return out


def plot_curve_pair(ax, curves, title, xlabel, ylabel, show_legend=False):
    style_axis(ax)
    ax.set_title(title, pad=4.5, color=TEXT, fontweight="bold")
    ax.set_xlabel(xlabel, color=SUBTEXT)
    ax.set_ylabel(ylabel, color=SUBTEXT)

    if curves is None:
        draw_unavailable(ax)
        return

    for g in GROUPS:
        d = curves[g]
        x, y, sd = d["x"], d["y"], d.get("std")
        if g == "real" and sd is not None and np.any(sd > 0):
            ax.fill_between(x, y - sd, y + sd, color=COLORS[g], alpha=0.20, linewidth=0, zorder=1)
        ax.plot(x, y, color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                solid_capstyle="round", label=LABELS[g], zorder=3 if g == "real" else 4)

    ax.margins(x=0.02, y=0.06)

    if show_legend:
        leg = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=SPINE,
                        framealpha=0.94, handlelength=2.3, borderpad=0.45, labelspacing=0.34)
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(8)


def build_panel_b(fig, card, s2_curves, c2_curves):
    bx_, by_, bw_, bh_ = card
    header_y = by_ + bh_ - 0.020
    fig.text(bx_ + 0.014, header_y, "Long-range structure",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    b_pl, b_pr, b_pt, b_pb, b_gx = 0.048, 0.020, 0.058, 0.052, 0.062
    plot_w = (bw_ - b_pl - b_pr - b_gx) / 2.0
    plot_h = bh_ - b_pt - b_pb
    y = by_ + b_pb
    x_left = bx_ + b_pl
    x_right = x_left + plot_w + b_gx

    ax_left = fig.add_axes([x_left, y, plot_w, plot_h])
    ax_right = fig.add_axes([x_right, y, plot_w, plot_h])

    plot_curve_pair(ax_left, s2_curves, "Radial $S_2(r)$", "Lag / radius (vox)", "$S_2(r)$",
                     show_legend=False)
    plot_curve_pair(ax_right, c2_curves, "Radial pore $C_2(r)$", "Lag / radius (vox)", "$C_2(r)$",
                     show_legend=True)


# ============================================================================
# 8. PANEL C -- scale-dependent heterogeneity / REV convergence at 512^3
# ============================================================================


def load_rev_scale():
    df = _read_csv_checked(
        REV_CSV,
        {"cohort", "block_size", "metric", "mean_within_volume_cv", "median_within_volume_cv",
         "p90_within_volume_cv", "mean_within_volume_std"},
        "rev-scale")
    if df is None:
        return None

    sub = df[df["metric"].astype(str) == "porosity"].copy()
    if sub.empty:
        warn(f"[rev-scale] no rows with metric=='porosity' in {REV_CSV}")
        return None

    out = {}
    for cohort, g in COHORT_TO_GROUP.items():
        s = sub[sub["cohort"].astype(str) == cohort].copy()
        s["_block"] = pd.to_numeric(s["block_size"], errors="coerce")
        s = s[s["_block"].isin(REV_BLOCK_SIZES)].sort_values("_block")
        blocks_present = sorted(int(b) for b in s["_block"].unique().tolist())
        if blocks_present != list(REV_BLOCK_SIZES):
            warn(f"[rev-scale] cohort '{cohort}': block sizes {blocks_present} != required "
                 f"{list(REV_BLOCK_SIZES)}")
            return None
        y = pd.to_numeric(s["mean_within_volume_cv"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(y)):
            warn(f"[rev-scale] cohort '{cohort}': non-finite mean_within_volume_cv values")
            return None
        out[g] = {"x": np.array(REV_BLOCK_SIZES, dtype=float), "y": y}

    log(f"[rev-scale] resolved metric='porosity' block_sizes={list(REV_BLOCK_SIZES)} "
        f"groups={list(out.keys())}")
    return out


def build_panel_c(fig, card, rev_data):
    cx_, cy_, cw_, ch_ = card
    header_y = cy_ + ch_ - 0.020
    fig.text(cx_ + 0.014, header_y, "Scale-dependent heterogeneity",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    legend_handles = [Patch(facecolor=COLORS[g], edgecolor=COLORS[g], alpha=0.85, label=LABELS[g])
                      for g in GROUPS]
    fig.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(cx_ + cw_ - 0.010, header_y + 0.006),
              bbox_transform=fig.transFigure, ncol=3, frameon=True,
              facecolor="white", edgecolor=SPINE, framealpha=0.94,
              borderpad=0.45, labelspacing=0.3, columnspacing=1.1,
              handlelength=1.4, handletextpad=0.5, fontsize=7.4).get_frame().set_linewidth(0.6)

    pad_l, pad_r, pad_t, pad_b = 0.058, 0.025, 0.078, 0.085
    ax = fig.add_axes([cx_ + pad_l, cy_ + pad_b, cw_ - pad_l - pad_r, ch_ - pad_t - pad_b])
    style_axis(ax)
    ax.set_xlabel("Subvolume size (voxels)", color=SUBTEXT)
    ax.set_ylabel("Local porosity CV", color=SUBTEXT)

    if rev_data is None:
        draw_unavailable(ax)
        return

    ax.set_xscale("log", base=2)
    for g in GROUPS:
        d = rev_data[g]
        ax.plot(d["x"], d["y"], color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                marker=MARKERS[g], markersize=6.2, markerfacecolor=COLORS[g],
                markeredgecolor="black", markeredgewidth=0.7, label=LABELS[g],
                zorder=3 if g == "real" else 4)
    ax.set_xticks(list(REV_BLOCK_SIZES))
    ax.set_xticklabels([f"${b}^3$" for b in REV_BLOCK_SIZES], fontsize=7.4)
    ax.minorticks_off()
    ax.margins(y=0.14)


# ============================================================================
# 9. PANEL D -- physical trade-off at 512^3 (generated / reference ratio)
# ============================================================================

_CSV_CACHE: dict = {}


def _read_csv_cached(path: Path, label: str):
    if path in _CSV_CACHE:
        return _CSV_CACHE[path]
    df = _read_csv_checked(path, {"cohort", "metric", "mean"}, label)
    _CSV_CACHE[path] = df
    return df


def load_group_summary_mean(path: Path, cohort: str, metric: str, label: str):
    df = _read_csv_cached(path, label)
    if df is None:
        return None
    row = df[(df["cohort"].astype(str) == cohort) & (df["metric"].astype(str) == metric)]
    if len(row) != 1:
        warn(f"[{label}] expected exactly 1 row for cohort='{cohort}' metric='{metric}' in "
             f"{path}, got {len(row)}")
        return None
    val = pd.to_numeric(row.iloc[0]["mean"], errors="coerce")
    if not np.isfinite(val):
        warn(f"[{label}] cohort='{cohort}' metric='{metric}': mean is not finite")
        return None
    return float(val)


def resolve_panel_d():
    rows = []
    for metric_key, label_text, path, log_label in PANEL_D_ROWS:
        real_v = load_group_summary_mean(path, "real_512", metric_key, log_label)
        poredit_v = load_group_summary_mean(path, "poredit_512", metric_key, log_label)
        survol_v = load_group_summary_mean(path, "survol_512", metric_key, log_label)

        ratios = {}
        if real_v is not None and real_v != 0:
            if poredit_v is not None:
                ratios["poredit"] = poredit_v / real_v
            if survol_v is not None:
                ratios["survol"] = survol_v / real_v
        elif real_v is not None and real_v == 0:
            warn(f"[panel-d:{log_label}] reference mean is exactly 0 for metric "
                 f"'{metric_key}' -- ratio undefined, skipping this row's markers")
        else:
            warn(f"[panel-d:{log_label}] reference mean unavailable for metric '{metric_key}' "
                 f"-- skipping this row's markers")

        sanity = SANITY_D.get(metric_key)
        if sanity is not None and real_v is not None and sanity["real"] != 0:
            rel = abs(real_v - sanity["real"]) / abs(sanity["real"])
            if rel > SANITY_D_REL_TOL:
                warn(f"[sanity:panel-d:{log_label}] metric '{metric_key}' real mean {real_v:.6g} "
                     f"far from sanity reference ~{sanity['real']:.6g} (rel diff {rel:.2f})")

        log(f"[panel-d:{log_label}] metric='{metric_key}'  real={real_v}  poredit={poredit_v}  "
            f"survol={survol_v}  ratios={ratios}")
        rows.append({"metric": metric_key, "label": label_text, "real": real_v,
                     "poredit": poredit_v, "survol": survol_v, "ratios": ratios})
    return rows


def build_panel_d(fig, card, rows):
    dx_, dy_, dw_, dh_ = card
    header_y = dy_ + dh_ - 0.020
    fig.text(dx_ + 0.014, header_y, "Physical-property fidelity",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    legend_handles = [
        Line2D([0], [0], marker=MARKERS["poredit"], linestyle="None", markersize=6.5,
               markerfacecolor=COLORS["poredit"], markeredgecolor="black", markeredgewidth=0.8,
               label=LABELS["poredit"]),
        Line2D([0], [0], marker=MARKERS["survol"], linestyle="None", markersize=6.5,
               markerfacecolor=COLORS["survol"], markeredgecolor="black", markeredgewidth=0.8,
               label=LABELS["survol"]),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(dx_ + dw_ - 0.010, header_y + 0.006),
              bbox_transform=fig.transFigure, ncol=2, frameon=True,
              facecolor="white", edgecolor=SPINE, framealpha=0.94,
              borderpad=0.45, labelspacing=0.3, columnspacing=1.0,
              handlelength=1.2, handletextpad=0.5, fontsize=7.2).get_frame().set_linewidth(0.6)

    pad_l, pad_r, pad_t, pad_b = 0.118, 0.030, 0.075, 0.062
    ax = fig.add_axes([dx_ + pad_l, dy_ + pad_b, dw_ - pad_l - pad_r, dh_ - pad_t - pad_b])
    style_axis(ax, grid=False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("Generated / reference mean  (1 = reference)", color=SUBTEXT, fontsize=7.6)
    ax.tick_params(axis="x", labelsize=7.2)

    n = len(rows)
    ys = np.arange(n)[::-1]  # first row on top
    any_plotted = False
    all_ratios = []
    for y, entry in zip(ys, rows):
        # A thin connector from the x=1 reference baseline out to each
        # generated model's ratio makes deviation-from-reference the visual
        # point, without needing a separate marker glyph at x=1 itself.
        for g, dy_off, connector_alpha in (("poredit", 0.13, 0.40), ("survol", -0.13, 0.40)):
            r = entry["ratios"].get(g)
            if r is None:
                continue
            ax.plot([1.0, r], [y + dy_off, y + dy_off], color=COLORS[g],
                    linewidth=0.9, alpha=connector_alpha, zorder=3, solid_capstyle="round")
            ax.scatter([r], [y + dy_off], s=90, marker=MARKERS[g], color=COLORS[g],
                       edgecolors="black", linewidths=0.9, zorder=5)
            any_plotted = True
            all_ratios.append(r)

    # The x=1 line is now the sole reference cue (no separate marker glyph),
    # so it is drawn a bit bolder/more visible than the earlier faint version.
    ax.axvline(1.0, color=SUBTEXT, linewidth=1.3, linestyle=(0, (3, 2)), alpha=0.8, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.0)
    ax.set_ylim(-0.6, n - 0.4)

    if any_plotted:
        lo = min([1.0] + all_ratios)
        hi = max([1.0] + all_ratios)
        pad = max(0.08, 0.12 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)
    else:
        draw_unavailable(ax)


# ============================================================================
# 10. PANEL E -- ensemble representativeness at 512^3
# ============================================================================


def load_diversity_ratios():
    df = _read_csv_checked(DIVERSITY_CSV, {"model", "metric", "std_ratio_gen_real"}, "diversity-core")
    if df is None:
        return None

    out = {}
    for cohort, g in (("poredit_512", "poredit"), ("survol_512", "survol")):
        sub = df[df["model"].astype(str) == cohort].copy()
        vals = pd.to_numeric(sub["std_ratio_gen_real"], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            warn(f"[diversity-core] model '{cohort}': no finite, positive std_ratio_gen_real "
                 f"values in {DIVERSITY_CSV}")
            return None
        median = float(np.median(vals))
        out[g] = vals
        sanity = SANITY_E_MEDIAN.get(g)
        if sanity is not None and abs(median - sanity) / max(abs(sanity), 1e-9) > SANITY_D_REL_TOL:
            warn(f"[sanity:diversity-core] model '{cohort}': median std_ratio_gen_real "
                 f"{median:.4f} far from sanity reference ~{sanity}")
        log(f"[diversity-core] model '{cohort}': n={vals.size} median={median:.4f}")

    return out


def load_correlation_discrepancy():
    df = _read_csv_checked(CORRELATION_CSV, {"model", "mean_abs_correlation_difference"},
                            "cross-property-correlation")
    if df is None:
        return None

    out = {}
    for cohort, g in (("poredit_512", "poredit"), ("survol_512", "survol")):
        row = df[df["model"].astype(str) == cohort]
        if len(row) != 1:
            warn(f"[cross-property-correlation] expected exactly 1 row for model='{cohort}' in "
                 f"{CORRELATION_CSV}, got {len(row)}")
            return None
        val = pd.to_numeric(row.iloc[0]["mean_abs_correlation_difference"], errors="coerce")
        if not np.isfinite(val):
            warn(f"[cross-property-correlation] model='{cohort}': value not finite")
            return None
        out[g] = float(val)
        sanity = SANITY_E_CORR.get(g)
        if sanity is not None and abs(out[g] - sanity) / max(abs(sanity), 1e-9) > SANITY_D_REL_TOL:
            warn(f"[sanity:cross-property-correlation] model='{cohort}': value {out[g]:.4f} far "
                 f"from sanity reference ~{sanity}")

    log(f"[cross-property-correlation] resolved values: {out}")
    return out


def build_panel_e(fig, card, diversity, correlation):
    ex_, ey_, ew_, eh_ = card
    header_y = ey_ + eh_ - 0.020
    fig.text(ex_ + 0.014, header_y, "Ensemble representativeness",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    e_pl, e_pr, e_pt, e_pb, e_gx = 0.058, 0.025, 0.078, 0.078, 0.075
    plot_w = (ew_ - e_pl - e_pr - e_gx) / 2.0
    plot_h = eh_ - e_pt - e_pb
    x_left = ex_ + e_pl
    x_right = x_left + plot_w + e_gx
    y0 = ey_ + e_pb

    ax_left = fig.add_axes([x_left, y0, plot_w, plot_h])
    ax_right = fig.add_axes([x_right, y0, plot_w, plot_h])

    # ---- e-left: descriptor spread (generated / real SD ratio) ----------
    style_axis(ax_left, grid=False)
    ax_left.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.9)
    ax_left.set_axisbelow(True)
    ax_left.set_title("Descriptor spread", pad=4.5, color=TEXT, fontweight="bold", fontsize=8.6)
    ax_left.set_ylabel("Generated / real descriptor SD", color=SUBTEXT, fontsize=7.2)

    if diversity is None:
        draw_unavailable(ax_left)
    else:
        rng = np.random.default_rng(11)
        ax_left.set_yscale("log")
        ax_left.axhline(1.0, color=SPINE, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        positions = [1, 2]
        for pos, g in zip(positions, ("poredit", "survol")):
            vals = diversity[g]
            ax_left.boxplot([vals], positions=[pos], widths=0.50, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color=TEXT, linewidth=1.2),
                            whiskerprops=dict(color=COLORS[g], linewidth=1.0),
                            capprops=dict(color=COLORS[g], linewidth=1.0),
                            boxprops=dict(facecolor=COLORS[g], edgecolor=COLORS[g],
                                          alpha=0.32, linewidth=1.0), zorder=3)
            jitter = pos + (rng.random(vals.size) - 0.5) * 0.26
            ax_left.scatter(jitter, vals, s=8.0, color=COLORS[g], alpha=0.55,
                            edgecolors="none", zorder=2)
        ax_left.set_xticks(positions)
        ax_left.set_xticklabels([LABELS["poredit"], LABELS["survol"]], fontsize=7.6)
        ax_left.set_xlim(0.4, 2.6)

    # ---- e-right: cross-property correlation discrepancy -----------------
    style_axis(ax_right, grid=False)
    ax_right.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.9)
    ax_right.set_axisbelow(True)
    ax_right.set_title("Cross-property correlation error", pad=4.5, color=TEXT, fontweight="bold",
                       fontsize=8.6)
    ax_right.set_ylabel("Mean |Δ correlation| ↓", color=SUBTEXT, fontsize=7.2)

    if correlation is None:
        draw_unavailable(ax_right)
    else:
        xs = [0, 1]
        vals = [correlation["poredit"], correlation["survol"]]
        colors = [COLORS["poredit"], COLORS["survol"]]
        ax_right.bar(xs, vals, width=0.55, color=colors, edgecolor="black", linewidth=0.8, zorder=3)
        ax_right.set_xticks(xs)
        ax_right.set_xticklabels([LABELS["poredit"], LABELS["survol"]], fontsize=7.6)
        ax_right.margins(y=0.18)


# ============================================================================
# 11. BUILD
# ============================================================================

def main():
    validate_package_root()

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PACKAGE_ROOT = {PACKAGE_ROOT}")
    for g in GROUPS:
        log(f"[paths] rep[{g}] = {REP_FILES[g]}")
        log(f"[paths] viz_folder_512[{g}] = {SAMPLE_DIRS_512[g]}")

    # ---- panel-a: validate transferred folders, load + validate reps -------
    for g in GROUPS:
        validate_visualization_folder(g, SAMPLE_DIRS_512[g])

    rep = {}
    for g in GROUPS:
        vol, phi = load_and_validate_representative(g, REP_FILES[g])
        rep[g] = {"vol": vol, "phi": phi, "file": REP_FILES[g]}

    validate_qc_manifest(REP_FILES)

    parallel_scale = 0.72 * max(rep[g]["vol"].shape[0] for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}
    render_audit = {}
    for g in GROUPS:
        render_group_isosurface(rep[g]["vol"], g, raw_paths[g], parallel_scale)
        render_audit[g] = {"file": str(rep[g]["file"]), "phi_pore": rep[g]["phi"],
                            "parallel_scale": parallel_scale, "downsample": DOWNSAMPLE}
        log(f"[panel-a-audit:{g}] file={rep[g]['file'].name}  phi(pore)={rep[g]['phi']:.4f}  "
            f"parallel_scale={parallel_scale:.4f}  downsample={DOWNSAMPLE}")
    finalize_renders(raw_paths, png_paths)

    # ---- panel b: long-range structure curves -------------------------------
    s2_curves = load_curve(S2_CSV, "s2_radial", "coordinate", "panel-b-s2")
    c2_curves = load_curve(C2_CSV, "pore_c2_radial", "lag_vox", "panel-b-c2")

    # ---- panel c: REV / scale convergence -----------------------------------
    rev_data = load_rev_scale()

    # ---- panel d: physical trade-off ratios ---------------------------------
    panel_d_rows = resolve_panel_d()

    # ---- panel e: ensemble representativeness -------------------------------
    diversity = load_diversity_ratios()
    correlation = load_correlation_discrepancy()

    # ---- canvas --------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.patch.set_facecolor(BG)

    add_card(fig, card_a)
    add_card(fig, card_b)
    add_card(fig, card_c)
    add_card(fig, card_d)
    add_card(fig, card_e)

    add_panel_label(fig, card_a[0] + 0.007, card_a[1] + card_a[3] + PANEL_LABEL_OFFSET, "a)")
    add_panel_label(fig, card_b[0] + 0.007, card_b[1] + card_b[3] + PANEL_LABEL_OFFSET, "b)")
    add_panel_label(fig, card_c[0] + 0.007, card_c[1] + card_c[3] + PANEL_LABEL_OFFSET, "c)")
    add_panel_label(fig, card_d[0] + 0.007, card_d[1] + card_d[3] + PANEL_LABEL_OFFSET, "d)")
    add_panel_label(fig, card_e[0] + 0.007, card_e[1] + card_e[3] + PANEL_LABEL_OFFSET, "e)")

    build_panel_a(fig, card_a, rep, png_paths)
    build_panel_b(fig, card_b, s2_curves, c2_curves)
    build_panel_c(fig, card_c, rev_data)
    build_panel_d(fig, card_d, panel_d_rows)
    build_panel_e(fig, card_e, diversity, correlation)

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

    # =========================== AUDIT JSON ==================================
    audit = {
        "package_root": str(PACKAGE_ROOT),
        "script": Path(__file__).name,
        "representative_ordinal": REP_ORDINAL,
        "central_index": CENTRAL_INDEX,
        "representative": {
            g: {"file": str(rep[g]["file"]), "phi_pore": rep[g]["phi"],
                "shape": list(rep[g]["vol"].shape), "labels": sorted(VALID_LABELS)}
            for g in GROUPS
        },
        "panel_a_render_audit": render_audit,
        "csv_paths": {
            "s2": str(S2_CSV), "c2": str(C2_CSV), "rev_scale": str(REV_CSV),
            "coordination": str(COORD_CSV), "transport": str(TRANSPORT_CSV),
            "capillary": str(CAPILLARY_CSV), "betti": str(BETTI_CSV),
            "diversity_core": str(DIVERSITY_CSV), "correlation_discrepancy": str(CORRELATION_CSV),
        },
        "panel_b_curve_keys": {"s2": "s2_radial", "c2": "pore_c2_radial"},
        "panel_b_available": {"s2": s2_curves is not None, "c2": c2_curves is not None},
        "panel_c_available": rev_data is not None,
        "panel_d_rows": panel_d_rows,
        "panel_e_diversity_medians": (
            {g: float(np.median(diversity[g])) for g in ("poredit", "survol")}
            if diversity is not None else None
        ),
        "panel_e_correlation_discrepancy": correlation,
        "saved": {"png": str(png), "pdf": str(pdf), "svg": str(svg), "tiff": str(tif_out)},
        "warnings": WARNINGS,
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out, AUDIT_JSON):
        log(" ", p.resolve())

    if WARNINGS:
        log(f"\n{len(WARNINGS)} warning(s) encountered (see figure_4_7_audit.json for the full list):")
        for w in WARNINGS:
            log(" -", w)
    else:
        log("\nNo warnings encountered -- every quantitative panel resolved from its official source.")


if __name__ == "__main__":
    sys.exit(main())
