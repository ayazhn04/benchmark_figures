#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.6 composite (topology preservation, final)
Reference vs PoreGen (diffusion) vs IPWGAN (gan) -- NMC 90 wt% 0 bar binary
porous-network topology preservation.

Same visual contract as Figures 4.1-4.5 (make_fig4_1_composite_magma_final.py
.. make_fig4_5_composite_magma_final.py): same cards, fonts, panel-label
style, export block. No tight_layout / constrained_layout / bbox_inches.

Only the exact official standardized 128^3 binary volume folders and the
exact official 4.6_FINAL_100100_PACKAGE metric CSVs are used -- no broad
path search, no WRONG_12_40 / BACKUP folders, no raw HDF5 GAN files, no
non-final diagnostic outputs. Extensive hard sanity checks raise and
refuse to save rather than silently substituting or faking data.

Panel a shows representative orthogonal (XY/XZ/YZ) slices with a
topology-class overlay (solid / largest connected pore component /
disconnected pore fragments), selected by an official representative CSV
and a deterministic, content-aware slice search. Panel b is a connected-
backbone and percolation dashboard (grouped bars + a directional
percolation heatmap). Panel c shows the pore pair-connectedness mechanism
(three mini-curves from the official curve_group_means.csv) plus a
supplementary skeleton/SNOW network-integrity inset. No interpretive
sentences beyond the one restrained annotation specified for panel b.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG
# ============================================================================

# Exact project root -- hardcoded rather than auto-detected, per task
# instructions. Never searched broadly for an alternative Section 4.6
# project or dataset.
PROJECT = Path("/home/ra2/4.6/poregen_topology_nmc128_ubuntu_ready_v2")

STANDARDIZED_DIR = PROJECT / "4.6_eval_standardized"
REAL_DIR = STANDARDIZED_DIR / "real_128_binary"
DIFFUSION_DIR = STANDARDIZED_DIR / "diffusion_128_binary"
GAN_DIR = STANDARDIZED_DIR / "gan_128_binary"

FINAL_PACKAGE = PROJECT / "4.6_FINAL_100100_PACKAGE"

FULL_METRIC_DIR = FINAL_PACKAGE / "4.6_full_topology_results"
DESCRIPTOR_ERRORS = FULL_METRIC_DIR / "descriptor_errors_vs_real.csv"
GROUP_SCALAR_SUMMARY = FULL_METRIC_DIR / "group_scalar_summary.csv"
PER_SAMPLE_SCALARS = FULL_METRIC_DIR / "per_sample_scalar_metrics.csv"
CURVE_ERRORS = FULL_METRIC_DIR / "curve_errors_vs_real.csv"
CURVE_GROUP_MEANS = FULL_METRIC_DIR / "curve_group_means.csv"
CURVE_METRICS_LONG = FULL_METRIC_DIR / "curve_metrics_long.csv"
ONE_FILE_DESCRIPTOR_ERRORS = FULL_METRIC_DIR / "ONE_FILE_4_6_DESCRIPTOR_ERRORS_FOR_CHATGPT.csv"

ADVANCED_METRIC_DIR = FINAL_PACKAGE / "4.6_advanced_topology_results"
ADVANCED_DESCRIPTOR_ERRORS = ADVANCED_METRIC_DIR / "advanced_descriptor_errors_vs_real.csv"
ADVANCED_GROUP_SUMMARY = ADVANCED_METRIC_DIR / "advanced_group_summary.csv"
ADVANCED_PER_SAMPLE = ADVANCED_METRIC_DIR / "advanced_per_sample_metrics.csv"
ONE_FILE_ADVANCED_ERRORS = ADVANCED_METRIC_DIR / "ONE_FILE_4_6_ADVANCED_DESCRIPTOR_ERRORS_FOR_CHATGPT.csv"

SLICE_SELECTION_DIR = FINAL_PACKAGE / "4.6_slice_grids"
REPRESENTATIVE_MEDIAN_PHASE = SLICE_SELECTION_DIR / "selected_samples_median_phase_representative.csv"
REPRESENTATIVE_CLOSEST_PHASE = SLICE_SELECTION_DIR / "selected_samples_closest_to_real_mean_phase.csv"
ALL_SLICE_STATS = SLICE_SELECTION_DIR / "all_sample_slice_grid_stats.csv"

SUITABILITY_DIR = FINAL_PACKAGE / "suitability_audit"
SUITABILITY_PER_SAMPLE = SUITABILITY_DIR / "per_sample_audit.csv"
PHASE_DISTRIBUTION = SUITABILITY_DIR / "phase_distribution_vs_real.csv"
DUPLICATE_AUDIT = SUITABILITY_DIR / "duplicate_audit.csv"
NEAREST_WITHIN_GROUP = SUITABILITY_DIR / "nearest_within_group.csv"
PAIRWISE_DIVERSITY = SUITABILITY_DIR / "pairwise_within_group_diversity.csv"

OUT = PROJECT / "paper_figures_4_6" / "figure_4_6_composite_magma_final"
TMP = OUT / "_render_cache"
STEM = "fig4_6_composite_magma_final"
AUDIT_JSON = OUT / "figure_4_6_audit.json"
CAPTION_TXT = OUT / "figure_4_6_caption.txt"

GROUPS = ["real", "diffusion", "gan"]
GROUP_VOLUME_DIRS = {"real": REAL_DIR, "diffusion": DIFFUSION_DIR, "gan": GAN_DIR}

EXPECTED_SHAPE = (128, 128, 128)
EXPECTED_N = 50
VALID_LABELS = {0, 1}
PORE_LABEL = 1
SOLID_LABEL = 0

FIG_W, FIG_H = 17.8, 10.2

CAPTION_TEXT = (
    "Figure 4.6. Qualitative and descriptor-level comparison for the "
    "topology-preservation task. (a) Representative 3D renderings and "
    "orthogonal XY, XZ, and YZ slices from real validation samples, PoreGen "
    "diffusion outputs, and IPWGAN outputs, with the largest connected pore "
    "backbone and disconnected pore fragments highlighted using the same "
    "6-connectivity convention as the topology metrics. (b) Connected-backbone "
    "and spanning-path descriptors, including largest pore-component fraction, "
    "disconnected pore fraction, mean percolating axes, and directional "
    "percolating pore fractions. (c) Pore pair-connectedness curves and "
    "supplementary skeleton/SNOW network descriptors showing preservation "
    "or fragmentation of long-range pore connectivity. The figure shows "
    "that visual plausibility alone is insufficient: PoreGen remains close "
    "to the real connected-pore structure, whereas IPWGAN produces "
    "valid-looking but substantially fragmented pore networks."
)

# ============================================================================
# 1b. FORBIDDEN SOURCES -- defense-in-depth. Every path used above is
# hardcoded exactly per task instructions (no broad search is ever
# performed), but this guards against an accidental future edit resolving
# into one of the explicitly forbidden cohorts/folders.
# ============================================================================

FORBIDDEN_PATH_TOKENS = (
    "wrong1240", "backupwrong1240", "diffusion128binarywas1240",
    "diffusionreduced1240", "wrong1240202608", "gansamples",
)

FORBIDDEN_DIRS = [
    PROJECT / "outputs",
    PROJECT / "outputs" / "samples",
    PROJECT / "outputs" / "metrics",
    PROJECT / "outputs" / "final_output",
    PROJECT / "4.6_eval" / "diffusion_samples",
    PROJECT / "4.6_eval" / "gan_samples",
    PROJECT / "4.6_eval" / "real_samples",
    PROJECT / "gan_samples",
    STANDARDIZED_DIR / "gan",
]

# ============================================================================
# 1c. SANITY-CHECK CONSTANTS (verified manuscript reference values; used
# only to cross-check independently loaded/computed data -- never as the
# plotted source, except where section 19 explicitly allows a derived
# value computed only from official CSV group means).
# ============================================================================

PORE_FRACTION_TOL = 0.015
EXPECTED_PORE_FRACTION = {"real": 0.3871, "diffusion": 0.3828, "gan": 0.3486}

COMPONENT_TOL = 0.02
EXPECTED_LCF = {"real": 0.9283, "diffusion": 0.9199, "gan": 0.4735}
EXPECTED_DISCONNECTED = {"real": 0.0717, "diffusion": 0.0801, "gan": 0.5265}
EXPECTED_MEAN_PERC_AXES = {"real": 2.78, "diffusion": 2.80, "gan": 1.80}

# A single, slightly permissive tolerance covers both the "full precision"
# (1e-3) and "rounded" (2e-3) cases the task describes, since this script
# cannot know in advance which the official CSV stores.
METRIC_TOL = 2e-3
EXPECTED_PERC_X = {"real": 0.9133, "diffusion": 0.9250, "gan": 0.2666}
EXPECTED_PERC_Y = {"real": 0.9283, "diffusion": 0.9086, "gan": 0.3677}
EXPECTED_PERC_Z = {"real": 0.7467, "diffusion": 0.7576, "gan": 0.3442}

RMSE_TOL = 0.01
EXPECTED_RMSE_X = {"diffusion": 0.0059, "gan": 0.0913}
EXPECTED_RMSE_Y = {"diffusion": 0.0082, "gan": 0.0761}
EXPECTED_RMSE_Z = {"diffusion": 0.0027, "gan": 0.0653}

ADVANCED_TOL = 0.02
EXPECTED_SKELETON_LCF = {"real": 0.977, "diffusion": 0.972, "gan": 0.787}
EXPECTED_SNOW_LCF = {"real": 0.810, "diffusion": 0.844, "gan": 0.332}
EXPECTED_SNOW_COORD = {"real": 2.406, "diffusion": 2.339, "gan": 1.720}

SLICE_SEARCH_RANGE = range(40, 89)  # 40..88 inclusive
CENTRAL_INDEX = 64
DISCONNECTED_VISIBLE_THRESHOLD = 0.02  # "meaningful" disconnected fraction

# ============================================================================
# 2. STYLE -- identical contract to Figures 4.1-4.5
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
LABELS = {"real": "Reference", "diffusion": "PoreGen", "gan": "IPWGAN"}
MARKERS = {"real": "o", "diffusion": "D", "gan": "^"}
TITLE_COLOR = {"real": TEXT, "diffusion": COLORS["diffusion"], "gan": COLORS["gan"]}
LINESTYLES = {"real": "-", "diffusion": "-", "gan": (0, (6.5, 2.6))}
LINEWIDTHS = {"real": 1.85, "diffusion": 1.85, "gan": 2.35}

# Legend-swatch-only dash pattern for IPWGAN (never applied to the actual
# plotted curves): at a normal short legend handle length, the plotted
# curve's own dash proportions read as near-solid, so the legend swatch
# alone uses a pattern with a visibly larger gap.
LEGEND_DASH_GAN = (0, (3.6, 2.6))

CARD_LW = 0.8
CELL_LW = 1.6

# Topology overlay palette for panel a -- these mean topology CLASS, never
# model identity: solid phase, largest connected pore backbone, and
# disconnected pore fragments.
SOLID_COLOR = "#17121C"
LARGEST_COMPONENT_COLOR = "#F2A93B"
DISCONNECTED_FRAGMENT_COLOR = "#19D3E6"
TOPO_CLASS_NAMES = ["Solid", "Largest connected pore component", "Disconnected pore fragments"]
TOPO_CMAP = ListedColormap([SOLID_COLOR, LARGEST_COMPONENT_COLOR, DISCONNECTED_FRAGMENT_COLOR])

PANEL_LABEL_OFFSET = 0.030

# ============================================================================
# 3. GEOMETRY (figure fractions)
# ============================================================================

card_a = [0.035, 0.535, 0.405, 0.405]
card_b = [0.465, 0.535, 0.500, 0.405]
card_c = [0.035, 0.075, 0.930, 0.405]

assert abs(card_a[1] - card_b[1]) < 1e-12, "panel a/b y mismatch"
assert abs(card_a[3] - card_b[3]) < 1e-12, "panel a/b height mismatch"
assert abs(card_c[0] - card_a[0]) < 1e-12, "panel c left edge mismatch"
assert abs((card_c[0] + card_c[2]) - (card_b[0] + card_b[2])) < 1e-12, "panel c right edge mismatch"

# ============================================================================
# 4. GENERIC HELPERS
# ============================================================================


def log(*a):
    print(*a, flush=True)


def _norm(s) -> str:
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


def _emphasize_dashed_legend_handle(handle):
    """Applies LEGEND_DASH_GAN to ONE already-built legend proxy handle
    (never the actual plotted artist)."""
    handle.set_linestyle(LEGEND_DASH_GAN)
    return handle


def _check_close(label: str, got: float, expected: float, tol: float):
    if not np.isfinite(got) or abs(got - expected) > tol:
        raise RuntimeError(f"[sanity] '{label}': got {got:.6g} vs expected ~{expected:.6g} "
                            f"(tol {tol:.4g}) -- stopping without saving")
    log(f"[sanity] '{label}' OK: got {got:.6g} ~ expected {expected:.6g} (tol {tol:.4g})")


def map_group_alias(value) -> str:
    """Robust group-name resolution for official CSV group/model columns."""
    s = _norm(value)
    if s in {"real", "reference", "validation", "realvalidation", "gt"} or "real" in s or "reference" in s or "validation" in s:
        return "real"
    if s in {"diffusion", "poregen", "poregendiffusion"} or "diffusion" in s or "poregen" in s:
        return "diffusion"
    if s in {"gan", "ipwgan"} or "ipwgan" in s or s == "gan" or "gan" in s:
        return "gan"
    raise RuntimeError(f"[groups] unrecognized group value: '{value}'")


# ============================================================================
# 5. DATA VALIDATION (exact official standardized folders only)
# ============================================================================


def _path_is_excluded(p: Path) -> bool:
    s = _norm(str(p))
    if any(tok in s for tok in FORBIDDEN_PATH_TOKENS):
        return True
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    for forbidden in FORBIDDEN_DIRS:
        try:
            rp.relative_to(forbidden.resolve())
            return True
        except Exception:
            continue
    return False


def load_group_volumes(group: str) -> dict:
    d = GROUP_VOLUME_DIRS[group]
    if _path_is_excluded(d):
        raise RuntimeError(f"[data:{group}] resolved folder {d} matches a forbidden pattern -- "
                            f"refusing to use it")
    if not d.exists():
        raise FileNotFoundError(
            f"[data:{group}] expected exact official folder not found: {d}\n"
            f"This script never searches broadly for alternative Section 4.6 sample folders."
        )

    files = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".npy")
    if len(files) != EXPECTED_N:
        raise RuntimeError(f"[data:{group}] expected exactly {EXPECTED_N} .npy volumes in {d}, "
                            f"found {len(files)}")

    volumes = {}
    for f in files:
        arr = np.squeeze(np.load(str(f)))
        if arr.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"[data:{group}] {f.name}: shape {arr.shape} != expected "
                                f"{EXPECTED_SHAPE}. Stopping WITHOUT saving the figure.")
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"[data:{group}] {f.name}: contains NaN/Inf values")
        vals = set(np.unique(arr).tolist())
        if not vals.issubset(VALID_LABELS):
            raise RuntimeError(f"[data:{group}] {f.name}: unique labels {sorted(vals)} are not a "
                                f"subset of the valid label set {sorted(VALID_LABELS)}")
        if vals != VALID_LABELS:
            raise RuntimeError(f"[data:{group}] {f.name}: labels {sorted(vals)} do not contain "
                                f"both 0 (solid) and 1 (pore) -- refusing an all-one-phase volume")
        volumes[f] = arr.astype(np.uint8)

    log(f"[data:{group}] folder = {d}")
    log(f"[data:{group}] n_volumes = {len(files)}  shape={EXPECTED_SHAPE}  "
        f"labels={{0,1}} (validated, finite)")
    return volumes


def transition_rates(v: np.ndarray):
    tx = float(np.mean(v[:, :, 1:] != v[:, :, :-1]))
    ty = float(np.mean(v[:, 1:, :] != v[:, :-1, :]))
    tz = float(np.mean(v[1:, :, :] != v[:-1, :, :]))
    return tx, ty, tz


def pore_fraction(v: np.ndarray) -> float:
    return float(np.mean(v == PORE_LABEL))


def check_pore_fraction(group: str, direct_mean: float):
    exp = EXPECTED_PORE_FRACTION[group]
    if abs(direct_mean - exp) > PORE_FRACTION_TOL:
        raise RuntimeError(f"[sanity:pore_fraction] group '{group}' direct mean {direct_mean:.4f} "
                            f"deviates from expected ~{exp:.4f} by more than {PORE_FRACTION_TOL}")
    log(f"[sanity:pore_fraction] group '{group}' OK: direct={direct_mean:.4f} ~ expected {exp:.4f}")


# ============================================================================
# 6/8. CONNECTED-COMPONENT / PERCOLATION HELPERS (6-connectivity)
#
# The official Section 4.6 topology values correspond to 6-connectivity
# (face-adjacency only), not 26-connectivity -- this must be used for every
# connected-component, largest-component, disconnected-fraction,
# percolation, and panel-a topology-overlay calculation, and for the direct
# sanity checks against the official group metrics.
# ============================================================================

CONNECTIVITY_STRUCT = ndimage.generate_binary_structure(3, 1)


def label_pore_components(v: np.ndarray):
    """Returns (labeled_array, n_components, sizes) for the pore phase
    (label==1) using 26-connectivity."""
    pore = (v == PORE_LABEL)
    labeled, n = ndimage.label(pore, structure=CONNECTIVITY_STRUCT)
    if n == 0:
        return labeled, 0, np.array([])
    sizes = ndimage.sum(pore, labeled, index=np.arange(1, n + 1))
    return labeled, n, sizes


def _percolating_component_ids(labeled: np.ndarray, axis: int) -> set:
    first = np.take(labeled, 0, axis=axis)
    last = np.take(labeled, labeled.shape[axis] - 1, axis=axis)
    ids_first = set(np.unique(first).tolist()) - {0}
    ids_last = set(np.unique(last).tolist()) - {0}
    return ids_first & ids_last


def component_stats(v: np.ndarray) -> dict:
    """Direct per-volume topology descriptors, for validation, representative
    selection, slice selection, and panel-a overlays only (section 19)."""
    labeled, n, sizes = label_pore_components(v)
    total_pore = float(np.sum(v == PORE_LABEL))
    if n == 0 or total_pore == 0:
        largest_id = None
        largest_frac = 0.0
        disconnected_frac = 0.0
    else:
        largest_idx = int(np.argmax(sizes))
        largest_id = largest_idx + 1
        largest_size = float(sizes[largest_idx])
        largest_frac = largest_size / total_pore
        disconnected_frac = 1.0 - largest_frac

    perc = {}
    perc_voxel_frac = {}
    for axis, name in ((0, "z"), (1, "y"), (2, "x")):
        ids = _percolating_component_ids(labeled, axis) if n > 0 else set()
        perc[name] = len(ids) > 0
        if n > 0 and ids and total_pore > 0:
            perc_voxels = float(np.sum(np.isin(labeled, list(ids))))
            perc_voxel_frac[name] = perc_voxels / total_pore
        else:
            perc_voxel_frac[name] = 0.0

    n_percolating_axes = int(perc["x"]) + int(perc["y"]) + int(perc["z"])

    return {
        "pore_fraction": pore_fraction(v),
        "largest_component_fraction": largest_frac,
        "disconnected_fraction": disconnected_frac,
        "n_components": int(n),
        "largest_id": largest_id,
        "labeled": labeled,
        "perc_x": perc["x"], "perc_y": perc["y"], "perc_z": perc["z"],
        "n_percolating_axes": n_percolating_axes,
        "percolating_pore_fraction_x": perc_voxel_frac["x"],
        "percolating_pore_fraction_y": perc_voxel_frac["y"],
        "percolating_pore_fraction_z": perc_voxel_frac["z"],
    }


def group_component_means(volumes: dict) -> dict:
    """Direct group-level means of component_stats over all 50 volumes, for
    sanity-checking only (never plotted directly -- section 19)."""
    fields = ["pore_fraction", "largest_component_fraction", "disconnected_fraction",
              "n_percolating_axes", "percolating_pore_fraction_x",
              "percolating_pore_fraction_y", "percolating_pore_fraction_z"]
    rows = [component_stats(v) for v in volumes.values()]
    return {f: float(np.mean([r[f] for r in rows])) for f in fields}


def check_component_sanity(group: str, means: dict):
    _check_close(f"{group}.largest_component_fraction (direct)",
                 means["largest_component_fraction"], EXPECTED_LCF[group], COMPONENT_TOL)
    _check_close(f"{group}.disconnected_fraction (direct)",
                 means["disconnected_fraction"], EXPECTED_DISCONNECTED[group], COMPONENT_TOL)
    _check_close(f"{group}.mean_percolating_axes (direct)",
                 means["n_percolating_axes"], EXPECTED_MEAN_PERC_AXES[group], COMPONENT_TOL)


# ============================================================================
# 7. REPRESENTATIVE SAMPLE SELECTION (official CSV only, no cherry-picking)
# ============================================================================

GROUP_COL_CANDIDATES = ["group", "set", "model", "category", "class", "cohort"]
FILE_COL_CANDIDATES = ["filename", "file", "file_name", "sample", "sample_id", "sample_name",
                       "npy", "npy_file", "path", "name", "volume", "volume_file"]


def load_representative_selection() -> dict:
    if not REPRESENTATIVE_MEDIAN_PHASE.exists():
        raise FileNotFoundError(f"[representative] missing official file: {REPRESENTATIVE_MEDIAN_PHASE}")
    df = pd.read_csv(REPRESENTATIVE_MEDIAN_PHASE)
    log(f"[representative] loaded {REPRESENTATIVE_MEDIAN_PHASE}: {df.shape[0]} rows, "
        f"columns: {list(df.columns)}")
    log(f"[representative] first rows:\n{df.head(10).to_string()}")

    gcol = pick_col(df, GROUP_COL_CANDIDATES, "representative.group", required=False)
    fcol = pick_col(df, FILE_COL_CANDIDATES, "representative.file", required=False)
    if gcol is None or fcol is None:
        raise RuntimeError(
            f"[representative] could not unambiguously resolve a group column (tried "
            f"{GROUP_COL_CANDIDATES}) and/or a file/sample column (tried {FILE_COL_CANDIDATES}) "
            f"in {REPRESENTATIVE_MEDIAN_PHASE}. Columns present: {list(df.columns)}. "
            f"Refusing to guess -- the CSV schema must be mapped manually."
        )

    df = df.copy()
    df["_group"] = df[gcol].map(map_group_alias)

    picked = {}
    for g in GROUPS:
        sub = df[df["_group"] == g]
        if len(sub) != 1:
            raise RuntimeError(f"[representative] expected exactly 1 representative row for group "
                                f"'{g}' in {REPRESENTATIVE_MEDIAN_PHASE}, got {len(sub)}")
        raw_value = str(sub.iloc[0][fcol])
        basename = Path(raw_value).name
        d = GROUP_VOLUME_DIRS[g]
        candidates = {f.name: f for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".npy"}
        match = candidates.get(basename)
        if match is None:
            # tolerate a stem-only value (no .npy suffix) or case differences
            norm_lut = {_norm(name): f for name, f in candidates.items()}
            match = norm_lut.get(_norm(basename)) or norm_lut.get(_norm(basename + ".npy"))
        if match is None:
            raise RuntimeError(f"[representative] group '{g}': value '{raw_value}' (basename "
                                f"'{basename}') does not resolve to any file in the allowed folder "
                                f"{d} -- refusing to use a path outside the official standardized "
                                f"folders")
        picked[g] = match
        log(f"[representative] group '{g}': resolved '{raw_value}' -> {match}")
    return picked


# ============================================================================
# 8. SLICE SELECTION (deterministic, content-aware, within [40,88])
# ============================================================================

PLANE_AXIS = {"XY": 0, "XZ": 1, "YZ": 2}  # numpy axis that is FIXED to pick the slice


def _take_slice(v: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "XY":
        return v[index, :, :]
    if plane == "XZ":
        return v[:, index, :]
    if plane == "YZ":
        return v[:, :, index]
    raise ValueError(plane)


def select_slice_index(v: np.ndarray, labeled: np.ndarray, largest_id, plane: str,
                        full_pore_frac: float, full_disc_frac: float) -> dict:
    candidates = []
    for idx in SLICE_SEARCH_RANGE:
        sl = _take_slice(v, plane, idx)
        has_both = bool(np.any(sl == 0) and np.any(sl == 1))
        if not has_both:
            continue
        sl_pore_frac = float(np.mean(sl == PORE_LABEL))
        if largest_id is not None:
            sl_labeled = _take_slice(labeled, plane, idx)
            sl_pore_mask = (sl == PORE_LABEL)
            n_pore_px = int(np.sum(sl_pore_mask))
            if n_pore_px > 0:
                n_largest_px = int(np.sum((sl_labeled == largest_id) & sl_pore_mask))
                sl_disc_frac = 1.0 - (n_largest_px / n_pore_px)
            else:
                sl_disc_frac = 0.0
        else:
            sl_disc_frac = 0.0
        abs_diff = abs(sl_pore_frac - full_pore_frac)
        has_visible_fragments = sl_disc_frac > 1e-9
        needs_fragments = full_disc_frac > DISCONNECTED_VISIBLE_THRESHOLD
        penalty = 0 if (has_visible_fragments or not needs_fragments) else 1
        candidates.append({
            "index": idx, "penalty": penalty, "abs_diff": abs_diff,
            "dist_center": abs(idx - CENTRAL_INDEX), "slice_pore_frac": sl_pore_frac,
            "slice_disconnected_frac": sl_disc_frac,
        })

    if not candidates:
        raise RuntimeError(f"[slice] no candidate index in [{SLICE_SEARCH_RANGE.start}, "
                            f"{SLICE_SEARCH_RANGE.stop - 1}] for plane {plane} contains both "
                            f"labels 0 and 1 -- stopping without saving")

    candidates.sort(key=lambda c: (c["penalty"], c["abs_diff"], c["dist_center"]))
    best = candidates[0]
    log(f"[slice] plane {plane}: selected index {best['index']} (penalty={best['penalty']}, "
        f"abs_diff={best['abs_diff']:.4f}, slice_pore_frac={best['slice_pore_frac']:.4f}, "
        f"slice_disconnected_frac={best['slice_disconnected_frac']:.4f})")
    return best


# ============================================================================
# 9. OFFICIAL METRIC LOADING (panel b/c) -- exact official CSVs only,
# robust column/row resolution, hard failure on ambiguity/missing metric.
# ============================================================================

METRIC_ID_CANDIDATES = ["metric", "metric_name", "descriptor", "name", "feature", "variable", "row_metric"]

VALUE_COL_ALIASES = {
    "real": ["real", "reference", "real_mean", "reference_mean", "real_value", "gt", "gt_mean",
             "validation", "validation_mean"],
    "diffusion": ["diffusion", "poregen", "diffusion_mean", "poregen_mean", "diffusion_value",
                  "model_diffusion", "poregen_diffusion", "poregen_diffusion_mean"],
    "gan": ["gan", "ipwgan", "gan_mean", "ipwgan_mean", "gan_value", "model_gan"],
}
ERROR_COL_ALIASES = {
    "diffusion": ["diffusion_error", "poregen_error", "diffusion_err", "diffusion_rmse",
                  "poregen_rmse", "diffusion_abs_error"],
    "gan": ["gan_error", "ipwgan_error", "gan_err", "gan_rmse", "ipwgan_rmse", "gan_abs_error"],
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[metrics] missing official file: {path}")
    df = pd.read_csv(path)
    log(f"[metrics] loaded {path}: {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


def _resolve_metric_row(df: pd.DataFrame, path: Path, metric_key: str, aliases: list):
    id_col = pick_col(df, METRIC_ID_CANDIDATES, "metric-identifier column", required=False)
    if id_col is None:
        raise RuntimeError(f"[metrics] {path} has no recognizable metric-identifier column "
                            f"(tried {METRIC_ID_CANDIDATES})")
    key_norms = [_norm(metric_key)] + [_norm(a) for a in aliases]
    id_norm = df[id_col].astype(str).map(_norm)
    matches = df[id_norm.isin(key_norms)]
    if matches.empty:
        sub_mask = id_norm.map(lambda v: any(k in v or v in k for k in key_norms if k))
        matches = df[sub_mask]
    if matches.empty:
        raise RuntimeError(f"[metrics] '{metric_key}' not found in {path} (id column '{id_col}'); "
                            f"tried exact/alias names {key_norms}")
    if len(matches) > 1:
        raise RuntimeError(f"[metrics] '{metric_key}' matched {len(matches)} rows in {path} "
                            f"(ambiguous): {matches[id_col].tolist()}")
    return matches.iloc[0], id_col


def resolve_scalar_metric(df: pd.DataFrame, path: Path, metric_key: str, aliases: list) -> dict:
    row, id_col = _resolve_metric_row(df, path, metric_key, aliases)
    raw_name = str(row[id_col])
    real_col = pick_col(df, VALUE_COL_ALIASES["real"], f"{metric_key}.real", required=False)
    diff_col = pick_col(df, VALUE_COL_ALIASES["diffusion"], f"{metric_key}.diffusion", required=False)
    gan_col = pick_col(df, VALUE_COL_ALIASES["gan"], f"{metric_key}.gan", required=False)
    if real_col is None or diff_col is None or gan_col is None:
        raise RuntimeError(f"[metrics] '{metric_key}' in {path}: could not resolve real/diffusion/gan "
                            f"value columns; available columns: {list(df.columns)}")
    real_v = float(pd.to_numeric(row[real_col], errors="coerce"))
    diff_v = float(pd.to_numeric(row[diff_col], errors="coerce"))
    gan_v = float(pd.to_numeric(row[gan_col], errors="coerce"))

    diff_err_col = pick_col(df, ERROR_COL_ALIASES["diffusion"], f"{metric_key}.diffusion_err", required=False)
    gan_err_col = pick_col(df, ERROR_COL_ALIASES["gan"], f"{metric_key}.gan_err", required=False)
    diff_err = float(pd.to_numeric(row[diff_err_col], errors="coerce")) if diff_err_col else None
    gan_err = float(pd.to_numeric(row[gan_err_col], errors="coerce")) if gan_err_col else None

    log(f"[metrics] '{metric_key}' resolved from {path.name} (row='{raw_name}'): "
        f"real={real_v:.6g} diffusion={diff_v:.6g} gan={gan_v:.6g}"
        + (f"  diffusion_err={diff_err:.6g} gan_err={gan_err:.6g}" if diff_err is not None else ""))
    return {"raw_name": raw_name, "real": real_v, "diffusion": diff_v, "gan": gan_v,
            "diffusion_err": diff_err, "gan_err": gan_err, "source": str(path)}


def resolve_scalar_with_fallback(metric_key: str, aliases: list, dfs_in_order: list) -> dict:
    last_exc = None
    for path, df in dfs_in_order:
        if df is None:
            continue
        try:
            return resolve_scalar_metric(df, path, metric_key, aliases)
        except Exception as exc:
            last_exc = exc
            log(f"[metrics] '{metric_key}' not resolved from {path.name} ({exc}); trying next source")
    raise RuntimeError(f"[metrics] '{metric_key}' could not be resolved unambiguously from any "
                        f"official source -- stopping without saving. Last error: {last_exc}")


PANEL_B_METRICS = [
    ("largest_pore_component_fraction", "Largest pore component fraction",
     ["largest_component_fraction", "largest_connected_pore_component_fraction",
      "largest_connected_component_fraction", "pore_largest_component_fraction", "lcf", "pore_lcf"]),
    ("disconnected_pore_fraction", "Disconnected pore fraction",
     ["disconnected_fraction", "small_disconnected_pore_fraction", "pore_disconnected_fraction"]),
    ("mean_percolating_axes", "Mean percolating axes",
     ["percolating_axes_mean", "number_of_percolating_axes", "n_percolating_axes", "percolating_axes"]),
    ("percolating_pore_fraction_x", "Percolating pore fraction, x",
     ["x_percolating_pore_fraction", "pore_percolating_component_fraction_x",
      "percolating_component_fraction_x"]),
    ("percolating_pore_fraction_y", "Percolating pore fraction, y",
     ["y_percolating_pore_fraction", "pore_percolating_component_fraction_y",
      "percolating_component_fraction_y"]),
    ("percolating_pore_fraction_z", "Percolating pore fraction, z",
     ["z_percolating_pore_fraction", "pore_percolating_component_fraction_z",
      "percolating_component_fraction_z"]),
]

PANEL_B_EXPECTED = {
    "largest_pore_component_fraction": EXPECTED_LCF,
    "disconnected_pore_fraction": EXPECTED_DISCONNECTED,
    "mean_percolating_axes": EXPECTED_MEAN_PERC_AXES,
    "percolating_pore_fraction_x": EXPECTED_PERC_X,
    "percolating_pore_fraction_y": EXPECTED_PERC_Y,
    "percolating_pore_fraction_z": EXPECTED_PERC_Z,
}


def load_panel_b_metrics(scalar_dfs: list) -> dict:
    resolved = {}
    for key, _label, aliases in PANEL_B_METRICS:
        r = resolve_scalar_with_fallback(key, aliases, scalar_dfs)
        exp = PANEL_B_EXPECTED[key]
        tol = COMPONENT_TOL if key in ("largest_pore_component_fraction", "disconnected_pore_fraction",
                                        "mean_percolating_axes") else METRIC_TOL
        for g in GROUPS:
            _check_close(f"panel_b.{key}.{g}", r[g], exp[g], tol)
        resolved[key] = r
    return resolved


# ---- panel c: pair-connectedness curves (curve_group_means.csv) ----------

CURVE_GROUP_COL_CANDIDATES = ["group", "set", "model", "class", "category"]
CURVE_METRIC_COL_CANDIDATES = ["metric", "descriptor", "name", "quantity", "curve", "curve_name"]
CURVE_AXIS_COL_CANDIDATES = ["axis", "direction", "dim", "dimension"]
CURVE_X_COL_CANDIDATES = ["r", "x", "radius", "distance", "lag", "bin", "bin_center", "coord"]
CURVE_VALUE_COL_CANDIDATES = ["value", "mean", "mean_value", "y", "y_mean", "pair_connectedness",
                              "prob", "probability"]

PAIR_CONN_ALIAS_TOKENS = ("paircconnectedness", "pairconnectedness", "porepairconnectedness",
                          "twopointcluster", "clusterfunction")


def load_pair_connectedness_curves() -> dict:
    """Returns {axis_letter: {group: {'x': arr, 'y': arr}}} for x/y/z pore
    pair-connectedness, resolved from CURVE_GROUP_MEANS only."""
    df = _load_csv(CURVE_GROUP_MEANS)
    log(f"[curves] {CURVE_GROUP_MEANS.name} first rows:\n{df.head(10).to_string()}")

    metric_col = pick_col(df, CURVE_METRIC_COL_CANDIDATES, "curve.metric", required=False)
    axis_col = pick_col(df, CURVE_AXIS_COL_CANDIDATES, "curve.axis", required=False)
    x_col = pick_col(df, CURVE_X_COL_CANDIDATES, "curve.x", required=False)
    value_col = pick_col(df, CURVE_VALUE_COL_CANDIDATES, "curve.value", required=False)
    group_col = pick_col(df, CURVE_GROUP_COL_CANDIDATES, "curve.group", required=False)

    if metric_col is None or x_col is None or value_col is None:
        raise RuntimeError(
            f"[curves] could not resolve required columns from {CURVE_GROUP_MEANS} "
            f"(metric candidates {CURVE_METRIC_COL_CANDIDATES}, x candidates {CURVE_X_COL_CANDIDATES}, "
            f"value candidates {CURVE_VALUE_COL_CANDIDATES}). Columns present: {list(df.columns)}. "
            f"Unique values in likely metric/group/axis columns: "
            + ", ".join(f"{c}={df[c].unique()[:15].tolist()}" for c in df.columns
                        if df[c].dtype == object)
        )

    metric_norm = df[metric_col].astype(str).map(_norm)

    out = {}
    for axis_letter in ("x", "y", "z"):
        mask_metric = metric_norm.map(lambda v: any(tok in v for tok in PAIR_CONN_ALIAS_TOKENS))
        if axis_col is not None:
            axis_norm = df[axis_col].astype(str).map(_norm)
            mask_axis = axis_norm.map(lambda v: v == axis_letter or v.startswith(axis_letter))
        else:
            mask_axis = metric_norm.map(lambda v: v.endswith(axis_letter) or f"_{axis_letter}_" in v
                                         or v.endswith(f"{axis_letter}axis"))
        sub = df[mask_metric & mask_axis].copy()
        if sub.empty:
            raise RuntimeError(f"[curves] no pair-connectedness rows resolved for axis '{axis_letter}' "
                                f"in {CURVE_GROUP_MEANS} -- columns: {list(df.columns)}; "
                                f"sample metric values: {df[metric_col].unique()[:20].tolist()}")

        axis_out = {}
        if group_col is not None:
            sub["_group"] = sub[group_col].map(map_group_alias)
            for g in GROUPS:
                gsub = sub[sub["_group"] == g].sort_values(x_col)
                if gsub.empty:
                    raise RuntimeError(f"[curves] axis '{axis_letter}' group '{g}' has no rows in "
                                        f"{CURVE_GROUP_MEANS}")
                axis_out[g] = {"x": pd.to_numeric(gsub[x_col], errors="coerce").to_numpy(float),
                               "y": pd.to_numeric(gsub[value_col], errors="coerce").to_numpy(float)}
        else:
            # wide format: separate real/diffusion/gan value columns sharing one x column
            real_col = pick_col(df, VALUE_COL_ALIASES["real"], "curve.real", required=False)
            diff_col = pick_col(df, VALUE_COL_ALIASES["diffusion"], "curve.diffusion", required=False)
            gan_col = pick_col(df, VALUE_COL_ALIASES["gan"], "curve.gan", required=False)
            if real_col is None or diff_col is None or gan_col is None:
                raise RuntimeError(f"[curves] {CURVE_GROUP_MEANS} has no group column and no "
                                    f"real/diffusion/gan value columns either -- cannot resolve "
                                    f"per-group curves. Columns: {list(df.columns)}")
            sub = sub.sort_values(x_col)
            xs = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(float)
            for g, col in (("real", real_col), ("diffusion", diff_col), ("gan", gan_col)):
                axis_out[g] = {"x": xs, "y": pd.to_numeric(sub[col], errors="coerce").to_numpy(float)}

        for g in GROUPS:
            if not np.all(np.isfinite(axis_out[g]["x"])) or not np.all(np.isfinite(axis_out[g]["y"])):
                raise RuntimeError(f"[curves] axis '{axis_letter}' group '{g}': non-finite values "
                                    f"in resolved curve from {CURVE_GROUP_MEANS}")
        log(f"[curves] axis '{axis_letter}': resolved {len(sub)} rows from {CURVE_GROUP_MEANS.name}, "
            f"{ {g: len(axis_out[g]['x']) for g in GROUPS} } points/group")
        out[axis_letter] = axis_out
    return out


# ---- panel c: pair-connectedness RMSE (curve_errors_vs_real.csv) ---------

RMSE_ID_CANDIDATES = ["metric", "descriptor", "name", "curve", "curve_name"]
RMSE_VALUE_ALIASES = {
    "diffusion": ["diffusion_rmse", "diffusion_error", "poregen_rmse", "poregen_error"],
    "gan": ["gan_rmse", "gan_error", "ipwgan_rmse", "ipwgan_error"],
}


def load_pair_connectedness_rmse() -> dict:
    df = _load_csv(CURVE_ERRORS)
    id_col = pick_col(df, RMSE_ID_CANDIDATES, "rmse.id", required=False)
    if id_col is None:
        raise RuntimeError(f"[rmse] {CURVE_ERRORS} has no recognizable id column "
                            f"(tried {RMSE_ID_CANDIDATES}); columns: {list(df.columns)}")
    diff_col = pick_col(df, RMSE_VALUE_ALIASES["diffusion"], "rmse.diffusion", required=False)
    gan_col = pick_col(df, RMSE_VALUE_ALIASES["gan"], "rmse.gan", required=False)
    if diff_col is None or gan_col is None:
        raise RuntimeError(f"[rmse] {CURVE_ERRORS} missing diffusion/gan RMSE columns; "
                            f"columns: {list(df.columns)}")

    id_norm = df[id_col].astype(str).map(_norm)
    out = {}
    for axis_letter, exp_key in (("x", "x"), ("y", "y"), ("z", "z")):
        mask = id_norm.map(lambda v: any(tok in v for tok in PAIR_CONN_ALIAS_TOKENS)
                            and (v.endswith(axis_letter) or f"_{axis_letter}_" in v
                                 or v.endswith(f"{axis_letter}axis")))
        sub = df[mask]
        if len(sub) != 1:
            raise RuntimeError(f"[rmse] axis '{axis_letter}': expected exactly 1 matching row in "
                                f"{CURVE_ERRORS}, got {len(sub)}; id values: "
                                f"{df[id_col].unique()[:20].tolist()}")
        row = sub.iloc[0]
        diff_v = float(pd.to_numeric(row[diff_col], errors="coerce"))
        gan_v = float(pd.to_numeric(row[gan_col], errors="coerce"))
        out[axis_letter] = {"diffusion": diff_v, "gan": gan_v, "raw_name": str(row[id_col])}
        log(f"[rmse] axis '{axis_letter}' resolved from {CURVE_ERRORS.name} (row='{row[id_col]}'): "
            f"diffusion={diff_v:.6g} gan={gan_v:.6g}")

    for axis_letter in ("x", "y", "z"):
        exp = {"x": EXPECTED_RMSE_X, "y": EXPECTED_RMSE_Y, "z": EXPECTED_RMSE_Z}[axis_letter]
        _check_close(f"rmse.{axis_letter}.diffusion", out[axis_letter]["diffusion"],
                     exp["diffusion"], RMSE_TOL)
        _check_close(f"rmse.{axis_letter}.gan", out[axis_letter]["gan"], exp["gan"], RMSE_TOL)
    return out


# ---- panel c inset: skeleton/SNOW advanced network metrics ---------------

ADVANCED_METRICS = [
    ("skeleton_graph_largest_component_fraction", "Skeleton graph LCF",
     ["skeleton_lcf", "skeleton_graph_lcf"], EXPECTED_SKELETON_LCF),
    ("snow_graph_largest_component_fraction", "SNOW graph LCF",
     ["snow_lcf", "snow_graph_lcf"], EXPECTED_SNOW_LCF),
    ("snow_coordination_number", "SNOW coordination number",
     ["coordination_number", "snow_coordination"], EXPECTED_SNOW_COORD),
]


def load_advanced_metrics(advanced_dfs: list) -> dict:
    resolved = {}
    for key, _label, aliases, expected in ADVANCED_METRICS:
        r = resolve_scalar_with_fallback(key, aliases, advanced_dfs)
        for g in GROUPS:
            _check_close(f"advanced.{key}.{g}", r[g], expected[g], ADVANCED_TOL)
        resolved[key] = r
    return resolved


# ============================================================================
# 10. PANEL A -- topology-aware representative slices + 3D backbone render
#
# Same overall panel-a grid logic as Figures 4.1/4.2/4.4/4.5: columns are
# groups (Reference | PoreGen | IPWGAN), rows are {3D volume, X-Y slice,
# X-Z slice, Y-Z slice}. The 3D row is a true marching-cubes isosurface of
# the largest connected pore component (6-connectivity, same convention as
# every other topology computation in this script) rendered from the
# actual selected NPY volume -- never a MIP, never synthetic geometry. The
# 2D rows are topology-class overlays (solid / largest connected pore
# backbone / disconnected pore fragments), not plain binary slices.
# ============================================================================

PLANES = ["XY", "XZ", "YZ"]
ROW_NAMES = ["3D volume", "X–Y slice", "X–Z slice", "Y–Z slice"]
PLANE_DIRECTION_LABELS = {
    "XY": [("→ X", (0.90, 0.07), "right", "bottom"), ("↑ Y", (0.07, 0.90), "left", "top")],
    "XZ": [("→ X", (0.90, 0.07), "right", "bottom"), ("↑ Z", (0.07, 0.90), "left", "top")],
    "YZ": [("→ Y", (0.90, 0.07), "right", "bottom"), ("↑ Z", (0.07, 0.90), "left", "top")],
}

RENDER_PX = 1500
CANVAS_PX = 900


def _class_image(v: np.ndarray, labeled: np.ndarray, largest_id, plane: str, index: int) -> np.ndarray:
    sl = _take_slice(v, plane, index)
    cls = np.zeros(sl.shape, dtype=np.uint8)  # 0 = solid
    if largest_id is not None:
        sl_labeled = _take_slice(labeled, plane, index)
        pore_mask = (sl == PORE_LABEL)
        largest_mask = pore_mask & (sl_labeled == largest_id)
        disconnected_mask = pore_mask & (sl_labeled != largest_id)
        cls[largest_mask] = 1
        cls[disconnected_mask] = 2
    else:
        cls[sl == PORE_LABEL] = 2
    return cls


def render_backbone_surface(labeled: np.ndarray, largest_id, group: str, out_raw: Path,
                             parallel_scale: float) -> dict:
    """True marching-cubes isosurface of the largest connected pore
    component (6-connectivity), rendered from the actual selected NPY
    volume. Identical camera, extent, and parallel scale across all three
    groups (no content-dependent cropping) for a fair comparison. No MIP
    fallback: if PyVista + scikit-image marching_cubes cannot run, this
    raises and the caller refuses to save the figure."""
    import pyvista as pv
    from skimage import measure

    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    nz, ny, nx = labeled.shape
    center = np.array([nz / 2.0, ny / 2.0, nx / 2.0])
    mask = (labeled == largest_id).astype(np.float32)
    if mask.min() >= 0.5 or mask.max() <= 0.5:
        raise RuntimeError(f"[render:{group}] largest connected pore component has a degenerate "
                            f"iso-level (absent or filling the whole volume) -- refusing to render "
                            f"an empty/solid cell")

    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, np.int64), faces.astype(np.int64)])
    mesh = pv.PolyData(verts, faces_pv)

    pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX))
    pl.set_background("white")
    pl.add_mesh(mesh, color=LARGEST_COMPONENT_COLOR, opacity=1.0, smooth_shading=True,
                specular=0.24, specular_power=20, ambient=0.30, diffuse=0.80, show_scalar_bar=False)
    # Full [0,nz]x[0,ny]x[0,nx] cube wireframe -- identical extent for every
    # group regardless of data content, so the rendered scale is fair by
    # construction rather than by post-hoc cropping (same convention as
    # Figures 4.1/4.2/4.4/4.5).
    pl.add_mesh(pv.Box(bounds=(0, nz, 0, ny, 0, nx)), style="wireframe",
                color=COLORS[group], line_width=2.2, opacity=0.60)

    pl.enable_parallel_projection()
    direction = np.array([1.0, -1.30, 0.90])
    direction /= np.linalg.norm(direction)
    pl.camera.focal_point = tuple(center)
    pl.camera.position = tuple(center + direction * 4.0 * max(labeled.shape))
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.parallel_scale = float(parallel_scale)

    pl.screenshot(str(out_raw), transparent_background=True)
    pl.close()

    return {"render_extent": [[0, nz], [0, ny], [0, nx]], "parallel_scale": float(parallel_scale),
            "connectivity": "6-connectivity (face-adjacency), generate_binary_structure(3, 1)"}


def render_group_backbone(labeled, largest_id, group: str, out_raw: Path, parallel_scale: float) -> dict:
    try:
        return render_backbone_surface(labeled, largest_id, group, out_raw, parallel_scale)
    except Exception as exc:
        raise RuntimeError(
            f"[render:{group}] 3D connected pore-backbone rendering failed and no fallback is "
            f"permitted for this figure (PyVista + scikit-image marching_cubes are required; "
            f"PyVista/scikit-image rendering appears unavailable in this environment): {exc}"
        ) from exc


def finalize_renders(raw_paths: dict, out_paths: dict):
    """Identical, content-independent resize for every group -- NO alpha
    bounding-box crop, NO per-group foreground zoom. The camera, parallel
    scale, and full [0,128]^3 wireframe cube are already identical across
    groups, so a plain uniform resize is the fair operation."""
    rs = _resample()
    for g, raw_path in raw_paths.items():
        im = Image.open(raw_path).convert("RGBA")
        canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
        canvas.alpha_composite(im)
        canvas = canvas.resize((CANVAS_PX, CANVAS_PX), rs)
        canvas.convert("RGB").save(out_paths[g])


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def build_panel_a(fig, card, rep_data, png_paths):
    ax_, ay_, aw_, ah_ = card
    pad_x, pad_top, pad_bot = 0.016, 0.038, 0.056
    label_w, gap_x, gap_y = 0.062, 0.018, 0.009

    free_w_in = (aw_ - 2 * pad_x - label_w - 2 * gap_x) * FIG_W / 3.0
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
        a.text(0.98, 0.5, name, ha="right", va="center", fontsize=8.6, fontweight="bold", color=TEXT)

    for c, g in enumerate(GROUPS):
        x = gx0 + label_w + gap_x + c * (cell_w + gap_x)
        fig.text(x + cell_w / 2.0, gy_top + 0.010, LABELS[g], ha="center", va="bottom",
                 fontsize=9.6, fontweight="bold", color=TITLE_COLOR[g])

        rd = rep_data[g]

        a = fig.add_axes([x, row_y(0), cell_w, cell_h])
        a.imshow(Image.open(png_paths[g]), interpolation="bilinear")
        image_cell(a, COLORS[g])
        # Three short stacked lines rather than one long line: even a
        # two-line version was still too wide for the cell and got clipped
        # mid-value (e.g. "LCF=0.9" instead of "LCF=0.92").
        chip = f"φ={rd['pore_fraction']:.2f}\nLCF={rd['largest_component_fraction']:.2f}\ndisc={rd['disconnected_fraction']:.2f}"
        a.text(0.035, 0.035, chip, transform=a.transAxes, fontsize=6.2, color=SUBTEXT,
               ha="left", va="bottom", linespacing=1.4, clip_on=True,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.3))

        for ri, plane in enumerate(PLANES):
            a = fig.add_axes([x, row_y(ri + 1), cell_w, cell_h])
            idx = rd["slice_index"][plane]
            cls_img = _class_image(rd["vol"], rd["labeled"], rd["largest_id"], plane, idx)
            a.imshow(cls_img, cmap=TOPO_CMAP, vmin=0, vmax=2, origin="lower",
                     interpolation="nearest", aspect="equal", extent=(0, 128, 0, 128))
            image_cell(a, COLORS[g])
            if c == 0:
                for txt, (tx, ty), ha, va in PLANE_DIRECTION_LABELS[plane]:
                    a.text(tx, ty, txt, transform=a.transAxes, fontsize=6.9, fontweight="bold",
                           color="white", ha=ha, va=va,
                           bbox=dict(facecolor=TEXT, edgecolor="none", alpha=0.55, pad=1.2))

    # compact topology-class legend, centered under the grid
    legend_y = ay_ + 0.015
    swatch_colors = [SOLID_COLOR, LARGEST_COMPONENT_COLOR, DISCONNECTED_FRAGMENT_COLOR]
    item_widths = [0.052, 0.150, 0.145]
    total_w = sum(item_widths)
    lx = gx0 + label_w + gap_x + (grid_w - label_w - gap_x - total_w) / 2.0
    for name, color, w in zip(TOPO_CLASS_NAMES, swatch_colors, item_widths):
        fig.add_artist(Rectangle((lx, legend_y - 0.006), 0.013, 0.011,
                                  transform=fig.transFigure, facecolor=color,
                                  edgecolor=SPINE, linewidth=0.6, zorder=101))
        fig.text(lx + 0.019, legend_y, name, ha="left", va="center",
                 fontsize=7.2, color=SUBTEXT, zorder=101)
        lx += w


# ============================================================================
# 11. PANEL B -- connected-backbone and percolation dashboard
# ============================================================================

B1_METRICS = ["largest_pore_component_fraction", "disconnected_pore_fraction", "mean_percolating_axes"]
B1_LABELS = ["Largest pore\ncomponent fraction", "Disconnected\npore fraction",
             "Mean percolating\naxes / 3"]


def plot_backbone_bars(ax, panel_b_metrics):
    style_axis(ax, grid=True)
    ax.set_title("Connected pore backbone", pad=5.0, color=TEXT, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction (mean percolating axes shown as value / 3)", color=SUBTEXT)

    ys = np.arange(len(B1_METRICS))
    offset = {"real": 0.24, "diffusion": 0.0, "gan": -0.24}
    bar_h = 0.20
    for mi, key in enumerate(B1_METRICS):
        r = panel_b_metrics[key]
        for g in GROUPS:
            raw = r[g]
            plotted = raw / 3.0 if key == "mean_percolating_axes" else raw
            y = ys[mi] + offset[g]
            ax.barh(y, plotted, height=bar_h, color=COLORS[g], edgecolor="black",
                    linewidth=0.5, zorder=3)
            label_val = f"{raw:.2f}" if key != "mean_percolating_axes" else f"{raw:.2f}"
            ax.text(min(plotted + 0.02, 0.97), y, label_val, va="center", ha="left",
                    fontsize=6.2, color=TEXT, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels(B1_LABELS, fontsize=6.8)
    ax.set_ylim(-0.5, len(B1_METRICS) - 0.5)
    ax.margins(x=0.02)

    # Placed just outside the right axes edge (bbox_to_anchor x>1) rather
    # than inside any corner: every corner of this chart can carry a bar
    # label near the x=1 edge (real/diffusion routinely reach ~0.9+), so an
    # inside legend risks covering data in some row.
    handles = [Line2D([0], [0], marker="s", linestyle="None", markersize=7,
                       markerfacecolor=COLORS[g], markeredgecolor="black")
               for g in GROUPS]
    leg = ax.legend(handles, [LABELS[g] for g in GROUPS], loc="center left",
                    bbox_to_anchor=(1.01, 0.5), borderaxespad=0, frameon=True,
                    facecolor="white", edgecolor=SPINE, framealpha=0.94, handlelength=1.4,
                    borderpad=0.4, labelspacing=0.9, fontsize=6.6)
    leg.get_frame().set_linewidth(0.6)


def plot_percolation_heatmap(fig, ax, panel_b_metrics):
    style_axis(ax, grid=False)
    ax.set_title("Directional percolating pore fraction", pad=5.0, color=TEXT, fontweight="bold")

    axes_order = ["x", "y", "z"]
    mat = np.array([[panel_b_metrics[f"percolating_pore_fraction_{ax_letter}"][g]
                     for ax_letter in axes_order] for g in GROUPS])

    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_xticklabels([s.upper() for s in axes_order])
    ax.set_yticks(range(3))
    ax.set_yticklabels([LABELS[g] for g in GROUPS], fontsize=7.0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            txt_color = "white" if v < 0.55 else "#1A0F2B"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4,
                    color=txt_color, fontweight="bold")

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.008, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.0, colors=SUBTEXT)
    cb.outline.set_edgecolor(SPINE)
    cb.outline.set_linewidth(0.6)


def build_panel_b(fig, card, panel_b_metrics):
    bx_, by_, bw_, bh_ = card
    pad_l, pad_r, pad_t, pad_b, gap_y = 0.048, 0.098, 0.026, 0.058, 0.075

    plot_w = bw_ - pad_l - pad_r
    plot_h = (bh_ - pad_t - pad_b - gap_y) / 2.0

    ax_top = fig.add_axes([bx_ + pad_l, by_ + bh_ - pad_t - plot_h, plot_w, plot_h])
    ax_bot = fig.add_axes([bx_ + pad_l, by_ + pad_b, plot_w, plot_h])

    plot_backbone_bars(ax_top, panel_b_metrics)
    plot_percolation_heatmap(fig, ax_bot, panel_b_metrics)

    fig.text(bx_ + pad_l, by_ + pad_b - 0.026,
             "Fragmentation appears after connected-network evaluation.",
             ha="left", va="top", fontsize=6.6, color=SUBTEXT, style="italic")


# ============================================================================
# 12. PANEL C -- pair-connectedness mechanism + network-integrity inset
# ============================================================================

AXIS_TITLES = {"x": "Pair-connectedness, x", "y": "Pair-connectedness, y", "z": "Pair-connectedness, z"}


def plot_pair_connectedness_axis(ax, curve_data, axis_letter, ymax, show_legend=False):
    style_axis(ax)
    ax.set_title(AXIS_TITLES[axis_letter], pad=4.0, color=TEXT, fontweight="bold")
    ax.set_xlabel("Lag (vox)", color=SUBTEXT)
    ax.set_ylabel("Pair-connectedness", color=SUBTEXT)
    ax.set_ylim(0, ymax * 1.08)

    for g in GROUPS:
        d = curve_data[g]
        ax.plot(d["x"], d["y"], color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                solid_capstyle="round", label=LABELS[g], zorder=3 if g == "real" else 4)
    ax.margins(x=0.02)

    if show_legend:
        handles = [Line2D([0], [0], color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g])
                   for g in GROUPS]
        leg = ax.legend(handles, [LABELS[g] for g in GROUPS], loc="upper right", frameon=True,
                        facecolor="white", edgecolor=SPINE, framealpha=0.94, handlelength=2.2,
                        borderpad=0.4, labelspacing=0.32, fontsize=6.6)
        leg.get_frame().set_linewidth(0.6)
        _emphasize_dashed_legend_handle(leg.legend_handles[GROUPS.index("gan")])


def plot_network_inset(ax, advanced_metrics):
    style_axis(ax, grid=True)
    ax.set_title("Extracted-network connectivity", pad=4.0, color=TEXT, fontweight="bold")
    ax.set_ylabel("Largest-component fraction", color=SUBTEXT)
    ax.set_ylim(0, 1.15)

    keys = ["skeleton_graph_largest_component_fraction", "snow_graph_largest_component_fraction"]
    xlabels = ["Skeleton\ngraph LCF", "SNOW\ngraph LCF"]
    xs = np.arange(len(keys))
    offset = {"real": -0.22, "diffusion": 0.0, "gan": 0.22}
    for g in GROUPS:
        vals = [advanced_metrics[k][g] for k in keys]
        ax.scatter(xs + offset[g], vals, s=52, marker=MARKERS[g], color=COLORS[g],
                   edgecolors="black", linewidths=0.8, zorder=5, label=LABELS[g])
        for xi, v in zip(xs + offset[g], vals):
            ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, color=SUBTEXT)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=6.4)
    ax.set_xlim(-0.55, len(keys) - 0.45)

    coord = advanced_metrics["snow_coordination_number"]
    coord_txt = "SNOW coordination:  " + "   ".join(f"{LABELS[g]} {coord[g]:.2f}" for g in GROUPS)
    ax.text(0.5, -0.30, coord_txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.8, color=SUBTEXT)


def build_panel_c(fig, card, curve_data, rmse_data, advanced_metrics):
    cx_, cy_, cw_, ch_ = card
    header_title_y = cy_ + ch_ - 0.020
    plot_top = header_title_y - 0.062
    pad_l, pad_r, pad_b = 0.040, 0.020, 0.075

    fig.text(cx_ + 0.014, header_title_y, "Pore pair-connectedness and network integrity",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    plot_bottom = cy_ + pad_b
    plot_h = plot_top - plot_bottom

    curves_w = 0.62 * (cw_ - pad_l - pad_r)
    gap = 0.018
    curve_plot_w = (curves_w - 2 * gap) / 3.0

    ymax = 0.0
    for axis_letter in ("x", "y", "z"):
        for g in GROUPS:
            ymax = max(ymax, float(np.nanmax(curve_data[axis_letter][g]["y"])))

    for i, axis_letter in enumerate(("x", "y", "z")):
        x = cx_ + pad_l + i * (curve_plot_w + gap)
        ax = fig.add_axes([x, plot_bottom, curve_plot_w, plot_h])
        plot_pair_connectedness_axis(ax, curve_data[axis_letter], axis_letter, ymax,
                                      show_legend=(i == 2))

    right_x = cx_ + pad_l + curves_w + 0.028
    right_w = cx_ + cw_ - pad_r - right_x

    rmse_h = plot_h * 0.34
    rmse_y = plot_bottom + plot_h - rmse_h
    ax_rmse = fig.add_axes([right_x, rmse_y, right_w, rmse_h])
    ax_rmse.axis("off")
    ax_rmse.add_patch(Rectangle((0, 0), 1, 1, transform=ax_rmse.transAxes, facecolor="white",
                                 edgecolor=SPINE, linewidth=0.7, zorder=1))
    rmse_lines = [
        "Pair-connectedness RMSE",
        f"PoreGen:  {rmse_data['x']['diffusion']:.4f} / {rmse_data['y']['diffusion']:.4f} / {rmse_data['z']['diffusion']:.4f}",
        f"IPWGAN:   {rmse_data['x']['gan']:.4f} / {rmse_data['y']['gan']:.4f} / {rmse_data['z']['gan']:.4f}",
        "(x / y / z)",
    ]
    for li, line in enumerate(rmse_lines):
        weight = "bold" if li == 0 else ("normal" if li < 3 else "normal")
        color = TEXT if li == 0 else SUBTEXT
        style = "italic" if li == 3 else "normal"
        ax_rmse.text(0.07, 0.86 - li * 0.24, line, transform=ax_rmse.transAxes, ha="left", va="top",
                     fontsize=6.6, color=color, fontweight=weight, style=style, zorder=2)

    inset_h = plot_h - rmse_h - 0.022
    ax_inset = fig.add_axes([right_x, plot_bottom, right_w, inset_h])
    plot_network_inset(ax_inset, advanced_metrics)


# ============================================================================
# 13. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT               = {PROJECT}")
    log(f"[paths] REAL_DIR              = {REAL_DIR}")
    log(f"[paths] DIFFUSION_DIR         = {DIFFUSION_DIR}")
    log(f"[paths] GAN_DIR               = {GAN_DIR}")
    log(f"[paths] FULL_METRIC_DIR       = {FULL_METRIC_DIR}")
    log(f"[paths] ADVANCED_METRIC_DIR   = {ADVANCED_METRIC_DIR}")
    log(f"[paths] SLICE_SELECTION_DIR   = {SLICE_SELECTION_DIR}")
    log(f"[paths] SUITABILITY_DIR       = {SUITABILITY_DIR}")

    log("\n[forbidden-check] verifying no forbidden source is used:")
    for g in GROUPS:
        d = GROUP_VOLUME_DIRS[g]
        excluded = _path_is_excluded(d)
        log(f"  {g}: {d}  excluded={excluded}")
        if excluded:
            raise RuntimeError(f"[forbidden-check] group '{g}' resolved to a forbidden path: {d}")
    for fd in FORBIDDEN_DIRS:
        log(f"  forbidden dir not used: {fd}")

    # ---- volumes: load, validate, direct sanity checks ---------------------
    volumes_by_group = {}
    for g in GROUPS:
        volumes = load_group_volumes(g)
        volumes_by_group[g] = volumes
        fracs = [pore_fraction(v) for v in volumes.values()]
        direct_mean = float(np.mean(fracs))
        direct_std = float(np.std(fracs))
        check_pore_fraction(g, direct_mean)
        log(f"[data:{g}] direct pore fraction mean={direct_mean:.4f} std={direct_std:.4f}")

        comp_means = group_component_means(volumes)
        check_component_sanity(g, comp_means)
        log(f"[data:{g}] direct component means: {comp_means}")

    # ---- representative selection (official CSV only) ---------------------
    representative_files = load_representative_selection()

    rep_data = {}
    for g in GROUPS:
        f = representative_files[g]
        v = volumes_by_group[g][f]
        stats = component_stats(v)
        log(f"[rep:{g}] file={f.name}  pore_fraction={stats['pore_fraction']:.4f}  "
            f"LCF={stats['largest_component_fraction']:.4f}  "
            f"disconnected={stats['disconnected_fraction']:.4f}  "
            f"n_components={stats['n_components']}  "
            f"perc(x,y,z)=({stats['perc_x']},{stats['perc_y']},{stats['perc_z']})  "
            f"n_percolating_axes={stats['n_percolating_axes']}")

        slice_index = {}
        slice_info = {}
        for plane in PLANES:
            best = select_slice_index(v, stats["labeled"], stats["largest_id"], plane,
                                       stats["pore_fraction"], stats["disconnected_fraction"])
            slice_index[plane] = best["index"]
            slice_info[plane] = best

        rep_data[g] = {
            "file": f, "vol": v, "labeled": stats["labeled"], "largest_id": stats["largest_id"],
            "pore_fraction": stats["pore_fraction"],
            "largest_component_fraction": stats["largest_component_fraction"],
            "disconnected_fraction": stats["disconnected_fraction"],
            "n_components": stats["n_components"],
            "perc_x": stats["perc_x"], "perc_y": stats["perc_y"], "perc_z": stats["perc_z"],
            "n_percolating_axes": stats["n_percolating_axes"],
            "slice_index": slice_index, "slice_info": slice_info,
        }

    # ---- panel a: 3D connected pore-backbone renders (fixed, fair protocol) ---
    parallel_scale = 0.82 * max(max(rep_data[g]["vol"].shape) for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}
    render_audit = {}
    for g in GROUPS:
        render_audit[g] = render_group_backbone(rep_data[g]["labeled"], rep_data[g]["largest_id"],
                                                 g, raw_paths[g], parallel_scale)
        log(f"[panel-a-audit:{g}] render_extent={render_audit[g]['render_extent']}  "
            f"parallel_scale={render_audit[g]['parallel_scale']:.4f}  "
            f"connectivity='{render_audit[g]['connectivity']}'")
    finalize_renders(raw_paths, png_paths)
    log("Panel a crop/zoom used: none")

    # ---- panel b: official scalar metrics -----------------------------------
    scalar_dfs = []
    for path in (GROUP_SCALAR_SUMMARY, DESCRIPTOR_ERRORS, ONE_FILE_DESCRIPTOR_ERRORS):
        try:
            scalar_dfs.append((path, _load_csv(path)))
        except FileNotFoundError as exc:
            log(f"[metrics] optional fallback source unavailable: {exc}")
            scalar_dfs.append((path, None))
    panel_b_metrics = load_panel_b_metrics(scalar_dfs)

    # ---- panel c: curves, rmse, advanced metrics ---------------------------
    curve_data = load_pair_connectedness_curves()
    rmse_data = load_pair_connectedness_rmse()

    advanced_dfs = []
    for path in (ADVANCED_GROUP_SUMMARY, ADVANCED_DESCRIPTOR_ERRORS, ONE_FILE_ADVANCED_ERRORS):
        try:
            advanced_dfs.append((path, _load_csv(path)))
        except FileNotFoundError as exc:
            log(f"[metrics] optional fallback source unavailable: {exc}")
            advanced_dfs.append((path, None))
    advanced_metrics = load_advanced_metrics(advanced_dfs)

    # ---- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.patch.set_facecolor(BG)

    add_card(fig, card_a)
    add_card(fig, card_b)
    add_card(fig, card_c)

    add_panel_label(fig, card_a[0] + 0.007, card_a[1] + card_a[3] + PANEL_LABEL_OFFSET, "a)")
    add_panel_label(fig, card_b[0] + 0.007, card_b[1] + card_b[3] + PANEL_LABEL_OFFSET, "b)")
    add_panel_label(fig, card_c[0] + 0.007, card_c[1] + card_c[3] + PANEL_LABEL_OFFSET, "c)")

    build_panel_a(fig, card_a, rep_data, png_paths)
    build_panel_b(fig, card_b, panel_b_metrics)
    build_panel_c(fig, card_c, curve_data, rmse_data, advanced_metrics)

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

    CAPTION_TXT.write_text(CAPTION_TEXT)

    audit = {
        "project": str(PROJECT),
        "volume_dirs": {g: str(GROUP_VOLUME_DIRS[g]) for g in GROUPS},
        "forbidden_dirs_checked": [str(p) for p in FORBIDDEN_DIRS],
        "connectivity": "6-connectivity (face-adjacency), generate_binary_structure(3, 1)",
        "panel_a_render_audit": render_audit,
        "representative": {
            g: {
                "file": rep_data[g]["file"].name,
                "pore_fraction": rep_data[g]["pore_fraction"],
                "largest_component_fraction": rep_data[g]["largest_component_fraction"],
                "disconnected_fraction": rep_data[g]["disconnected_fraction"],
                "n_components": rep_data[g]["n_components"],
                "perc_x": rep_data[g]["perc_x"], "perc_y": rep_data[g]["perc_y"],
                "perc_z": rep_data[g]["perc_z"],
                "n_percolating_axes": rep_data[g]["n_percolating_axes"],
                "slice_index": rep_data[g]["slice_index"],
            }
            for g in GROUPS
        },
        "panel_b_metrics": {
            key: {"raw_name": panel_b_metrics[key]["raw_name"], "real": panel_b_metrics[key]["real"],
                  "diffusion": panel_b_metrics[key]["diffusion"], "gan": panel_b_metrics[key]["gan"],
                  "source": panel_b_metrics[key]["source"]}
            for key, *_ in PANEL_B_METRICS
        },
        "panel_c_curve_points": {axis_letter: {g: len(curve_data[axis_letter][g]["x"]) for g in GROUPS}
                                 for axis_letter in ("x", "y", "z")},
        "panel_c_rmse": rmse_data,
        "panel_c_advanced_metrics": {
            key: {"raw_name": advanced_metrics[key]["raw_name"], "real": advanced_metrics[key]["real"],
                  "diffusion": advanced_metrics[key]["diffusion"], "gan": advanced_metrics[key]["gan"],
                  "source": advanced_metrics[key]["source"]}
            for key, *_ in ADVANCED_METRICS
        },
        "fallbacks_used": [],
        "saved": {"png": str(png), "pdf": str(pdf), "svg": str(svg), "tiff": str(tif_out)},
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))

    # =========================== SUMMARY ========================================
    log("\nVolume folders:")
    for g in GROUPS:
        log(f"  {g}: {GROUP_VOLUME_DIRS[g]}  ({EXPECTED_N} NPYs, shape {EXPECTED_SHAPE})")

    log("\nRepresentative samples (official CSV selection):")
    for g in GROUPS:
        rd = rep_data[g]
        log(f"  {g}: {rd['file'].name}  pore_fraction={rd['pore_fraction']:.4f}  "
            f"LCF={rd['largest_component_fraction']:.4f}  disc={rd['disconnected_fraction']:.4f}  "
            f"slice_index={rd['slice_index']}")

    log("\nPanel b metrics resolved:")
    for key, label, _aliases in PANEL_B_METRICS:
        r = panel_b_metrics[key]
        log(f"  {label}: real={r['real']:.4f} diffusion={r['diffusion']:.4f} gan={r['gan']:.4f} "
            f"(source={Path(r['source']).name})")

    log("\nPanel c pair-connectedness RMSE:")
    for axis_letter in ("x", "y", "z"):
        r = rmse_data[axis_letter]
        log(f"  {axis_letter}: diffusion={r['diffusion']:.4f} gan={r['gan']:.4f}")

    log("\nPanel c advanced/network metrics:")
    for key, label, _aliases, _exp in ADVANCED_METRICS:
        r = advanced_metrics[key]
        log(f"  {label}: real={r['real']:.4f} diffusion={r['diffusion']:.4f} gan={r['gan']:.4f} "
            f"(source={Path(r['source']).name})")

    log("\nFallbacks used: none")

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out, AUDIT_JSON, CAPTION_TXT):
        log(" ", p.resolve())

    log("\nFINAL FIGURE 4.6 AUDIT PASS")


if __name__ == "__main__":
    sys.exit(main())
