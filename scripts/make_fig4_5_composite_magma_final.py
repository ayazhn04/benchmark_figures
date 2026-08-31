#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.5 composite (magma, final)
Reference vs MicroGen3D vs SurVol -- multi-phase Muller graphite-silicon
anode microstructures (3-phase: pore / active material / CBD).

Same visual contract as Figures 4.1-4.4 (make_fig4_1_composite_magma_final.py
.. make_fig4_4_composite_magma_final.py): same cards, fonts, panel-label
style, export block. No tight_layout / constrained_layout / bbox_inches.

Only the exact official final B_late volume folders and metric CSV/JSON
files are used -- no broad path search, no bestckpt MicroGen3D, no
non-B_late SurVol, no old grayscale/calibrated GAN outputs. Extensive hard
sanity checks raise and refuse to save rather than silently substituting
or faking data.

Panel a uses a FIXED camera/extent/cutaway protocol identical across all
three groups (no content-dependent cropping, no alpha-bounding-box zoom),
fixed central slices (z=y=x=64), and grayscale phase coloring. Panel b is
a 2x2 grid of official structural descriptor curves. Panel c is a compact
two-part mechanistic panel: an interface/contact-hierarchy fingerprint and
an active-domain continuity/fragmentation plot. No interpretive sentences
are drawn inside the figure.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG
# ============================================================================

# Exact project root -- hardcoded rather than auto-detected, per task
# instructions. Never searched broadly for an alternative Section 4.5
# project or dataset.
PROJECT = Path("/home/ra2/4_5_microgen3d_muller128cube")

REAL_DIR = PROJECT / "outputs" / "evaluation" / "microgen3d_muller128cube_40h_v1" / "real_3phase"
DIFFUSION_DIR = PROJECT / "outputs" / "evaluation" / "microgen3d_muller128cube_40h_v1" / "generated_3phase"
GAN_DIR = PROJECT / "outputs" / "evaluation" / "survol_nphase_direct128_B_late" / "gan_3phase"

FULL_METRIC_DIR = PROJECT / "outputs" / "metrics" / "section_4_5_full_multiphase_metrics_survol_B_late"
SCALAR_COMPARISON = FULL_METRIC_DIR / "section_4_5_scalar_comparison_master.csv"
CURVE_COMPARISON = FULL_METRIC_DIR / "section_4_5_curve_comparison_master.csv"
CURVE_PROFILES = FULL_METRIC_DIR / "section_4_5_curve_profiles_long.csv"
PER_SAMPLE_SCALARS = FULL_METRIC_DIR / "section_4_5_per_sample_scalar_descriptors.csv"
ALL_METRIC_COMPARISON = FULL_METRIC_DIR / "section_4_5_all_metric_comparison_master.csv"

CLEANED_DIR = PROJECT / "outputs" / "metrics" / "section_4_5_cleaned_master_survol_B_late"
MAIN_PAPER_CANDIDATES = CLEANED_DIR / "section_4_5_main_paper_candidate_metrics.csv"
CLEANED_SCIENTIFIC_MASTER = CLEANED_DIR / "section_4_5_cleaned_scientific_metric_master.csv"

ACTIVE_COMPONENT_DIR = PROJECT / "outputs" / "metrics" / "section_4_5_active_component_diagnostic"
ACTIVE_COMPONENT_REPORT = ACTIVE_COMPONENT_DIR / "active_component_sizes_report.json"
ACTIVE_COMPONENT_BY_SAMPLE = ACTIVE_COMPONENT_DIR / "active_component_sizes_by_sample.csv"
ACTIVE_COMPONENT_SUMMARY = ACTIVE_COMPONENT_DIR / "active_component_sizes_summary.csv"

OUT = PROJECT / "paper_figures_4_5" / "figure_4_5_composite_magma_final"
TMP = OUT / "_render_cache"
LOGS = PROJECT / "outputs" / "logs"
STEM = "fig4_5_composite_magma_final"
AUDIT_JSON = OUT / "figure_4_5_audit.json"
CAPTION_TXT = OUT / "figure_4_5_caption.txt"

GROUPS = ["real", "diffusion", "gan"]
GROUP_VOLUME_DIRS = {"real": REAL_DIR, "diffusion": DIFFUSION_DIR, "gan": GAN_DIR}

EXPECTED_SHAPE = (128, 128, 128)
EXPECTED_N = 50
VALID_LABELS = {0, 1, 2}

# Defense-in-depth: even though every path above is hardcoded exactly per
# task instructions (no broad search is ever performed), this guards
# against an accidental future edit resolving into one of the explicitly
# forbidden cohorts/folders.
EXCLUDE_PATH_TOKENS = ("bestckpt", "ganmuller128cube50", "ganmuller128cube50tif", "smoke",
                       "preview", "checkpointcompare", "inferencesmoke", "fixeddecode",
                       "newhrqc", "inputaudit", "calibrated", "grayscale")

RENDER_PX = 1500
CANVAS_PX = 900

FIG_W, FIG_H = 17.8, 10.2

CAPTION_TEXT = (
    "Figure 4.5. Qualitative and descriptor-level comparison for the multi-phase "
    "microstructure task. (a) Representative fixed-index orthogonal central slices "
    "(z = y = x = 64) and 3D cutaway renderings, using the full 128³ field of view "
    "with an identical camera, cube extent, and octant cutaway across all three groups, "
    "for real Muller graphite-silicon anode samples, adapted MicroGen3D outputs, and "
    "n-phase SurVol outputs. Labels correspond to pore space, active material "
    "(graphite + silicon), and carbon-binder domain (CBD); pore is not rendered as a "
    "3D surface. Representative samples were selected deterministically from "
    "phase-fraction statistics, not manual visual cherry-picking. (b) Official "
    "structural descriptor curves -- active same-phase and pore-active cross two-point "
    "correlation, an active chord-length distribution, and one further resolved curve "
    "descriptor -- each showing the real reference (with spread band where available) "
    "and the two generated ensembles overlaid. (c) Mechanistic descriptors: an "
    "interface/contact-hierarchy fingerprint (pairwise phase interfaces, the "
    "triple-phase-contact proxy, and CBD contact-role descriptors) and active-domain "
    "continuity/fragmentation (active Euler characteristic and active "
    "connected-component statistics, log scale). The figure reports raw generated "
    "outputs without morphology-based post-processing."
)

# ============================================================================
# 1b. SANITY-CHECK CONSTANTS (verified reference values; used only to
# cross-check independently loaded/computed data -- never as the plotted
# source)
# ============================================================================

PHASE_FRACTION_TOL = 0.03
TRANSITION_REL_TOL = 0.25
TRANSITION_ABS_FLOOR = 0.003
METRIC_ERR_REL_TOL = 0.40
METRIC_ERR_ABS_FLOOR = 5.0  # percentage points
COMPONENT_REL_TOL = 0.40
COMPONENT_ABS_FLOOR = 0.5

EXPECTED_PHASE_FRACTIONS = {
    "real":      {"pore": 0.4799, "active": 0.4325, "cbd": 0.0876},
    "diffusion": {"pore": 0.4648, "active": 0.4710, "cbd": 0.0642},
    "gan":       {"pore": 0.5378, "active": 0.3872, "cbd": 0.0750},
}

EXPECTED_TRANSITIONS = {
    "real":      {"x": 0.0524, "y": 0.0543, "z": 0.0667},
    "diffusion": {"x": 0.0668, "y": 0.0699, "z": 0.0846},
    "gan":       {"x": 0.0364, "y": 0.0322, "z": 0.0448},
}

# Panel-b/c scalar metric keys, mapped to (concise label, strict alias
# list, expected MicroGen3D relative-error %, expected SurVol relative-error %).
# These feed BOTH panel c-left (raw real/diffusion/gan values) and the
# sanity checks; panel b itself now plots official curves, not these errors.
PANEL_B_METRICS = [
    ("cbd_fraction", "CBD fraction",
     ["frac_cbd", "fraction_cbd", "phase_fraction_2", "volume_fraction_2",
      "frac_phase2", "phase2_fraction", "cbd_fraction"], 26.75, 14.35),
    ("interface_pore_active", "Pore–active",
     ["interface_0_1", "interface_pore_active", "pore_active_interface_density"], 64.98, 31.98),
    ("interface_pore_cbd", "Pore–CBD",
     ["interface_0_2", "interface_pore_cbd", "pore_cbd_interface_density"], 76.38, 36.55),
    ("interface_active_cbd", "Active–CBD",
     ["interface_1_2", "interface_active_cbd", "active_cbd_interface_density"], 362.76, 48.02),
    # Must resolve exactly to the density variant -- the official CSV also
    # carries tpb_voxel_count_2x2x2 and tpb_fraction_2x2x2 for the same
    # 2x2x2 neighborhood, so the bare "tpb"/"tpb_proxy"/"triple_phase_contact"
    # aliases are deliberately NOT included here: they are substrings of all
    # three column names and would make this metric ambiguous.
    ("tpb_proxy", "TPB proxy",
     ["tpb_density_2x2x2", "tpb_proxy_density", "triple_phase_contact_proxy_density"], 479.01, 50.78),
    ("active_to_pore", "Active→pore",
     ["active_surface_exposed_to_pore_fraction", "active_surface_exposed_to_pore",
      "active_contact_pore", "active_surface_contacting_pore", "muller_active_surface_pore"],
     12.87, 3.59),
    ("active_to_cbd", "Active→CBD",
     ["active_surface_contacting_cbd_fraction", "active_surface_contacting_cbd",
      "active_contact_cbd", "muller_active_surface_cbd"], 111.60, 31.15),
    ("cbd_to_pore", "CBD→pore",
     ["cbd_contacting_pore_fraction", "cbd_surface_contacting_pore_fraction",
      "cbd_surface_contacting_pore", "cbd_contact_pore", "muller_cbd_surface_pore"], 71.31, 0.81),
    ("cbd_to_active", "CBD→active",
     ["cbd_contacting_active_fraction", "cbd_contacting_active_material_fraction",
      "cbd_surface_contacting_active_fraction", "cbd_surface_contacting_active",
      "cbd_surface_contacting_active_material", "cbd_contact_active",
      "muller_cbd_surface_active"], 552.89, 6.31),
]

ACTIVE_EULER_ALIASES = ["euler_active_abs", "abs_euler_active", "active_euler_abs",
                        "active_phase_euler", "euler_number_active", "active_euler_number",
                        "euler_abs_active"]

EXPECTED_COMPONENTS = {
    "real":      {"components": 88.66, "singleton": 30.10, "size_2_10": 25.00, "small_frac": 0.000194699},
    "diffusion": {"components": 7002.06, "singleton": 4424.90, "size_2_10": 2405.32, "small_frac": 0.012598196},
    "gan":       {"components": 110.82, "singleton": 55.16, "size_2_10": 42.54, "small_frac": 0.000259086},
}

# ============================================================================
# 2. STYLE -- identical contract to Figures 4.1-4.4
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
LABELS = {"real": "Reference", "diffusion": "MicroGen3D", "gan": "SurVol"}
MARKERS = {"real": "o", "diffusion": "D", "gan": "^"}
TITLE_COLOR = {"real": TEXT, "diffusion": COLORS["diffusion"], "gan": COLORS["gan"]}
LINESTYLES = {"real": "-", "diffusion": "-", "gan": (0, (5.5, 2.2))}
LINEWIDTHS = {"real": 1.85, "diffusion": 1.85, "gan": 1.95}

CARD_LW = 0.8
CELL_LW = 1.6

# Categorical phase colors for panel a -- these color the material phase
# inside each slice AND the 3D cutaway render (the same three flat colors
# in both rows, so the reader immediately reads them as the same three
# phases), not the model group (model identity is carried by the column
# titles and cell borders instead, as in Figures 4.1-4.4). Flat categorical
# colors, not a decorative gradient: pore = deep purple/near-black,
# active = orange/copper, CBD = pale/bright yellow.
PHASE_COLORS = {0: "#1A0B2E", 1: "#E0792A", 2: "#F5E27A"}
PHASE_NAMES = {0: "pore", 1: "active", 2: "CBD"}
PHASE_CMAP = ListedColormap([PHASE_COLORS[0], PHASE_COLORS[1], PHASE_COLORS[2]])

# ============================================================================
# 3. GEOMETRY (figure fractions)
#
# Panel a (3 groups x {3D, XY, XZ, YZ}) is the same shape as the panel-a
# image grid in Figures 4.1/4.2/4.4, so the proven card_a geometry is
# reused directly. Panel b is now a 2x2 curve grid, the same shape as
# panel b in Figures 4.1/4.2, so that proven geometry is reused too. Panel
# c is a full-width card split into two side-by-side subplots, the same
# shape as panel c in Figure 4.2, reused for the same reason.
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


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _path_is_excluded(p: Path) -> bool:
    s = _norm(p.name)
    return any(tok in s for tok in EXCLUDE_PATH_TOKENS)


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


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


# ============================================================================
# 5. CARD / LABEL / AXIS PRIMITIVES -- reused verbatim from Figures 4.1-4.4
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
# 6. VOLUME LOADING + VALIDATION (exact official B_late folders only)
# ============================================================================


def load_group_volumes(group: str) -> dict:
    d = GROUP_VOLUME_DIRS[group]
    if not d.exists():
        raise FileNotFoundError(
            f"[data:{group}] expected exact official folder not found: {d}\n"
            f"This script never searches broadly for alternative Section 4.5 sample folders."
        )
    if _path_is_excluded(d):
        raise RuntimeError(f"[data:{group}] resolved folder {d} matches a forbidden pattern "
                            f"({EXCLUDE_PATH_TOKENS}) -- refusing to use it")

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
        volumes[f] = arr.astype(np.uint8)

    log(f"[data:{group}] folder = {d}")
    log(f"[data:{group}] n_volumes = {len(files)}  shape={EXPECTED_SHAPE}  "
        f"labels subset of {{0,1,2}} (validated, finite)")
    return volumes


# Physical axis mapping for v(z,y,x): x = numpy axis 2, y = numpy axis 1,
# z = numpy axis 0. Never changed.


def transition_rates(v: np.ndarray):
    tx = float(np.mean(v[:, :, 1:] != v[:, :, :-1]))
    ty = float(np.mean(v[:, 1:, :] != v[:, :-1, :]))
    tz = float(np.mean(v[1:, :, :] != v[:-1, :, :]))
    return tx, ty, tz


def phase_fractions(v: np.ndarray) -> dict:
    n = v.size
    return {"pore": float(np.sum(v == 0)) / n, "active": float(np.sum(v == 1)) / n,
            "cbd": float(np.sum(v == 2)) / n}


def compute_group_descriptors(volumes: dict) -> list:
    rows = []
    for f, v in volumes.items():
        fr = phase_fractions(v)
        tx, ty, tz = transition_rates(v)
        rows.append({"file": f, **fr, "tx": tx, "ty": ty, "tz": tz})
    return rows


def check_phase_fractions(group: str, direct_mean: dict):
    exp = EXPECTED_PHASE_FRACTIONS[group]
    bad = [(k, direct_mean[k], exp[k]) for k in ("pore", "active", "cbd")
           if abs(direct_mean[k] - exp[k]) > PHASE_FRACTION_TOL]
    if bad:
        detail = ", ".join(f"{k}: got {g:.4f} vs expected ~{e:.4f}" for k, g, e in bad)
        raise RuntimeError(f"[sanity:phase_fraction] group '{group}' deviates from expected "
                            f"phase fractions by more than {PHASE_FRACTION_TOL} absolute ({detail})")
    log(f"[sanity:phase_fraction] group '{group}' OK: pore={direct_mean['pore']:.4f} "
        f"active={direct_mean['active']:.4f} cbd={direct_mean['cbd']:.4f} "
        f"(manuscript reference ~{exp})")


def check_transitions(group: str, direct_txyz):
    exp = EXPECTED_TRANSITIONS[group]
    for axis, val in zip("xyz", direct_txyz):
        e = exp[axis]
        tol = max(TRANSITION_REL_TOL * abs(e), TRANSITION_ABS_FLOOR)
        if abs(val - e) > tol:
            raise RuntimeError(f"[sanity:transitions] group '{group}' axis {axis}: direct={val:.6f} "
                                f"vs expected~{e:.6f} (tol {tol:.6f}) -- axis mapping or data source "
                                f"appears wrong, stopping without saving")
    log(f"[sanity:transitions] group '{group}' OK: direct=(tx={direct_txyz[0]:.6f}, "
        f"ty={direct_txyz[1]:.6f}, tz={direct_txyz[2]:.6f})  manuscript reference ~{exp}")


# ============================================================================
# 7. REPRESENTATIVE-SAMPLE SELECTION (independent per group, no pairing
# assumption, deterministic)
# ============================================================================


def _phase_vec(row):
    return np.array([row["pore"], row["active"], row["cbd"]], dtype=float)


def choose_representative_real(rows: list):
    files = [r["file"] for r in rows]
    mean_vec = np.mean([_phase_vec(r) for r in rows], axis=0)
    d = np.array([np.sum(np.abs(_phase_vec(r) - mean_vec)) for r in rows])
    dmin = float(d.min())
    tol = 1e-9 * max(1.0, abs(dmin))
    tied = sorted((i for i in range(len(rows)) if d[i] <= dmin + tol), key=lambda i: files[i].name)
    k = tied[0]
    log(f"[rep:real] selected {rows[k]['file'].name}  pore={rows[k]['pore']:.6f} "
        f"active={rows[k]['active']:.6f} cbd={rows[k]['cbd']:.6f}  tx={rows[k]['tx']:.6f} "
        f"ty={rows[k]['ty']:.6f} tz={rows[k]['tz']:.6f}  L1-distance-to-group-mean={d[k]:.6f}"
        + ("  (tie-break: lexicographically first filename)" if len(tied) > 1 else ""))
    return rows[k], mean_vec


def choose_representative_generated(group: str, rows: list, real_mean_vec: np.ndarray):
    """Median-L1-error sample relative to the Reference group mean phase-fraction
    vector -- not the best-matching sample, per task instructions."""
    files = [r["file"] for r in rows]
    err = np.array([np.sum(np.abs(_phase_vec(r) - real_mean_vec)) for r in rows])
    order = np.argsort(err, kind="stable")
    median_rank = (len(rows) - 1) // 2
    median_err = err[order[median_rank]]
    tol = 1e-9 * max(1.0, abs(median_err))
    tied = sorted((i for i in range(len(rows)) if abs(err[i] - median_err) <= tol),
                  key=lambda i: files[i].name)
    k = tied[0] if tied else int(order[median_rank])
    r = rows[k]
    log(f"[rep:{group}] selected {r['file'].name}  pore={r['pore']:.6f} active={r['active']:.6f} "
        f"cbd={r['cbd']:.6f}  tx={r['tx']:.6f} ty={r['ty']:.6f} tz={r['tz']:.6f}  "
        f"L1-error-vs-real-mean={err[k]:.6f}  (median-error sample, rank "
        f"{median_rank + 1}/{len(rows)})")
    return r


# ============================================================================
# 8. SLICE SELECTION -- fixed central slices for all groups, for a fair,
# content-independent visual comparison (no phase-bounding-box search).
# ============================================================================

CENTRAL_SLICE_INDEX = 64  # z = y = x = 64, for a (128,128,128) volume


# ============================================================================
# 9. 3D RENDERING (panel a) -- a solid, opaque, categorically-colored
# cutaway cube (all three phases as flat-colored voxel cells), restricted to
# a fixed geometric octant-cutaway mask applied IDENTICALLY to all three
# groups -- the exact rendering technique used by Figure 4.2 panel a
# (render_cutaway_cube_pyvista), reused here so the two figures read as the
# same family: a solid per-voxel-phase cube read through pv.ImageData cell
# data and a threshold, not translucent marching-cubes isosurfaces (which
# read as a "soft cloud" rather than a clean cube). The cube is downsampled
# by majority vote for the render ONLY (a resolution reduction for display
# performance, not a content-dependent crop -- the render_extent audited
# below, the wireframe box, and the camera all still span the FULL
# [0,128]x[0,128]x[0,128] volume). No MIP fallback: if PyVista cannot run,
# the script fails loudly rather than silently substituting another
# representation.
# ============================================================================

CUBE_DOWNSAMPLE = 4  # 128^3 -> 32^3 for the 3D render only (same 32^3 render
                      # resolution as Figure 4.2's 64^3 -> 32^3); all metrics,
                      # sanity checks, and the panel-a 2D slices elsewhere in
                      # this script use the full-resolution 128^3 volume.


def _octant_keep_mask_zyx(nz: int, ny: int, nx: int) -> np.ndarray:
    """True everywhere except the removed corner octant, in the same (z,y,x)
    index order as the volumes themselves. At this figure's fixed camera
    direction (1.0, -1.30, 0.90) in (z,y,x)-space, the vertex nearest the
    viewer is (high-z, low-y, high-x) -- so that corner is removed here,
    putting the cutaway facing the viewer (same convention as Figure 4.2)."""
    keep = np.ones((nz, ny, nx), dtype=bool)
    keep[nz // 2:, :ny // 2, nx // 2:] = False
    return keep


def _downsample_labels_mode(vol: np.ndarray, factor: int) -> np.ndarray:
    """Block-mode (majority-vote) downsample that only ever emits values
    already present in the volume -- i.e. real voxel labels, not
    interpolated/fake ones. Same technique as Figure 4.2's cutaway-cube
    renderer."""
    nz, ny, nx = vol.shape
    nz2, ny2, nx2 = (nz // factor) * factor, (ny // factor) * factor, (nx // factor) * factor
    v = vol[:nz2, :ny2, :nx2]
    b = v.reshape(nz2 // factor, factor, ny2 // factor, factor, nx2 // factor, factor)
    b = b.transpose(0, 2, 4, 1, 3, 5).reshape(nz2 // factor, ny2 // factor, nx2 // factor, -1)
    counts = np.stack([(b == k).sum(-1) for k in (0, 1, 2)], axis=-1)
    return np.argmax(counts, axis=-1).astype(np.uint8)


def render_multiphase(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float) -> dict:
    import pyvista as pv

    try:
        pv.start_xvfb(wait=0.2)
    except Exception:
        pass

    nz, ny, nx = vol.shape  # full (128,128,128) -- the audited render extent
    center = np.array([nz / 2.0, ny / 2.0, nx / 2.0])

    ds = _downsample_labels_mode(vol, CUBE_DOWNSAMPLE)
    dz, dy, dx = ds.shape
    spacing = float(CUBE_DOWNSAMPLE)  # each downsampled cell spans CUBE_DOWNSAMPLE
                                       # original voxels, so the grid's physical
                                       # extent is still the full [0,nz]x[0,ny]x[0,nx].

    grid = pv.ImageData()
    grid.dimensions = np.array([dx, dy, dz]) + 1
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (spacing, spacing, spacing)
    grid.cell_data["phase"] = np.transpose(ds, (2, 1, 0)).flatten(order="F")
    keep_zyx = _octant_keep_mask_zyx(dz, dy, dx)
    grid.cell_data["keep"] = np.transpose(keep_zyx, (2, 1, 0)).flatten(order="F").astype(np.uint8)

    sub = grid.threshold(0.5, scalars="keep")

    pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX))
    pl.set_background("white")
    # Solid, opaque, flat-colored voxel cells (cell data, not point data --
    # no interpolation/smoothing across phase boundaries) for all three
    # phases at once, exactly like Figure 4.2's cutaway-cube render; pore is
    # included here (unlike the old isosurface version) because a solid cube
    # needs every phase filled in to read as a cube rather than a hollow shell.
    pl.add_mesh(sub, scalars="phase", cmap=PHASE_CMAP, clim=[0, 2], show_scalar_bar=False)

    # Full [0,nz]x[0,ny]x[0,nx] cube wireframe -- identical extent for every
    # group regardless of data content, so the rendered scale is fair by
    # construction rather than by post-hoc cropping.
    pl.add_mesh(pv.Box(bounds=(0, nz, 0, ny, 0, nx)), style="wireframe",
                color=COLORS[group], line_width=2.4, opacity=0.65)

    # identical camera + identical parallel scale across groups => comparable
    # size, no cropping; same convention as Figures 4.1/4.2/4.4.
    pl.enable_parallel_projection()
    direction = np.array([1.0, -1.30, 0.90])
    direction /= np.linalg.norm(direction)
    pl.camera.focal_point = tuple(center)
    pl.camera.position = tuple(center + direction * 4.0 * max(vol.shape))
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.parallel_scale = float(parallel_scale)

    pl.screenshot(str(out_raw), transparent_background=True)
    pl.close()

    return {"render_extent": [[0, nz], [0, ny], [0, nx]], "parallel_scale": float(parallel_scale),
            "cutaway_rule": "remove octant z>=nz/2, y<ny/2, x>=nx/2 (nearest the fixed camera)",
            "camera_direction": direction.tolist(),
            "render_downsample_factor": CUBE_DOWNSAMPLE}


def render_group_multiphase(vol: np.ndarray, group: str, out_raw: Path, parallel_scale: float) -> dict:
    try:
        return render_multiphase(vol, group, out_raw, parallel_scale)
    except Exception as exc:
        raise RuntimeError(
            f"[render:{group}] 3D multiphase cutaway-cube rendering failed and no fallback is "
            f"permitted for this figure (PyVista is required): {exc}"
        ) from exc


def finalize_renders(raw_paths: dict, out_paths: dict):
    """Identical, content-independent resize for every group -- NO alpha
    bounding-box crop, NO per-group foreground zoom, NO padding based on
    rendered content. Because the camera, parallel scale, and the full
    [0,128]^3 wireframe cube are already identical across groups (see
    render_multiphase), a plain uniform resize is the fair operation: any
    bounding-box crop would instead risk rescaling groups differently
    according to how much of the frame their surfaces happen to fill."""
    rs = _resample()
    for g, raw_path in raw_paths.items():
        im = Image.open(raw_path).convert("RGBA")
        canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
        canvas.alpha_composite(im)
        canvas = canvas.resize((CANVAS_PX, CANVAS_PX), rs)
        canvas.convert("RGB").save(out_paths[g])


# ============================================================================
# 10. OFFICIAL SCALAR METRIC LOADING -- exact official B_late CSV files
# only.
#
# The scalar/cleaned CSVs are long-format: one row per metric, identified
# by a metric-name column, with per-group value/error columns. The
# extractor below normalizes column names, resolves the metric-name column,
# matches the target metric by exact name first and then by its strict
# alias list, and aborts loudly if the match is missing or ambiguous --
# never guessing or silently substituting a nearby row.
# ============================================================================

METRIC_ID_CANDIDATES = ["metric", "metric_name", "descriptor", "name", "feature", "variable", "row_metric"]

VALUE_COL_ALIASES = {
    "real": ["real", "reference", "real_mean", "reference_mean", "real_value", "gt", "gt_mean"],
    "diffusion": ["diffusion", "microgen3d", "diffusion_mean", "microgen3d_mean", "diffusion_value",
                  "model_diffusion"],
    "gan": ["gan", "survol", "gan_mean", "survol_mean", "gan_value", "model_gan"],
}
# Only columns confirmed to already be percentages belong here. In
# particular "diffusion_error" / "gan_error" (and "microgen3d_error" /
# "survol_error") are deliberately EXCLUDED: in
# section_4_5_cleaned_scientific_metric_master.csv and
# section_4_5_main_paper_candidate_metrics.csv those columns hold absolute
# errors, not percentages, so matching them here would silently mislabel an
# absolute error as a relative-error percentage. When no column below is
# present, resolve_metric() falls back to computing the relative error from
# the real/diffusion/gan means instead of guessing at an ambiguous column.
ERROR_COL_ALIASES = {
    "diffusion": ["diffusion_relative_error_percent", "diffusion_relative_error_pct",
                  "diffusion_error_pct", "diffusion_rel_error_pct", "diffusion_pct_error",
                  "microgen3d_relative_error_percent", "microgen3d_error_pct",
                  "microgen3d_rel_error_pct"],
    "gan": ["gan_relative_error_percent", "gan_relative_error_pct", "gan_error_pct",
            "gan_rel_error_pct", "gan_pct_error", "survol_relative_error_percent",
            "survol_error_pct", "survol_rel_error_pct"],
}


def _load_metric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[metrics] missing official file: {path}")
    df = pd.read_csv(path)
    log(f"[metrics] loaded {path}: {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


def _resolve_metric_row(df: pd.DataFrame, path: Path, metric_key: str, aliases: list):
    id_col = pick_col(df, METRIC_ID_CANDIDATES, "metric-identifier column", required=False)
    if id_col is None:
        raise RuntimeError(f"[metrics] {path} has no recognizable metric-identifier column "
                            f"(tried {METRIC_ID_CANDIDATES}) -- cannot resolve '{metric_key}'")

    key_norms = [_norm(metric_key)] + [_norm(a) for a in aliases]
    id_norm = df[id_col].astype(str).map(_norm)
    matches = df[id_norm.isin(key_norms)]

    if matches.empty:
        # substring fallback, still required to be unique
        sub_mask = id_norm.map(lambda v: any(k in v or v in k for k in key_norms if k))
        matches = df[sub_mask]

    if matches.empty:
        raise RuntimeError(f"[metrics] '{metric_key}' not found in {path} (id column '{id_col}'); "
                            f"tried exact/alias names {key_norms}")
    if len(matches) > 1:
        raise RuntimeError(f"[metrics] '{metric_key}' matched {len(matches)} rows in {path} "
                            f"(ambiguous) -- refusing to guess which row is correct: "
                            f"{matches[id_col].tolist()}")
    return matches.iloc[0], id_col


def resolve_metric(df: pd.DataFrame, path: Path, metric_key: str, aliases: list) -> dict:
    """Returns {'real':.., 'diffusion':.., 'gan':.., 'diffusion_err':.., 'gan_err':..,
    'err_source': 'official'|'computed_from_means', 'raw_name':..}."""
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

    if diff_err_col is not None and gan_err_col is not None:
        diff_err = float(pd.to_numeric(row[diff_err_col], errors="coerce"))
        gan_err = float(pd.to_numeric(row[gan_err_col], errors="coerce"))
        err_source = "official"
    else:
        # Fallback permitted by task instructions: the official CSV provides
        # group means but not a precomputed relative-error column, so the
        # error is computed here (and explicitly logged as such) rather than
        # left unresolved.
        denom = abs(real_v) if real_v != 0 else np.nan
        diff_err = 100.0 * abs(diff_v - real_v) / denom
        gan_err = 100.0 * abs(gan_v - real_v) / denom
        err_source = "computed_from_means"

    log(f"[metrics] '{metric_key}' resolved from {path.name} (row metric='{raw_name}', "
        f"err_source={err_source}): real={real_v:.6g} diffusion={diff_v:.6g} gan={gan_v:.6g}  "
        f"diffusion_err={diff_err:.3f}%  gan_err={gan_err:.3f}%")

    return {"raw_name": raw_name, "real": real_v, "diffusion": diff_v, "gan": gan_v,
            "diffusion_err": diff_err, "gan_err": gan_err, "err_source": err_source, "source": str(path)}


def resolve_metric_with_fallback(metric_key: str, aliases: list, dfs_in_order: list) -> dict:
    """Tries SCALAR_COMPARISON first; falls back to CLEANED_SCIENTIFIC_MASTER
    then MAIN_PAPER_CANDIDATES only if the metric is genuinely absent there,
    per task instructions. Raises if no source resolves it unambiguously."""
    last_exc = None
    for path, df in dfs_in_order:
        if df is None:
            continue
        try:
            return resolve_metric(df, path, metric_key, aliases)
        except Exception as exc:
            last_exc = exc
            log(f"[metrics] '{metric_key}' not resolved from {path.name} ({exc}); trying next source")
    raise RuntimeError(f"[metrics] '{metric_key}' could not be resolved unambiguously from any "
                        f"official source -- stopping without saving. Last error: {last_exc}")


def check_metric_sanity(metric_key: str, resolved: dict, exp_diff: float, exp_gan: float):
    for label, got, exp in (("diffusion", resolved["diffusion_err"], exp_diff),
                             ("gan", resolved["gan_err"], exp_gan)):
        tol = max(METRIC_ERR_REL_TOL * abs(exp), METRIC_ERR_ABS_FLOOR)
        if abs(got - exp) > tol:
            raise RuntimeError(f"[sanity:metric] '{metric_key}' {label} error {got:.2f}% far from "
                                f"expected ~{exp:.2f}% (tol {tol:.2f} pts) -- stopping without saving")
    log(f"[sanity:metric] '{metric_key}' OK (diffusion {resolved['diffusion_err']:.2f}% ~"
        f"{exp_diff:.2f}%, gan {resolved['gan_err']:.2f}% ~{exp_gan:.2f}%)")


def check_duplicated_transition_interface_metric(df: pd.DataFrame, path: Path):
    """If 'mean transition rate' and 'total interface density' both exist
    and are numerically identical (same voxel-neighborhood definition),
    print the required note and never plot both."""
    id_col = pick_col(df, METRIC_ID_CANDIDATES, "metric-identifier column", required=False)
    if id_col is None:
        return
    id_norm = df[id_col].astype(str).map(_norm)
    trans_row = df[id_norm.isin([_norm("mean_transition_rate"), _norm("mean transition rate")])]
    iface_row = df[id_norm.isin([_norm("total_interface_density"), _norm("total interface density")])]
    if trans_row.empty or iface_row.empty:
        return
    real_col = pick_col(df, VALUE_COL_ALIASES["real"], "duplicate-check.real", required=False)
    if real_col is None:
        return
    v1 = float(pd.to_numeric(trans_row.iloc[0][real_col], errors="coerce"))
    v2 = float(pd.to_numeric(iface_row.iloc[0][real_col], errors="coerce"))
    if np.isfinite(v1) and np.isfinite(v2) and abs(v1 - v2) < 1e-9 * max(1.0, abs(v1)):
        log("Mean nearest-neighbor unlike-phase rate and total interface density are equivalent "
            "under the current voxel-neighborhood definition; plotting only one total-boundary "
            "metric.")


def load_panel_metrics(scalar_df, cleaned_df, main_paper_df):
    check_duplicated_transition_interface_metric(scalar_df, SCALAR_COMPARISON)
    sources = [(SCALAR_COMPARISON, scalar_df), (CLEANED_SCIENTIFIC_MASTER, cleaned_df),
               (MAIN_PAPER_CANDIDATES, main_paper_df)]
    resolved = {}
    for key, label, aliases, exp_diff, exp_gan in PANEL_B_METRICS:
        r = resolve_metric_with_fallback(key, aliases, sources)
        check_metric_sanity(key, r, exp_diff, exp_gan)
        resolved[key] = r
    return resolved


def load_active_euler(scalar_df, cleaned_df, main_paper_df):
    sources = [(SCALAR_COMPARISON, scalar_df), (CLEANED_SCIENTIFIC_MASTER, cleaned_df),
               (MAIN_PAPER_CANDIDATES, main_paper_df)]
    return resolve_metric_with_fallback("active_euler_abs", ACTIVE_EULER_ALIASES, sources)


# ============================================================================
# 11. ACTIVE-COMPONENT DIAGNOSTIC -- the per-sample CSV is the clean,
# unambiguous source of truth for group means (the summary CSV can carry
# awkward pandas multi-index headers). The JSON report and the summary CSV
# are used only as best-effort cross-checks: a source that fails to load
# is skipped with a note, but a source that DOES load and disagrees beyond
# tolerance still stops the run.
# ============================================================================

COMPONENT_FIELD_ALIASES = {
    "components": ["n_components_active", "active_components_per_sample", "mean_active_components",
                   "n_components", "component_count", "num_components", "active_component_count"],
    "singleton": ["components_size_1", "singleton_active_components", "mean_singleton_components",
                  "n_singleton", "singleton_count", "num_singleton"],
    "size_2_10": ["components_size_2_10", "active_components_size_2_10", "mean_components_2_10",
                  "n_components_2_10", "components_2_10", "size_2_10_count"],
    "small_frac": ["small_components_le10_fraction_of_active_voxels",
                   "small_components_active_voxel_fraction", "small_component_active_voxel_fraction",
                   "tiny_component_voxel_fraction", "small_frac", "small_components_voxel_fraction"],
}
GROUP_KEY_ALIASES = {
    "real": ["real", "reference", "ref", "gt", "real_test"],
    "diffusion": ["diffusion", "microgen3d"],
    "gan": ["gan", "survol", "survol_b_late"],
}


def load_active_component_means_from_by_sample() -> dict:
    if not ACTIVE_COMPONENT_BY_SAMPLE.exists():
        raise FileNotFoundError(f"[components] missing official file: {ACTIVE_COMPONENT_BY_SAMPLE}")
    df = pd.read_csv(ACTIVE_COMPONENT_BY_SAMPLE)
    log(f"[components] loaded {ACTIVE_COMPONENT_BY_SAMPLE}: {df.shape[0]} rows, "
        f"columns: {list(df.columns)}")
    gcol = pick_col(df, ["group", "set", "model", "category"], "components-by-sample.group")
    df = df.copy()
    df["_group"] = df[gcol].astype(str).str.strip().str.lower()

    out = {}
    for g in GROUPS:
        candidates = {_norm(a) for a in GROUP_KEY_ALIASES[g]}
        sub = df[df["_group"].map(_norm).isin(candidates)]
        if sub.empty:
            raise RuntimeError(f"[components] no by-sample rows for group '{g}' in "
                                f"{ACTIVE_COMPONENT_BY_SAMPLE}")
        rec = {}
        for field, aliases in COMPONENT_FIELD_ALIASES.items():
            col = pick_col(df, aliases, f"components-by-sample.{field}")
            rec[field] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        out[g] = rec
        log(f"[components:by_sample] group '{g}' (n={len(sub)}): {rec}")
    return out


def _deep_find_group_block(node, group: str, _depth=0):
    """Recursively search a nested JSON structure for a dict keyed by one of
    group's aliases, returning that subtree. Handles arbitrary nesting since
    the diagnostic JSON's exact schema is not specified."""
    if _depth > 6 or not isinstance(node, dict):
        return None
    aliases_norm = {_norm(a) for a in GROUP_KEY_ALIASES[group]}
    for k, v in node.items():
        if _norm(k) in aliases_norm and isinstance(v, dict):
            return v
    for v in node.values():
        if isinstance(v, dict):
            found = _deep_find_group_block(v, group, _depth + 1)
            if found is not None:
                return found
    return None


def _deep_find_field(node, aliases: list, _depth=0):
    if _depth > 6 or not isinstance(node, (dict, list)):
        return None
    aliases_norm = {_norm(a) for a in aliases}
    if isinstance(node, dict):
        for k, v in node.items():
            if _norm(k) in aliases_norm and isinstance(v, (int, float)):
                return float(v)
        for v in node.values():
            found = _deep_find_field(v, aliases, _depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _deep_find_field(item, aliases, _depth + 1)
            if found is not None:
                return found
    return None


def load_active_component_json() -> dict:
    if not ACTIVE_COMPONENT_REPORT.exists():
        raise FileNotFoundError(f"[components] missing official file: {ACTIVE_COMPONENT_REPORT}")
    data = json.loads(ACTIVE_COMPONENT_REPORT.read_text())
    out = {}
    for g in GROUPS:
        block = _deep_find_group_block(data, g)
        if block is None:
            raise RuntimeError(f"[components] could not locate a '{g}' block in "
                                f"{ACTIVE_COMPONENT_REPORT} (tried keys {GROUP_KEY_ALIASES[g]})")
        rec = {}
        for field, aliases in COMPONENT_FIELD_ALIASES.items():
            v = _deep_find_field(block, aliases)
            if v is None:
                raise RuntimeError(f"[components] group '{g}' block in {ACTIVE_COMPONENT_REPORT} "
                                    f"is missing field '{field}' (tried {aliases})")
            rec[field] = v
        out[g] = rec
        log(f"[components:json] group '{g}': {rec}")
    return out


def load_active_component_summary_csv() -> dict:
    if not ACTIVE_COMPONENT_SUMMARY.exists():
        raise FileNotFoundError(f"[components] missing official file: {ACTIVE_COMPONENT_SUMMARY}")
    df = pd.read_csv(ACTIVE_COMPONENT_SUMMARY)
    log(f"[components] loaded {ACTIVE_COMPONENT_SUMMARY}: {df.shape[0]} rows, "
        f"columns: {list(df.columns)}")
    gcol = pick_col(df, ["group", "set", "model", "category"], "components-summary.group")
    df = df.copy()
    df["_group"] = df[gcol].astype(str).str.strip().str.lower()
    out = {}
    for g in GROUPS:
        candidates = {_norm(a) for a in GROUP_KEY_ALIASES[g]}
        row = df[df["_group"].map(_norm).isin(candidates)]
        if len(row) != 1:
            raise RuntimeError(f"[components] expected exactly 1 summary row for group '{g}' in "
                                f"{ACTIVE_COMPONENT_SUMMARY}, got {len(row)}")
        row = row.iloc[0]
        rec = {}
        for field, aliases in COMPONENT_FIELD_ALIASES.items():
            col = pick_col(df, aliases, f"components-summary.{field}")
            rec[field] = float(pd.to_numeric(row[col], errors="coerce"))
        out[g] = rec
        log(f"[components:summary_csv] group '{g}': {rec}")
    return out


def cross_check_component_sources_optional(by_sample_means: dict) -> list:
    """Best-effort cross-check of by-sample means (the source of truth)
    against the JSON report and the summary CSV. Either source is skipped
    -- with a logged note, NOT counted under fallbacks_used, since the
    by-sample CSV remains the primary source either way -- if it cannot
    be loaded/parsed at all (the summary CSV in particular is known to
    carry awkward pandas multi-index headers); but if a source DOES
    load, a genuine disagreement beyond tolerance still stops the run
    rather than being silently ignored."""
    notes = []

    try:
        json_means = load_active_component_json()
    except Exception as exc:
        note = f"optional JSON cross-check skipped ({exc})"
        log(f"[components] {note}")
        notes.append(note)
        json_means = None
    if json_means is not None:
        for g in GROUPS:
            for field in COMPONENT_FIELD_ALIASES:
                a, b = by_sample_means[g][field], json_means[g][field]
                tol = max(COMPONENT_REL_TOL * abs(b), COMPONENT_ABS_FLOOR if field != "small_frac" else 1e-4)
                if abs(a - b) > tol:
                    raise RuntimeError(f"[components] group '{g}' field '{field}': by-sample mean "
                                        f"{a:.6g} disagrees with JSON report value {b:.6g} "
                                        f"(tol {tol:.6g}) -- diagnostic sources are inconsistent, stopping")
        log("[sanity:components] by-sample means agree with the JSON report within tolerance")

    try:
        summary_means = load_active_component_summary_csv()
    except Exception as exc:
        note = f"optional summary CSV cross-check skipped ({exc})"
        log(f"[components] {note}")
        notes.append(note)
        summary_means = None
    if summary_means is not None:
        for g in GROUPS:
            for field in COMPONENT_FIELD_ALIASES:
                a, b = by_sample_means[g][field], summary_means[g][field]
                tol = max(COMPONENT_REL_TOL * abs(b), COMPONENT_ABS_FLOOR if field != "small_frac" else 1e-4)
                if abs(a - b) > tol:
                    raise RuntimeError(f"[components] group '{g}' field '{field}': by-sample mean "
                                        f"{a:.6g} disagrees with summary CSV value {b:.6g} "
                                        f"(tol {tol:.6g}) -- diagnostic sources are inconsistent, stopping")
        log("[sanity:components] by-sample means agree with the summary CSV within tolerance")

    return notes


def check_component_sanity(by_sample_means: dict):
    for g in GROUPS:
        exp = EXPECTED_COMPONENTS[g]
        for field in ("components", "singleton", "size_2_10", "small_frac"):
            got = by_sample_means[g][field]
            e = exp[field]
            tol = max(COMPONENT_REL_TOL * abs(e), COMPONENT_ABS_FLOOR if field != "small_frac" else 1e-4)
            if abs(got - e) > tol:
                raise RuntimeError(f"[sanity:components] group '{g}' field '{field}': got {got:.6g} "
                                    f"vs expected ~{e:.6g} (tol {tol:.6g})")
    log("[sanity:components] all group/field values within tolerance of manuscript reference values")


# ============================================================================
# 12. OFFICIAL CURVE LOADING (panel b) -- resolves four structural
# descriptor curves from the official long-format curve-profiles CSV. No
# fake curves, no interpolation across mismatched grids: either the four
# curves resolve unambiguously with matching per-group coordinate grids,
# or the script prints the available curve names and stops.
# ============================================================================

# Panel-b curve selection is candidate-based, not single-name-mandatory: for
# each of the four 2x2-grid slots, candidates are tried IN PRIORITY ORDER
# (exact official descriptor names, confirmed against the real
# curve_profiles_long.csv) and the first candidate that passes full
# validation (see try_load_curve_candidate) is used. A candidate that is
# present but scientifically unusable (e.g. non-finite values for a
# non-chord-hist family) is rejected and logged -- this is candidate
# selection among official curves, never a fake/interpolated substitute,
# so it is not recorded under fallbacks_used.
SLOT_1_CANDIDATES = ["tpcf_active_radial", "lcc_tpcf_proxy_active_radial", "psd_active_radial",
                     "lineal_path_active_x", "lineal_path_active_y", "lineal_path_active_z"]
SLOT_2_CANDIDATES = ["cross_tpcf_pore_active_radial", "cross_tpcf_pore_cbd_radial",
                     "cross_tpcf_active_cbd_radial"]
SLOT_3_CANDIDATES = ["chord_hist_active_all_axes", "lineal_path_active_x", "lineal_path_active_y",
                     "lineal_path_active_z", "psd_active_radial", "lcc_tpcf_proxy_active_radial"]
SLOT_4_CANDIDATES = ["tpcf_cbd_radial", "psd_cbd_radial", "lcc_tpcf_proxy_cbd_radial",
                     "chord_hist_cbd_all_axes", "lineal_path_cbd_x", "lineal_path_cbd_y",
                     "lineal_path_cbd_z"]

PANEL_B_SLOTS = [
    ("slot1_active_same_phase", "SLOT 1 active same-phase structure", SLOT_1_CANDIDATES),
    ("slot2_cross_phase_pore_active", "SLOT 2 cross-phase pore-active structure", SLOT_2_CANDIDATES),
    ("slot3_domain_size_continuity", "SLOT 3 domain-size / continuity structure", SLOT_3_CANDIDATES),
    ("slot4_cbd_minority_phase", "SLOT 4 CBD / minority-phase structure", SLOT_4_CANDIDATES),
]

# Concise plot titles for every known official descriptor (covers all
# candidates above plus the other identities that may appear in the file,
# so a future candidate-list edit doesn't need a matching title edit).
DESCRIPTOR_TITLES = {
    "tpcf_active_radial": "Active same-phase TPCF", "psd_active_radial": "Active radial PSD",
    "lcc_tpcf_proxy_active_radial": "Active LCC-TPCF proxy",
    "lineal_path_active_x": "Active lineal path (X)", "lineal_path_active_y": "Active lineal path (Y)",
    "lineal_path_active_z": "Active lineal path (Z)",
    "chord_hist_active_all_axes": "Active chord-length distribution",
    "cross_tpcf_pore_active_radial": "Pore–active cross TPCF",
    "cross_tpcf_pore_cbd_radial": "Pore–CBD cross TPCF",
    "cross_tpcf_active_cbd_radial": "Active–CBD cross TPCF",
    "tpcf_cbd_radial": "CBD same-phase TPCF", "psd_cbd_radial": "CBD radial PSD",
    "lcc_tpcf_proxy_cbd_radial": "CBD LCC-TPCF proxy",
    "chord_hist_cbd_all_axes": "CBD chord-length distribution",
    "lineal_path_cbd_x": "CBD lineal path (X)", "lineal_path_cbd_y": "CBD lineal path (Y)",
    "lineal_path_cbd_z": "CBD lineal path (Z)",
    "tpcf_pore_radial": "Pore same-phase TPCF", "psd_pore_radial": "Pore radial PSD",
    "lcc_tpcf_proxy_pore_radial": "Pore LCC-TPCF proxy",
    "chord_hist_pore_all_axes": "Pore chord-length distribution",
    "lineal_path_pore_x": "Pore lineal path (X)", "lineal_path_pore_y": "Pore lineal path (Y)",
    "lineal_path_pore_z": "Pore lineal path (Z)",
}


# The official curve-profiles CSV is a long PER-SAMPLE table (one row per
# source x sample_id x descriptor-combo x r), identified by curve family
# via five columns together -- family, descriptor, phase, pair, axis --
# not a single curve-name column, and with no precomputed mean/std/count:
# those must be aggregated here from the per-sample "value" column. The
# group/source column is literally named "source".
CURVE_ID_COLS_CANDIDATES = ["family", "descriptor", "phase", "pair", "axis"]


def curve_schema(df: pd.DataFrame) -> dict:
    return {
        "group": pick_col(df, ["source", "group", "set", "model", "category", "class"],
                          "curve group/source column"),
        "id_cols": [pick_col(df, [c], f"curve '{c}' column") for c in CURVE_ID_COLS_CANDIDATES],
        "sample_id": pick_col(df, ["sample_id", "sample", "id"], "curve sample-id column",
                              required=False),
        "x": pick_col(df, ["r", "x", "radius_vox", "radius", "lag", "bin", "bin_center",
                           "coord", "chord_length"], "curve x column"),
        "value": pick_col(df, ["value", "mean", "y_mean", "value_mean", "y", "mean_value"],
                          "curve value column"),
    }


def map_curve_source_to_group(x) -> str:
    """Robust source->group mapping for the curve-profiles CSV. The real
    'source' column values are not the short 'real'/'diffusion'/'gan'
    tokens used elsewhere (they are full run identifiers such as
    'diffusion_microgen3d_128_final' or 'survol_nphase_direct128_B_late'),
    so this matches by keyword rather than exact/alias-set membership.
    Raises on a genuinely unrecognized source value rather than silently
    dropping those rows."""
    s = _norm(str(x))
    if s in {"realtest", "real", "reference", "ref"} or "real" in s or "reference" in s:
        return "real"
    if "diffusion" in s or "microgen3d" in s:
        return "diffusion"
    if "survol" in s or s == "gan" or "gan" in s:
        return "gan"
    raise RuntimeError(f"[curves] unrecognized curve source value: '{x}'")


def _curve_id_composite(id_cols: list, values: tuple) -> str:
    """Normalized composite key over the (family, descriptor, phase, pair,
    axis) tuple, used only for strict/alias name matching -- NaN/missing
    components are simply omitted, never guessed."""
    parts = [str(v) for v in values if pd.notna(v)]
    return _norm("_".join(parts))


def match_curve_value(series: pd.Series, value) -> pd.Series:
    """NaN-safe, format-tolerant equality for a curve-identity column: a
    NaN target matches only NaN entries; otherwise compares normalized
    string forms so minor case/formatting differences don't break the
    match (the underlying value is never altered, only compared)."""
    if pd.isna(value):
        return series.isna()
    return series.astype(str).map(_norm) == _norm(value)


def _row_mask_for_curve_id(df: pd.DataFrame, id_cols: list, values: tuple) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for c, v in zip(id_cols, values):
        mask &= match_curve_value(df[c], v)
    return mask


def _distinct_curve_ids(df: pd.DataFrame, id_cols: list) -> list:
    sub = df[id_cols].drop_duplicates()
    return [tuple(row[c] for c in id_cols) for _, row in sub.iterrows()]


def resolve_curve_id(df: pd.DataFrame, sch: dict, aliases: list):
    """Returns the matching (family, descriptor, phase, pair, axis) tuple,
    or None. Matches the composite of all five identifying columns'
    values against the alias list -- exact composite match first, then a
    substring fallback that must be unique."""
    ids = _distinct_curve_ids(df, sch["id_cols"])
    composites = {values: _curve_id_composite(sch["id_cols"], values) for values in ids}

    key_norms = [_norm(a) for a in aliases]
    for values, composite in composites.items():
        if composite in key_norms:
            return values

    sub_matches = [values for values, composite in composites.items()
                   if any(k and (k in composite or composite in k) for k in key_norms)]
    if len(sub_matches) == 1:
        return sub_matches[0]
    if len(sub_matches) > 1:
        readable = [dict(zip(sch["id_cols"], v)) for v in sub_matches]
        raise RuntimeError(f"[curve] aliases {aliases} matched {len(sub_matches)} distinct curve "
                            f"identities ambiguously: {readable}")
    return None


EXPECTED_SAMPLES_PER_CURVE = 50


def try_load_curve_candidate(curve_df: pd.DataFrame, sch: dict, descriptor_name: str):
    """Attempts to load and fully validate ONE candidate curve by exact
    'descriptor' value (falling back to the composite identity resolver
    only if the exact descriptor string isn't present). Never raises:
    returns (result_dict, None) on success or (None, reason_str) on
    failure, so the caller can try the next candidate in priority order.

    result_dict = {"data": {group: {x,y,std,count}}, "info": {...}}
    """
    desc_col = sch["id_cols"][1]  # CURVE_ID_COLS_CANDIDATES[1] == "descriptor"
    exact_mask = match_curve_value(curve_df[desc_col], descriptor_name)
    if exact_mask.any():
        sub = curve_df[exact_mask].copy()
    else:
        values = resolve_curve_id(curve_df, sch, [descriptor_name])
        if values is None:
            return None, f"descriptor '{descriptor_name}' not found in {CURVE_PROFILES.name}"
        sub = curve_df[_row_mask_for_curve_id(curve_df, sch["id_cols"], values)].copy()

    n_rows = len(sub)
    if n_rows == 0:
        return None, "matched 0 rows"

    family_col, phase_col, pair_col, axis_col = sch["id_cols"][0], sch["id_cols"][2], sch["id_cols"][3], sch["id_cols"][4]
    family_val = str(sub[family_col].iloc[0])
    raw_sources = sorted(sub[sch["group"]].astype(str).unique())
    sub["_mapped_group"] = sub[sch["group"]].map(map_curve_source_to_group)
    mapped_present = sorted(sub["_mapped_group"].dropna().unique().tolist())
    n_samples_per_group = {
        g: (int(sub.loc[sub["_mapped_group"] == g, sch["sample_id"]].nunique())
            if sch["sample_id"] is not None else int((sub["_mapped_group"] == g).sum()))
        for g in GROUPS
    }
    log(f"[curves] candidate '{descriptor_name}': {n_rows} matched rows, raw sources={raw_sources}, "
        f"mapped groups present={mapped_present}, rows/group="
        f"{ {g: int((sub['_mapped_group'] == g).sum()) for g in GROUPS} }, "
        f"unique sample_id/group={n_samples_per_group}")

    sub["_r"] = pd.to_numeric(sub[sch["x"]], errors="coerce")
    sub["_val"] = pd.to_numeric(sub[sch["value"]], errors="coerce")
    n_bad_r = int((~np.isfinite(sub["_r"])).sum())
    n_bad_val = int((~np.isfinite(sub["_val"])).sum())
    if n_bad_r or n_bad_val:
        bad = sub[(~np.isfinite(sub["_r"])) | (~np.isfinite(sub["_val"]))]
        log(f"[curves] candidate '{descriptor_name}': {n_bad_r} non-finite/non-numeric 'r', "
            f"{n_bad_val} non-finite/non-numeric 'value'; first rows: "
            f"{bad.head(5)[[sch['group'], sch['x'], sch['value']]].to_dict('records')}")

    if n_bad_r > 0:
        return None, f"{n_bad_r} non-finite/non-numeric 'r' entries"

    if n_bad_val > 0:
        if family_val.lower().startswith("chord"):
            # Documented, explicit exception: chord-hist NaN "value" entries
            # (a chord histogram legitimately has no count at some lengths)
            # may be filled with 0 -- and ONLY for family=="chord_hist*".
            sub["_val"] = sub["_val"].fillna(0.0)
            log(f"[curves] candidate '{descriptor_name}': filled {n_bad_val} NaN chord 'value' "
                f"entries with 0.0 (chord_hist family only, documented exception)")
        else:
            return None, f"{n_bad_val} non-finite/non-numeric 'value' entries (family '{family_val}' " \
                         f"is not chord_hist, so these are not filled)"

    for g in GROUPS:
        if g not in mapped_present:
            return None, f"mapped group '{g}' has no rows (raw sources present: {raw_sources})"

    per_group = {}
    for g in GROUPS:
        gsub = sub[sub["_mapped_group"] == g]
        if sch["sample_id"] is not None:
            count = gsub.groupby("_r")[sch["sample_id"]].nunique()
        else:
            count = gsub.groupby("_r")["_val"].count()
        agg = gsub.groupby("_r")["_val"].agg(["mean", "std"])
        agg = agg.join(count.rename("count")).reset_index().sort_values("_r")
        x = agg["_r"].to_numpy(float)
        y = agg["mean"].to_numpy(float)
        std = agg["std"].to_numpy(float)
        cnt = agg["count"].to_numpy(int)
        if not np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(std)):
            return None, f"group '{g}': non-finite aggregated r/mean/std (a NaN std usually means " \
                         f"a coordinate had only 1 sample)"
        if not np.all(cnt == cnt[0]):
            return None, f"group '{g}': sample count varies across r ({dict(zip(x.tolist(), cnt.tolist()))})"
        if cnt[0] != EXPECTED_SAMPLES_PER_CURVE:
            return None, f"group '{g}': {cnt[0]} samples/r-coordinate, expected exactly " \
                         f"{EXPECTED_SAMPLES_PER_CURVE}"
        per_group[g] = {"x": x, "y": y, "std": std, "count": cnt}

    ref_x = per_group["real"]["x"]
    for g in GROUPS:
        if per_group[g]["x"].shape != ref_x.shape or not np.allclose(per_group[g]["x"], ref_x):
            return None, f"group '{g}' r-coordinate grid differs from 'real' -- refusing to interpolate"

    info = {
        "descriptor": descriptor_name, "family": family_val,
        "phase": (sub[phase_col].iloc[0] if pd.notna(sub[phase_col].iloc[0]) else None),
        "pair": (sub[pair_col].iloc[0] if pd.notna(sub[pair_col].iloc[0]) else None),
        "axis": (sub[axis_col].iloc[0] if pd.notna(sub[axis_col].iloc[0]) else None),
        "n_r_coords": int(len(ref_x)),
        "samples_per_group": {g: int(per_group[g]["count"][0]) for g in GROUPS},
        "filled_nan_chord_values": int(n_bad_val) if (n_bad_val > 0 and family_val.lower().startswith("chord")) else 0,
    }
    return {"data": per_group, "info": info}, None


def resolve_panel_b_curves(curve_df: pd.DataFrame):
    sch = curve_schema(curve_df)

    # ---- source/group mapping diagnostics (printed before any resolution;
    # unchanged from the working version) -----------------------------------
    raw_sources = sorted(curve_df[sch["group"]].astype(str).unique())
    log(f"[curves] unique raw '{sch['group']}' values in {CURVE_PROFILES.name}: {raw_sources}")
    mapped = curve_df[sch["group"]].map(map_curve_source_to_group)
    for raw in raw_sources:
        mapped_to = mapped[curve_df[sch["group"]].astype(str) == raw].iloc[0]
        log(f"[curves]   '{raw}' -> '{mapped_to}'")
    for g in GROUPS:
        gmask = mapped == g
        n_rows = int(gmask.sum())
        n_samples = (curve_df.loc[gmask, sch["sample_id"]].nunique()
                     if sch["sample_id"] is not None else None)
        log(f"[curves] mapped group '{g}': {n_rows} rows"
            + (f", {n_samples} distinct sample_id values" if n_samples is not None else ""))

    distinct_ids = _distinct_curve_ids(curve_df, sch["id_cols"])
    readable_ids = [dict(zip(sch["id_cols"], v)) for v in distinct_ids]
    log(f"[curves] {len(distinct_ids)} distinct curve identities available in {CURVE_PROFILES.name} "
        f"(columns {sch['id_cols']}): {readable_ids}")

    curves = {}
    all_rejections = {}
    for slot_key, slot_label, candidates in PANEL_B_SLOTS:
        log(f"[curves] {slot_label}:")
        rejections = []
        selected = None
        for cand in candidates:
            result, reason = try_load_curve_candidate(curve_df, sch, cand)
            if result is None:
                log(f"[curves]   rejected {cand}: {reason}")
                rejections.append({"descriptor": cand, "reason": reason})
                continue
            log(f"[curves]   selected {cand}: passed, {result['info']['n_r_coords']} r-coordinates, "
                f"{EXPECTED_SAMPLES_PER_CURVE} samples/group/coordinate")
            selected = (cand, result)
            break

        if selected is None:
            raise RuntimeError(
                f"[curves] {slot_label}: none of the candidates {candidates} passed validation -- "
                f"stopping rather than inventing a plot. Rejections: {rejections}")

        cand_name, result = selected
        title = DESCRIPTOR_TITLES.get(cand_name, cand_name)
        curves[slot_key] = {"title": title, "raw_name": cand_name, "data": result["data"],
                            "info": result["info"], "rejections": rejections}
        all_rejections[slot_key] = rejections

    return curves, all_rejections


# ============================================================================
# 13. PANEL A -- representative multiphase morphology (fair, fixed protocol)
# ============================================================================

ROW_NAMES = ["3D volume", "X–Y slice", "X–Z slice", "Y–Z slice"]

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
        r = rep[g]
        a.text(0.035, 0.035, f"{r['pore']:.2f}/{r['active']:.2f}/{r['cbd']:.2f}",
               transform=a.transAxes, fontsize=6.6, color=SUBTEXT,
               ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))

        v = r["vol"]
        i = CENTRAL_SLICE_INDEX
        # XY: v[i,:,:] -> horizontal=X, vertical=Y. XZ: v[:,i,:] -> horizontal=X,
        # vertical=Z. YZ: v[:,:,i] -> horizontal=Y, vertical=Z. origin="lower"
        # makes each vertical axis increase upward -- no transpose needed. Fixed
        # index i=64 for every group -- full (128,128) slice extent, equal
        # aspect, no phase-bounding-box crop/zoom.
        slices = [v[i, :, :], v[:, i, :], v[:, :, i]]
        for ri, sl in enumerate(slices):
            a = fig.add_axes([x, row_y(ri + 1), cell_w, cell_h])
            a.imshow(sl, cmap=PHASE_CMAP, vmin=0, vmax=2, origin="lower",
                     interpolation="nearest", aspect="equal", extent=(0, 128, 0, 128))
            image_cell(a, COLORS[g])
            if c == 0:
                for txt, (tx, ty), ha, va in DIRECTION_LABELS[ROW_NAMES[ri + 1]]:
                    a.text(tx, ty, txt, transform=a.transAxes, fontsize=6.9, fontweight="bold",
                           color="white", ha=ha, va=va,
                           bbox=dict(facecolor=TEXT, edgecolor="none", alpha=0.55, pad=1.2))

    # compact phase legend (pore / active / CBD), centered under the grid
    legend_y = ay_ + 0.015
    item_widths = [0.062, 0.072, 0.062]
    total_w = sum(item_widths)
    lx = gx0 + label_w + gap_x + (grid_w - label_w - gap_x - total_w) / 2.0
    for label_id, w in zip((0, 1, 2), item_widths):
        fig.add_artist(Rectangle((lx, legend_y - 0.006), 0.013, 0.011,
                                  transform=fig.transFigure, facecolor=PHASE_COLORS[label_id],
                                  edgecolor=SPINE, linewidth=0.6, zorder=101))
        fig.text(lx + 0.019, legend_y, PHASE_NAMES[label_id], ha="left", va="center",
                 fontsize=7.4, color=SUBTEXT, zorder=101)
        lx += w


# ============================================================================
# 14. PANEL B -- official structural descriptor curves (2x2 grid)
# ============================================================================


# Per-family axis labels -- more informative than a single generic
# "Coordinate (vox)" / "Value" pair for every subplot, keyed by the
# official curve-profiles "family" value of the SELECTED candidate (not by
# descriptor string, so this stays correct however the candidate-priority
# fallback in resolve_panel_b_curves resolves each slot).
FAMILY_AXIS_LABELS = {
    "tpcf": ("Lag / radius (vox)", "Two-point correlation"),
    "cross_tpcf": ("Lag / radius (vox)", "Two-point correlation"),
    "lineal_path": ("Segment length (vox)", "Lineal-path probability"),
    "chord_hist": ("Chord length (vox)", "Density"),
    "psd": ("Radial frequency / bin", "Spectral density"),
    "lcc_tpcf_proxy": ("Lag / radius (vox)", "Two-point correlation (LCC proxy)"),
}
DEFAULT_AXIS_LABELS = ("Coordinate (vox)", "Value")


def axis_labels_for_family(family) -> tuple:
    return FAMILY_AXIS_LABELS.get(str(family).lower(), DEFAULT_AXIS_LABELS)


def plot_descriptor_curve(ax, curve_entry, show_legend=False):
    style_axis(ax)
    ax.set_title(curve_entry["title"], pad=4.5, color=TEXT, fontweight="bold")
    xlabel, ylabel = axis_labels_for_family(curve_entry["info"]["family"])
    ax.set_xlabel(xlabel, color=SUBTEXT)
    ax.set_ylabel(ylabel, color=SUBTEXT)

    data = curve_entry["data"]
    for g in GROUPS:
        d = data[g]
        x, y, std = d["x"], d["y"], d["std"]
        if g == "real" and std is not None and np.any(np.isfinite(std)) and np.any(std > 0):
            # Kept subtle (low alpha, drawn first/below) so the MicroGen3D
            # and SurVol lines on top of it stay clearly readable.
            ax.fill_between(x, y - std, y + std, color=COLORS[g], alpha=0.14, linewidth=0, zorder=1)
        ax.plot(x, y, color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                solid_capstyle="round", label=LABELS[g], zorder=3 if g == "real" else 4)

    ax.margins(x=0.02, y=0.06)

    if show_legend:
        leg = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=SPINE,
                        framealpha=0.94, handlelength=2.3, borderpad=0.45, labelspacing=0.34)
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(8)


def build_panel_b(fig, card, curves):
    bx_, by_, bw_, bh_ = card
    b_pl, b_pr, b_pt, b_pb = 0.045, 0.018, 0.040, 0.046
    b_gx, b_gy = 0.058, 0.076

    plot_w = (bw_ - b_pl - b_pr - b_gx) / 2.0
    plot_h = (bh_ - b_pt - b_pb - b_gy) / 2.0

    order = ["active_same_tpcf", "pore_active_cross_tpcf", "active_chord", "slot4"]
    keys = list(curves.keys())
    # keys are inserted in slot order by resolve_panel_b_curves: [slot1, slot2, slot3, slot4-actual]
    for i, key in enumerate(keys):
        r, c = divmod(i, 2)
        x = bx_ + b_pl + c * (plot_w + b_gx)
        y = by_ + bh_ - b_pt - (r + 1) * plot_h - r * b_gy
        ax = fig.add_axes([x, y, plot_w, plot_h])
        plot_descriptor_curve(ax, curves[key], show_legend=(i == 1))


# ============================================================================
# 15. PANEL C -- mechanistic descriptors: interface/contact-hierarchy
# fingerprint (c-left) and active-domain continuity/fragmentation (c-right)
# ============================================================================

FINGERPRINT_ORDER = ["interface_pore_active", "interface_pore_cbd", "interface_active_cbd",
                     "tpb_proxy", "cbd_to_pore", "cbd_to_active"]
FINGERPRINT_LABELS = ["Pore–active", "Pore–CBD", "Active–CBD", "TPB proxy", "CBD→pore", "CBD→active"]

FRAGMENTATION_ORDER = ["euler", "components", "singleton", "size_2_10", "small_frac"]
FRAGMENTATION_LABELS = ["Active Euler\n|χ|", "Active\ncomponents", "Singleton\ncomponents",
                        "Components\n2–10 vox", "Tiny-component\nvoxel fraction"]


def plot_interface_fingerprint(ax, panel_metrics):
    style_axis(ax)
    ax.set_title("Interface / contact-hierarchy fingerprint", pad=6.5, color=TEXT, fontweight="bold")
    ax.set_ylabel("Density / fraction (log scale)", color=SUBTEXT)
    ax.set_yscale("log")

    # Interface densities, the TPB proxy density, and CBD contact fractions
    # span different units/magnitudes; a shared linear axis flattens the
    # smaller ones, so this uses a shared log axis instead (kept simple
    # rather than a per-metric ratio-to-Reference normalization).
    all_vals = np.array([panel_metrics[key][g] for key in FINGERPRINT_ORDER for g in GROUPS], dtype=float)
    positive = all_vals[all_vals > 0]
    floor = float(positive.min()) / 10.0 if positive.size else 1e-6

    xs = np.arange(len(FINGERPRINT_ORDER))
    for g in GROUPS:
        vals = np.array([panel_metrics[key][g] for key in FINGERPRINT_ORDER], dtype=float)
        n_floored = int(np.sum(vals <= 0))
        if n_floored:
            log(f"[panel-c] group '{g}': {n_floored} fingerprint value(s) <= 0, floored to "
                f"{floor:.3e} for log-scale display only (underlying values unchanged elsewhere)")
        vals_plot = np.where(vals > 0, vals, floor)
        ax.plot(xs, vals_plot, color=COLORS[g], linestyle=LINESTYLES[g], linewidth=LINEWIDTHS[g],
                marker=MARKERS[g], markersize=5.5, markerfacecolor=COLORS[g],
                markeredgecolor="black", markeredgewidth=0.6,
                label=LABELS[g], zorder=3 if g == "real" else 4)

    ax.set_xticks(xs)
    ax.set_xticklabels(FINGERPRINT_LABELS, fontsize=7.2)
    ax.set_xlim(-0.4, len(FINGERPRINT_ORDER) - 0.6)
    ax.margins(y=0.16)

    # "best" rather than a fixed corner: the fingerprint's value pattern
    # (which metric is lowest/highest) is data-dependent, so a fixed corner
    # can end up sitting on top of a curve (e.g. a near-zero first point);
    # letting matplotlib pick the least-overlapping corner keeps every
    # curve visible.
    leg = ax.legend(loc="best", frameon=True, facecolor="white", edgecolor=SPINE,
                    framealpha=0.94, handlelength=2.0, borderpad=0.4, labelspacing=0.3, fontsize=7.0)
    leg.get_frame().set_linewidth(0.6)


def plot_active_fragmentation(ax, euler_resolved, component_means):
    style_axis(ax)
    ax.set_title("Active-domain continuity / fragmentation", pad=6.5, color=TEXT, fontweight="bold")
    ax.set_ylabel("Value (log scale)", color=SUBTEXT)
    ax.set_yscale("log")

    values = {
        "euler": {g: abs(euler_resolved[g]) for g in GROUPS},
        "components": {g: component_means[g]["components"] for g in GROUPS},
        "singleton": {g: component_means[g]["singleton"] for g in GROUPS},
        "size_2_10": {g: component_means[g]["size_2_10"] for g in GROUPS},
        "small_frac": {g: component_means[g]["small_frac"] for g in GROUPS},
    }

    xs = np.arange(len(FRAGMENTATION_ORDER))
    offset = {"real": -0.16, "diffusion": 0.0, "gan": 0.16}
    for g in GROUPS:
        vals = np.array([values[key][g] for key in FRAGMENTATION_ORDER])
        ax.scatter(xs + offset[g], vals, s=70, marker=MARKERS[g], color=COLORS[g],
                   edgecolors="black", linewidths=0.9, zorder=5, label=LABELS[g])

    ax.set_xticks(xs)
    ax.set_xticklabels(FRAGMENTATION_LABELS, fontsize=7.0)
    ax.set_xlim(-0.5, len(FRAGMENTATION_ORDER) - 0.5)
    # 30% extra headroom on the log scale (matplotlib computes margins in
    # the transformed/log coordinate space once set_yscale is applied, so
    # this adds genuine extra decades rather than a naive linear fraction)
    # so the highest MicroGen3D markers never touch the top border, while
    # the smallest tiny-component-fraction markers stay clearly visible.
    ax.margins(y=0.30)


def build_panel_c(fig, card, panel_metrics, euler_resolved, component_means):
    cx_, cy_, cw_, ch_ = card
    header_title_y = cy_ + ch_ - 0.020
    # Extra headroom below the panel-level header ("Interface hierarchy and
    # active-domain continuity") before the subplot titles/axes start, so
    # the header and the c-left/c-right subplot titles don't read as
    # cramped against each other.
    plot_top = header_title_y - 0.046
    c_pl, c_pr, c_pb, c_gx = 0.045, 0.020, 0.046, 0.055
    plot_bottom = cy_ + c_pb
    plot_h_c = plot_top - plot_bottom
    plot_w_c = (cw_ - c_pl - c_pr - c_gx) / 2.0
    left_x = cx_ + c_pl
    right_x = left_x + plot_w_c + c_gx

    fig.text(cx_ + 0.014, header_title_y, "Interface hierarchy and active-domain continuity",
             ha="left", va="top", fontsize=9.8, fontweight="bold", color=TEXT)

    ax_left = fig.add_axes([left_x, plot_bottom, plot_w_c, plot_h_c])
    ax_right = fig.add_axes([right_x, plot_bottom, plot_w_c, plot_h_c])

    plot_interface_fingerprint(ax_left, panel_metrics)
    plot_active_fragmentation(ax_right, euler_resolved, component_means)


# ============================================================================
# 16. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT               = {PROJECT}")
    log(f"[paths] REAL_DIR              = {REAL_DIR}")
    log(f"[paths] DIFFUSION_DIR         = {DIFFUSION_DIR}")
    log(f"[paths] GAN_DIR               = {GAN_DIR}")
    log(f"[paths] SCALAR_COMPARISON     = {SCALAR_COMPARISON}")
    log(f"[paths] CURVE_COMPARISON      = {CURVE_COMPARISON} (logged for provenance; panel b uses CURVE_PROFILES)")
    log(f"[paths] CURVE_PROFILES        = {CURVE_PROFILES}")
    log(f"[paths] PER_SAMPLE_SCALARS    = {PER_SAMPLE_SCALARS} (not used by this figure's panels; logged for provenance)")
    log(f"[paths] ALL_METRIC_COMPARISON = {ALL_METRIC_COMPARISON} (not used by this figure's panels; logged for provenance)")
    log(f"[paths] MAIN_PAPER_CANDIDATES = {MAIN_PAPER_CANDIDATES}")
    log(f"[paths] CLEANED_SCIENTIFIC_MASTER = {CLEANED_SCIENTIFIC_MASTER}")
    log(f"[paths] ACTIVE_COMPONENT_REPORT   = {ACTIVE_COMPONENT_REPORT}")
    log(f"[paths] ACTIVE_COMPONENT_BY_SAMPLE = {ACTIVE_COMPONENT_BY_SAMPLE}  (primary source of truth for group means)")
    log(f"[paths] ACTIVE_COMPONENT_SUMMARY   = {ACTIVE_COMPONENT_SUMMARY}  (optional cross-check only)")

    fallback_notes = []

    # ---- volumes: load, validate, compute descriptors, sanity-check -------
    volumes_by_group, descriptors_by_group = {}, {}
    for g in GROUPS:
        volumes = load_group_volumes(g)
        volumes_by_group[g] = volumes
        rows = compute_group_descriptors(volumes)
        descriptors_by_group[g] = rows

        direct_mean = {k: float(np.mean([r[k] for r in rows])) for k in ("pore", "active", "cbd")}
        direct_txyz = (float(np.mean([r["tx"] for r in rows])),
                       float(np.mean([r["ty"] for r in rows])),
                       float(np.mean([r["tz"] for r in rows])))
        check_phase_fractions(g, direct_mean)
        check_transitions(g, direct_txyz)

    # ---- representative selection (independent per group) -----------------
    real_row, real_mean_vec = choose_representative_real(descriptors_by_group["real"])
    rep_rows = {"real": real_row}
    for g in ("diffusion", "gan"):
        rep_rows[g] = choose_representative_generated(g, descriptors_by_group[g], real_mean_vec)

    rep = {}
    i = CENTRAL_SLICE_INDEX
    for g in GROUPS:
        r = rep_rows[g]
        vol = volumes_by_group[g][r["file"]]
        labels_present = sorted(np.unique(vol).tolist())
        log(f"[panel-a-audit:{g}] file={r['file']}  shape={vol.shape}  labels={labels_present}  "
            f"slice_indices=(z={i}, y={i}, x={i})  full_volume_used=yes  crop_applied=none")
        rep[g] = {"file": r["file"], "vol": vol, "pore": r["pore"], "active": r["active"],
                  "cbd": r["cbd"], "tx": r["tx"], "ty": r["ty"], "tz": r["tz"],
                  "z_idx": i, "y_idx": i, "x_idx": i}

    # ---- panel a: 3D multiphase isosurfaces (fixed, fair protocol) --------
    parallel_scale = 0.72 * max(max(rep[g]["vol"].shape) for g in GROUPS)
    raw_paths = {g: TMP / f"raw_{g}.png" for g in GROUPS}
    png_paths = {g: TMP / f"render_{g}.png" for g in GROUPS}
    render_audit = {}
    for g in GROUPS:
        render_audit[g] = render_group_multiphase(rep[g]["vol"], g, raw_paths[g], parallel_scale)
        log(f"[panel-a-audit:{g}] render_extent={render_audit[g]['render_extent']}  "
            f"parallel_scale={render_audit[g]['parallel_scale']:.4f}  "
            f"cutaway_rule='{render_audit[g]['cutaway_rule']}'  "
            f"render_downsample_factor={render_audit[g]['render_downsample_factor']} "
            f"(display-resolution reduction only, not a crop -- render_extent above is still the "
            f"full volume)")
    finalize_renders(raw_paths, png_paths)
    log("Panel a crop/zoom used: none")

    # ---- panel b/c: official scalar metrics --------------------------------
    scalar_df = _load_metric_csv(SCALAR_COMPARISON)
    try:
        cleaned_df = _load_metric_csv(CLEANED_SCIENTIFIC_MASTER)
    except FileNotFoundError as exc:
        log(f"[metrics] optional fallback source unavailable: {exc}")
        cleaned_df = None
    try:
        main_paper_df = _load_metric_csv(MAIN_PAPER_CANDIDATES)
    except FileNotFoundError as exc:
        log(f"[metrics] optional fallback source unavailable: {exc}")
        main_paper_df = None

    panel_metrics = load_panel_metrics(scalar_df, cleaned_df, main_paper_df)
    for key, _label, _aliases, _ed, _eg in PANEL_B_METRICS:
        r = panel_metrics[key]
        if Path(r["source"]) != SCALAR_COMPARISON:
            fallback_notes.append(f"'{key}' resolved from {Path(r['source']).name} instead of "
                                   f"{SCALAR_COMPARISON.name} (metric missing/ambiguous there)")
        if r["err_source"] == "computed_from_means":
            fallback_notes.append(f"'{key}' relative error computed from official group means "
                                   f"(no precomputed error column in {Path(r['source']).name})")

    euler_row = load_active_euler(scalar_df, cleaned_df, main_paper_df)
    if Path(euler_row["source"]) != SCALAR_COMPARISON:
        fallback_notes.append(f"'active_euler_abs' resolved from {Path(euler_row['source']).name} "
                               f"instead of {SCALAR_COMPARISON.name}")
    euler_resolved = {"real": euler_row["real"], "diffusion": euler_row["diffusion"], "gan": euler_row["gan"]}

    # ---- panel c: active-component diagnostic (by-sample primary) ---------
    component_means = load_active_component_means_from_by_sample()
    # Not a fallback: the by-sample CSV is the primary source of truth
    # either way, so a skipped JSON/summary-CSV cross-check is logged
    # separately rather than under fallbacks_used.
    component_cross_check_notes = cross_check_component_sources_optional(component_means)
    check_component_sanity(component_means)

    fingerprint_values = {key: {g: panel_metrics[key][g] for g in GROUPS} for key in FINGERPRINT_ORDER}

    # ---- panel b: official curves -------------------------------------------
    curve_df = _load_metric_csv(CURVE_PROFILES)
    curves, curve_rejections = resolve_panel_b_curves(curve_df)

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
    build_panel_b(fig, card_b, curves)
    build_panel_c(fig, card_c, panel_metrics, euler_resolved, component_means)

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
        "panel_a_audit": {
            g: {"file": rep[g]["file"].name, "shape": list(rep[g]["vol"].shape),
                "slice_indices": {"z": rep[g]["z_idx"], "y": rep[g]["y_idx"], "x": rep[g]["x_idx"]},
                "full_volume_used": True, "crop_applied": "none",
                "render_extent": render_audit[g]["render_extent"],
                "parallel_scale": render_audit[g]["parallel_scale"],
                "cutaway_rule": render_audit[g]["cutaway_rule"],
                "render_downsample_factor": render_audit[g]["render_downsample_factor"]}
            for g in GROUPS
        },
        "representative": {
            g: {"file": rep[g]["file"].name, "pore": rep[g]["pore"], "active": rep[g]["active"],
                "cbd": rep[g]["cbd"], "tx": rep[g]["tx"], "ty": rep[g]["ty"], "tz": rep[g]["tz"]}
            for g in GROUPS
        },
        "panel_b_curves": {
            key: {"slot": key, "selected_descriptor": curves[key]["raw_name"],
                  "family": curves[key]["info"]["family"], "phase": curves[key]["info"]["phase"],
                  "pair": curves[key]["info"]["pair"], "axis": curves[key]["info"]["axis"],
                  "title": curves[key]["title"], "n_r_coords": curves[key]["info"]["n_r_coords"],
                  "samples_per_group": curves[key]["info"]["samples_per_group"],
                  "filled_nan_chord_values": curves[key]["info"]["filled_nan_chord_values"],
                  "candidates_rejected_before_selection": curves[key]["rejections"]}
            for key in curves
        },
        "panel_b_curve_candidate_rejections": curve_rejections,
        "panel_c_metrics": {
            key: {"raw_name": panel_metrics[key]["raw_name"], "real": panel_metrics[key]["real"],
                  "diffusion": panel_metrics[key]["diffusion"], "gan": panel_metrics[key]["gan"],
                  "source": panel_metrics[key]["source"]}
            for key, *_ in PANEL_B_METRICS
        },
        "active_euler_abs": {"real": euler_row["real"], "diffusion": euler_row["diffusion"],
                              "gan": euler_row["gan"], "source": euler_row["source"]},
        "active_component_means_by_sample": component_means,
        "component_cross_check_notes": component_cross_check_notes,
        "fallbacks_used": fallback_notes,
        "saved": {"png": str(png), "pdf": str(pdf), "svg": str(svg), "tiff": str(tif_out)},
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))

    # =========================== SUMMARY ========================================
    log("\nVolume folders:")
    for g in GROUPS:
        log(f"  {g}: {GROUP_VOLUME_DIRS[g]}  ({EXPECTED_N} NPYs, shape {EXPECTED_SHAPE})")

    log("\nRepresentative samples (independent per-group selection):")
    for g in GROUPS:
        r = rep[g]
        log(f"  {g}: {r['file'].name}  pore={r['pore']:.6f} active={r['active']:.6f} cbd={r['cbd']:.6f}  "
            f"tx={r['tx']:.6f} ty={r['ty']:.6f} tz={r['tz']:.6f}")

    log("\nPanel b curves resolved:")
    for slot_key, slot_label, _candidates in PANEL_B_SLOTS:
        log(f"[curves] {slot_label}:")
        for rej in curves[slot_key]["rejections"]:
            log(f"  rejected {rej['descriptor']}: {rej['reason']}")
        info = curves[slot_key]["info"]
        log(f"  selected {curves[slot_key]['raw_name']}: passed, {info['n_r_coords']} r-coordinates, "
            f"{EXPECTED_SAMPLES_PER_CURVE} samples/group/coordinate -> \"{curves[slot_key]['title']}\"")

    log("\nActive-component diagnostic means (by-sample, primary source of truth):")
    for g in GROUPS:
        log(f"  {g}: {component_means[g]}")

    if component_cross_check_notes:
        log("\nComponent cross-check notes (not fallbacks -- by-sample CSV remains the source of truth):")
        for note in component_cross_check_notes:
            log(f"  - {note}")

    log("\nFallbacks used:")
    if fallback_notes:
        for note in fallback_notes:
            log(f"  - {note}")
    else:
        log("  none")

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out, AUDIT_JSON, CAPTION_TXT):
        log(" ", p.resolve())

    log("\nFINAL FIGURE 4.5 AUDIT PASS")


if __name__ == "__main__":
    sys.exit(main())
