#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual / graphical abstract (magma, final)
"Task-oriented benchmark" hub-and-spoke map over Sections 4.1-4.7.

RUNTIME NOTE: this script is written and reviewed without access to any of
the RA2 authoritative data paths encoded in the seven Section 4.1-4.7
scripts. It never searches the filesystem for a substitute location, never
fabricates a scientific image, and never invents an alternative path. Every
material structure shown in a final diffusion/GAN cell is produced by
DIRECTLY REUSING the corresponding Section 4.x script's own representative-
selection and rendering functions -- imported as a module (never executed
as `__main__`, so only their module-level constants/functions load, not
their own main()/figure-saving pipeline) and called exactly as that
section's own main() calls them. Nothing here reimplements or approximates
that scientific logic.

PATH BINDING: Sections 4.1-4.3 derive their own PROJECT from `__file__`,
which is only correct when that section script runs standalone from its
own checkout location. Once THIS script imports it as a module from a
different project layout, that derivation resolves to the wrong root --
confirmed by a real RA2 path audit. So immediately after import, this
script explicitly rebinds each of those three modules' path-valued
globals (PROJECT/ROOT/GROUP_DIRS/... etc.) to the exact RA2 roots in
RA2_ROOTS below (see configure_section_paths()). Sections 4.4-4.7 already
hardcode their own correct RA2 root inside the locked script and are left
untouched -- only checked for agreement with RA2_ROOTS (see
validate_section_paths()). A full preflight validates all seven tasks'
authoritative inputs before any scientific loading or rendering starts,
so a broken path on a later task is caught immediately rather than after
earlier tasks already rendered.

The only genuinely new drawing in this script is:
  (a) the hub-and-spoke page layout and card chrome (style-matched to
      Figures 4.1-4.7: BG/CARD_FACE/CARD_EDGE/GRID/SPINE/TEXT/SUBTEXT,
      DejaVu Sans, orange/purple model-family colors), and
  (b) seven tiny, deliberately minimal, purely-vector challenge schematics
      (rectangles/arrows/wireframe cubes/nodes-edges/an oriented ellipse),
      built only from matplotlib primitives -- explanatory diagrams, never
      fabricated microstructures -- except the super-resolution schematic,
      which instead crops two small deterministic (top-left, fixed-size,
      never manually chosen) windows out of the already-loaded real LR/HR
      representative pair, per the task brief's explicit preference for a
      fully grounded LR->HR schematic there.

Fail-safe behavior: if an authoritative Section 4.x input required for a
final diffusion/GAN visual is missing or fails validation, this script
raises and stops -- it never substitutes a placeholder, a different
checkpoint, a different representative, or a fabricated texture. Purely
vector schematics need no dataset and cannot fail this way.

Run on RA2 (after this script and the seven Section 4.1-4.7 scripts are
pulled to the same `scripts/` directory):

    cd /home/ra2/benchmark_figures
    conda activate poregen_nmc128

    mkdir -p visual_abstract

    python -u scripts/make_visual_abstract_magma_final.py \
      2>&1 | tee visual_abstract/visual_abstract_magma_final_run.log
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================================
# 1. PATHS
# ============================================================================

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent

OUT = REPO / "visual_abstract"
TMP = OUT / "_render_cache"
STEM = "visual_abstract_magma_final"
AUDIT_JSON = OUT / f"{STEM}_run_metadata.json"

# Exact, authoritative Section 4.x scripts this abstract reuses. Never an
# alternative/older/debug copy -- these are the same locked files that
# produce Figures 4.1-4.7 themselves.
SECTION_SCRIPTS = {
    "multiscale": SCRIPTS_DIR / "make_fig4_1_composite_magma_final.py",
    "2d_to_3d": SCRIPTS_DIR / "make_fig4_2_composite_magma_final.py",
    "super_resolution": SCRIPTS_DIR / "make_fig4_3_composite_magma_final.py",
    "anisotropy": SCRIPTS_DIR / "make_fig4_4_composite_magma_final.py",
    "multiphase": SCRIPTS_DIR / "make_fig4_5_composite_magma_final.py",
    "topology": SCRIPTS_DIR / "make_fig4_6_composite_magma_final.py",
    "large_volume": SCRIPTS_DIR / "make_fig4_7_composite_magma_final.py",
}

# Confirmed-on-RA2 authoritative project/data roots for each task. Sections
# 4.1-4.3 originally derived PROJECT from `__file__`, which resolves wrong
# once THIS script (not the section script itself) is the one importing
# them -- so those three get their real path family rebound explicitly in
# configure_section_paths() below. Sections 4.4-4.7 already hardcode their
# own correct RA2 root inside the locked script; those are only checked for
# agreement against this registry in validate_section_paths(), never
# rebound. Never searched for broadly, never symlinked -- these are the
# exact paths confirmed directly on RA2.
RA2_ROOTS = {
    "multiscale": Path("/home/ra2/4.1/poregen_drp471_4_1"),
    "2d_to_3d": Path("/home/ra2/4.2/microlad_nmc_2d_to_3d"),
    "super_resolution": Path("/home/ra2/4.3"),
    "anisotropy": Path("/home/ra2/4_4_survol_glass_anisotropy_clean"),
    "multiphase": Path("/home/ra2/4_5_microgen3d_muller128cube"),
    "topology": Path("/home/ra2/4.6/poregen_topology_nmc128_ubuntu_ready_v2"),
    "large_volume": Path("/home/ra2/section_4_7_transfer/section_4_7_transfer_package"),
}

TASK_TITLES = {
    "multiscale": "Multiscale",
    "2d_to_3d": "2D→3D",
    "super_resolution": "Super-resolution",
    "anisotropy": "Anisotropy",
    "multiphase": "Multiphase",
    "topology": "Topology",
    "large_volume": "Large volume",
}

TOP_ROW = ["multiscale", "2d_to_3d", "super_resolution"]
BOTTOM_ROW = ["anisotropy", "multiphase", "topology", "large_volume"]

# ============================================================================
# 2. STYLE -- identical contract to Figures 4.1-4.7
# ============================================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.linewidth": 0.7,
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

# Locked model-family colors (identical to the "diffusion"/"gan" colors
# already used throughout Figures 4.1-4.7).
DIFFUSION = "#F2A93B"
GAN = "#772A8E"

CARD_LW = 0.9
HUB_LW = 1.3
CELL_LW = 1.8
CONNECTOR_COLOR = "#A79BC0"

FIG_W, FIG_H = 17.8, 10.2

# ============================================================================
# 3. GEOMETRY -- hub + 3-top / 4-bottom task grid, all in figure fractions
# ============================================================================

MARGIN = 0.03
GAP = 0.020

ROW_H = 0.40
BOTTOM_Y0 = MARGIN
TOP_Y0 = 1.0 - MARGIN - ROW_H

HUB_W, HUB_H = 0.150, 0.085
HUB_X0 = 0.5 - HUB_W / 2.0
HUB_Y0 = 0.5 - HUB_H / 2.0
HUB_CARD = [HUB_X0, HUB_Y0, HUB_W, HUB_H]

FULL_W = 1.0 - 2 * MARGIN


def _row_cards(y0, n):
    card_w = (FULL_W - (n - 1) * GAP) / n
    cards = []
    for i in range(n):
        x0 = MARGIN + i * (card_w + GAP)
        cards.append([x0, y0, card_w, ROW_H])
    return cards


TOP_CARDS = dict(zip(TOP_ROW, _row_cards(TOP_Y0, 3)))
BOTTOM_CARDS = dict(zip(BOTTOM_ROW, _row_cards(BOTTOM_Y0, 4)))
TASK_CARDS = {**TOP_CARDS, **BOTTOM_CARDS}

assert HUB_Y0 >= BOTTOM_Y0 + ROW_H - 1e-9, "hub card overlaps the bottom task row"
assert HUB_Y0 + HUB_H <= TOP_Y0 + 1e-9, "hub card overlaps the top task row"

# ============================================================================
# 4. GENERIC HELPERS
# ============================================================================

RUN_LOG: list = []


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    RUN_LOG.append(msg)


def import_section_module(name: str, path: Path):
    """Load a Section 4.x script as an importable module WITHOUT running its
    `if __name__ == "__main__":` block -- only its module-level constants
    and function definitions execute (no data loading, no rendering, no
    figure saving happens at import time in any of the seven scripts)."""
    if not path.exists():
        raise FileNotFoundError(
            f"[import:{name}] required Section 4.x script not found: {path}\n"
            f"This script only reuses the exact locked scripts already in this "
            f"repository's scripts/ directory -- it never substitutes another copy."
        )
    mod_name = f"_visual_abstract_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    log(f"[import:{name}] loaded module functions/constants from {path}")
    return module


def configure_section_paths(key, module):
    """Bind an imported Section 4.x module to its ACTUAL authoritative RA2
    project/data root before any scientific file is loaded.

    Sections 4.1-4.3 originally derive PROJECT from `__file__`, which is
    correct only when that script itself is the one running from its own
    checkout location -- once this visual-abstract script imports it from
    a different directory, that derivation resolves to the wrong root. So
    those three sections have their confirmed real path family rebound
    here explicitly. Sections 4.4-4.7 already hardcode their own correct
    RA2 root directly in the locked script; those are intentionally left
    untouched here (see validate_section_paths, which checks them for
    agreement with RA2_ROOTS instead of rebinding anything)."""
    if key == "multiscale":
        project = RA2_ROOTS["multiscale"]
        root = project / "4.1_final_all_metrics_gan_best_vs_poregen"
        module.PROJECT = project
        module.ROOT = root
        module.RESULTS = root / "results"
        module.GROUP_DIRS = {
            "real": root / "real_samples",
            "poregen": root / "poregen_samples",
            "gan_best": root / "gan_best_samples",
        }
        module.STAGE1_METRICS = (
            root / "results" / "stage1_core_morphology_topology" / "stage1_per_sample_metrics.csv"
        )
        module.STAGE2_CURVES = (
            root / "results" / "stage2_multiscale_curves" / "stage2_group_mean_curves.csv"
        )
        module.STAGE6_CURVES = (
            root / "results" / "stage6_capillary_porosimetry"
            / "stage6b_capillary_porosimetry_group_mean_curves.csv"
        )
        log(f"[configure:{key}] PROJECT={module.PROJECT}")
        log(f"[configure:{key}] ROOT={module.ROOT}")

    elif key == "2d_to_3d":
        project = RA2_ROOTS["2d_to_3d"]
        module.PROJECT = project
        module.METRICS_DIR = project / "evaluation_4_2_results_all_metrics"
        module.GROUP_VOLUME_DIRS = {
            "real": project / "evaluation_4_2_all_metrics_inputs" / "real",
            "microlad": project / "evaluation_4_2_all_metrics_inputs" / "diffusion",
            "slicegan": project / "evaluation_4_2_all_metrics_inputs" / "gan",
        }
        log(f"[configure:{key}] PROJECT={module.PROJECT}")

    elif key == "super_resolution":
        project = RA2_ROOTS["super_resolution"]
        module.PROJECT = project
        module.RESSHIFT_ROOT = project / "resshift_superres_4_3"
        module.HR_DIR = (
            module.RESSHIFT_ROOT / "data" / "processed" / "2d_x4_sandstone" / "test50" / "hr"
        )
        module.LR_DIR = (
            module.RESSHIFT_ROOT / "data" / "processed" / "2d_x4_sandstone" / "test50" / "lr"
        )
        module.RESSHIFT_DIR = (
            module.RESSHIFT_ROOT / "outputs" / "samples" / "final_50_2d_resshift_release"
        )
        module.SURVOL_ROOT = project / "survol_gan_4_3"
        module.SURVOL_DIR_EXPECTED = (
            module.SURVOL_ROOT / "outputs" / "samples" / "survol_xy_selected_epoch7_test50"
            / "predictions"
        )
        module.METRICS_DIR = project / "section43_superres_final_metrics"
        module.EVALUATOR_PATH = module.METRICS_DIR / "section43_full_metrics_2d.py"
        module.MAIN_TABLE_CSV = module.METRICS_DIR / "main_table_candidates.csv"
        module.PER_IMAGE_CSV = module.METRICS_DIR / "per_image_metrics.csv"
        log(f"[configure:{key}] PROJECT={module.PROJECT}")

    else:
        # Sections 4.4-4.7 already hardcode their own correct RA2 root
        # inside the locked script -- validated below, never rebound.
        log(f"[configure:{key}] no rebind needed (script already hardcodes its confirmed RA2 root)")


def validate_section_paths(key, module):
    """Return [(label, ok, path), ...] for this task's authoritative RA2
    inputs. Never raises by itself -- main()'s preflight aggregates every
    task's checks and raises exactly once, before any rendering, if
    anything required is missing."""
    checks = []

    def chk(label, path):
        p = Path(path)
        checks.append((label, p.exists(), str(p)))

    def chk_eq(label, actual, expected):
        checks.append((label, Path(actual) == Path(expected), str(actual)))

    if key == "multiscale":
        chk("PROJECT", module.PROJECT)
        chk("ROOT", module.ROOT)
        chk("real", module.GROUP_DIRS["real"])
        chk("diffusion", module.GROUP_DIRS["poregen"])
        chk("gan", module.GROUP_DIRS["gan_best"])
        chk("stage1", module.STAGE1_METRICS)

    elif key == "2d_to_3d":
        chk("PROJECT", module.PROJECT)
        chk("real", module.GROUP_VOLUME_DIRS["real"])
        chk("diffusion", module.GROUP_VOLUME_DIRS["microlad"])
        chk("gan", module.GROUP_VOLUME_DIRS["slicegan"])

    elif key == "super_resolution":
        chk("PROJECT", module.PROJECT)
        chk("hr", module.HR_DIR)
        chk("lr", module.LR_DIR)
        chk("resshift", module.RESSHIFT_DIR)
        chk("survol", module.SURVOL_DIR_EXPECTED)
        chk("per_image_csv", module.PER_IMAGE_CSV)

    elif key == "anisotropy":
        chk_eq("PROJECT", module.PROJECT, RA2_ROOTS["anisotropy"])
        chk("real", module.GROUP_VOLUME_DIRS["real"])
        chk("diffusion", module.GROUP_VOLUME_DIRS["diffusion"])
        chk("gan", module.GROUP_VOLUME_DIRS["gan"])

    elif key == "multiphase":
        chk_eq("PROJECT", module.PROJECT, RA2_ROOTS["multiphase"])
        chk("real", module.GROUP_VOLUME_DIRS["real"])
        chk("diffusion", module.GROUP_VOLUME_DIRS["diffusion"])
        chk("gan", module.GROUP_VOLUME_DIRS["gan"])

    elif key == "topology":
        chk_eq("PROJECT", module.PROJECT, RA2_ROOTS["topology"])
        chk("real", module.GROUP_VOLUME_DIRS["real"])
        chk("diffusion", module.GROUP_VOLUME_DIRS["diffusion"])
        chk("gan", module.GROUP_VOLUME_DIRS["gan"])

    elif key == "large_volume":
        chk_eq("PACKAGE_ROOT", module.PACKAGE_ROOT, RA2_ROOTS["large_volume"])
        chk("diffusion_folder", module.SAMPLE_DIRS_512["poredit"])
        chk("gan_folder", module.SAMPLE_DIRS_512["survol"])
        chk("diffusion_rep_file", module.REP_FILES["poredit"])
        chk("gan_rep_file", module.REP_FILES["survol"])

    else:
        raise ValueError(f"unknown task key: {key}")

    return checks


def add_card(fig, xywh, lw=CARD_LW):
    x, y, w, h = xywh
    fig.add_artist(Rectangle(
        (x, y), w, h, transform=fig.transFigure,
        facecolor=CARD_FACE, edgecolor=CARD_EDGE, linewidth=lw, zorder=-50,
    ))


def image_cell(ax, color, lw=CELL_LW):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(color)


def schematic_axis(fig, xywh):
    ax = fig.add_axes(xywh)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linestyle((0, (4, 3)))
        sp.set_linewidth(1.0)
        sp.set_color(SPINE)
    return ax


def draw_connector(fig, x0, y0, x1, y1):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0,
        linewidth=1.15, color=CONNECTOR_COLOR, alpha=0.85, zorder=-60,
    )
    fig.add_artist(arrow)


def draw_wireframe_cube(ax, cx, cy, size, color, lw=1.3):
    """Minimal pseudo-3D wireframe cube icon (vector primitive only)."""
    half = size / 2.0
    front = [(cx - half, cy - half), (cx + half, cy - half),
              (cx + half, cy + half), (cx - half, cy + half)]
    dx, dy = size * 0.34, size * 0.24
    back = [(x + dx, y + dy) for x, y in front]
    ax.add_patch(Polygon(front, closed=True, fill=False, edgecolor=color, linewidth=lw, zorder=3))
    ax.add_patch(Polygon(back, closed=True, fill=False, edgecolor=color, linewidth=lw * 0.8,
                          alpha=0.75, zorder=2))
    for (x0, y0), (x1, y1) in zip(front, back):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw * 0.8, alpha=0.75, zorder=2)


def draw_double_arrow(ax, x0, y0, x1, y1, color=SUBTEXT, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=lw, shrinkA=0, shrinkB=0))


def draw_single_arrow(ax, x0, y0, x1, y1, color=SUBTEXT, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw, shrinkA=0, shrinkB=0))


# ============================================================================
# 5. CHALLENGE SCHEMATICS -- pure vector primitives (no dataset required),
# except super-resolution's, which crops two small deterministic windows
# from the already-loaded real LR/HR representative pair (see section 6).
# ============================================================================


def schem_multiscale(ax):
    """Small field-of-view <-> heterogeneous larger structure."""
    ax.add_patch(Rectangle((0.10, 0.14), 0.60, 0.60, fill=False, edgecolor=SUBTEXT, linewidth=1.3))
    ax.add_patch(Rectangle((0.16, 0.20), 0.22, 0.22, fill=False, edgecolor=SUBTEXT, linewidth=1.3))
    draw_double_arrow(ax, 0.40, 0.31, 0.80, 0.31, color=SUBTEXT)


def schem_2d_to_3d(ax):
    """Flat 2D slice -> 3D volume."""
    ax.add_patch(Rectangle((0.06, 0.32), 0.30, 0.36, fill=False, edgecolor=SUBTEXT, linewidth=1.3))
    for frac in (0.42, 0.50, 0.58):
        ax.plot([0.09, 0.33], [0.32 + frac * 0.36, 0.32 + frac * 0.36],
                color=SPINE, linewidth=0.9)
    draw_single_arrow(ax, 0.40, 0.50, 0.56, 0.50, color=SUBTEXT)
    draw_wireframe_cube(ax, 0.76, 0.46, 0.28, SUBTEXT)


def schem_super_resolution(ax, lr_crop, hr_crop):
    """LR crop -> HR crop, both real deterministic windows of the
    already-loaded Section 4.3 representative pair (see prepare_super_resolution)."""
    ax_lr = ax.inset_axes([0.06, 0.24, 0.34, 0.52])
    ax_lr.imshow(lr_crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    image_cell(ax_lr, SUBTEXT, lw=1.1)
    ax_hr = ax.inset_axes([0.58, 0.24, 0.34, 0.52])
    ax_hr.imshow(hr_crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    image_cell(ax_hr, SUBTEXT, lw=1.1)
    draw_single_arrow(ax, 0.42, 0.50, 0.56, 0.50, color=SUBTEXT)


def schem_anisotropy(ax):
    """Direction-dependent structure: an oriented ellipse plus a long
    characteristic length along one axis and a short one transversely."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((0.5, 0.5), width=0.62, height=0.22, angle=0,
                          fill=False, edgecolor=SUBTEXT, linewidth=1.4))
    draw_double_arrow(ax, 0.16, 0.5, 0.84, 0.5, color=SUBTEXT)
    draw_double_arrow(ax, 0.5, 0.36, 0.5, 0.64, color=SUBTEXT)


def schem_multiphase(ax):
    """Three touching phase regions, colored with Figure 4.5's own
    categorical phase palette (pore/active/CBD) -- schematic only."""
    phase_colors = ["#3C1A5B", "#E0792A", "#F5E27A"]
    xs = [0.10, 0.40, 0.70]
    for x0, c in zip(xs, phase_colors):
        ax.add_patch(Rectangle((x0, 0.22), 0.22, 0.56, facecolor=c, edgecolor=SPINE, linewidth=0.9))


def schem_topology(ax):
    """Disconnected clusters <-> one spanning connected network."""
    # left: three small disconnected 2-node clusters (fixed, deterministic layout)
    clusters = [((0.10, 0.66), (0.19, 0.60)), ((0.08, 0.34), (0.17, 0.40)),
                ((0.22, 0.20), (0.28, 0.30))]
    for (x0, y0), (x1, y1) in clusters:
        ax.plot([x0, x1], [y0, y1], color=SUBTEXT, linewidth=1.1)
        ax.add_patch(Circle((x0, y0), 0.018, facecolor=SUBTEXT, edgecolor="none"))
        ax.add_patch(Circle((x1, y1), 0.018, facecolor=SUBTEXT, edgecolor="none"))
    draw_single_arrow(ax, 0.36, 0.45, 0.50, 0.45, color=SUBTEXT)
    # right: one small connected spanning network
    nodes = [(0.62, 0.62), (0.78, 0.68), (0.90, 0.52), (0.76, 0.42), (0.60, 0.34), (0.86, 0.28)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (2, 5)]
    for i, j in edges:
        (x0, y0), (x1, y1) = nodes[i], nodes[j]
        ax.plot([x0, x1], [y0, y1], color=SUBTEXT, linewidth=1.1)
    for x, y in nodes:
        ax.add_patch(Circle((x, y), 0.018, facecolor=SUBTEXT, edgecolor="none"))


def schem_large_volume(ax):
    """Small domain -> large domain, wireframe cubes only."""
    draw_wireframe_cube(ax, 0.20, 0.42, 0.16, SUBTEXT)
    draw_single_arrow(ax, 0.34, 0.46, 0.52, 0.48, color=SUBTEXT)
    draw_wireframe_cube(ax, 0.74, 0.46, 0.36, SUBTEXT)


# ============================================================================
# 6. SECTION-SPECIFIC ASSET PREPARATION -- each function reuses the exact
# representative-selection + rendering functions of its Section 4.x script,
# imported as a module. Nothing here re-derives or approximates that logic.
# ============================================================================


def _assert_locked_colors(module, diffusion_key, gan_key, section_name):
    d_col = module.COLORS[diffusion_key]
    g_col = module.COLORS[gan_key]
    if d_col != DIFFUSION or g_col != GAN:
        raise RuntimeError(
            f"[{section_name}] locked model-family colors have drifted from the paper-wide "
            f"convention: expected diffusion={DIFFUSION} gan={GAN}, got diffusion={d_col} "
            f"gan={g_col} in {module.__file__}. Stopping rather than silently using a "
            f"different color."
        )


def prepare_multiscale(m1):
    """Section 4.1 -- PoreGen/DiffSci vs True2Dto3Drecon 3D pore isosurfaces."""
    diffusion_key, gan_key = "poregen", "gan_best"
    _assert_locked_colors(m1, diffusion_key, gan_key, "multiscale")

    stage1, scol, _ = m1.load_stage1()
    rep = {}
    for g in m1.GROUPS:
        f = m1.choose_representative_file(g, stage1, scol)
        vol = m1.load_volume(f)
        rep[g] = {"file": f, "vol": vol}
        log(f"[multiscale:{g}] representative file = {f}")

    parallel_scale = 0.72 * max(max(rep[g]["vol"].shape) for g in m1.GROUPS)
    render_groups = [diffusion_key, gan_key]
    raw_paths = {g: TMP / f"multiscale_raw_{g}.png" for g in render_groups}
    png_paths = {g: TMP / f"multiscale_render_{g}.png" for g in render_groups}
    for g in render_groups:
        m1.render_volume(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    m1.finalize_renders(raw_paths, png_paths)

    return {
        "key": "multiscale",
        "diffusion": {"label": m1.LABELS[diffusion_key], "color": m1.COLORS[diffusion_key],
                      "image": Image.open(png_paths[diffusion_key]), "cmap": None,
                      "source": str(rep[diffusion_key]["file"])},
        "gan": {"label": m1.LABELS[gan_key], "color": m1.COLORS[gan_key],
                "image": Image.open(png_paths[gan_key]), "cmap": None,
                "source": str(rep[gan_key]["file"])},
        "render_mechanism": "make_fig4_1_composite_magma_final.render_volume "
                             "(PyVista marching-cubes pore isosurface, solid model color)",
        "schematic_fn": schem_multiscale,
        "schematic_source": "vector",
    }


def prepare_2d_to_3d(m2):
    """Section 4.2 -- MicroLad vs SliceGAN three-phase cutaway cubes."""
    diffusion_key, gan_key = "microlad", "slicegan"
    _assert_locked_colors(m2, diffusion_key, gan_key, "2d_to_3d")

    groups_data = {}
    for g in m2.GROUPS:
        gd = m2.load_group(g)
        m2.check_phase_fractions(g, gd["mean_fracs"])
        groups_data[g] = gd

    rep = {}
    for g in m2.GROUPS:
        f, vol = m2.choose_representative(g, groups_data[g])
        rep[g] = {"file": f, "vol": vol}
        log(f"[2d_to_3d:{g}] representative file = {f}")

    render_groups = [diffusion_key, gan_key]
    parallel_scale = 0.60 * max(max(rep[g]["vol"].shape) for g in m2.GROUPS)
    raw_paths = {g: TMP / f"2d3d_raw_{g}.png" for g in render_groups}
    png_paths = {g: TMP / f"2d3d_render_{g}.png" for g in render_groups}
    for g in render_groups:
        m2.render_group_cube(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    m2.finalize_renders(raw_paths, png_paths)

    return {
        "key": "2d_to_3d",
        "diffusion": {"label": m2.LABELS[diffusion_key], "color": m2.COLORS[diffusion_key],
                      "image": Image.open(png_paths[diffusion_key]), "cmap": None,
                      "source": str(rep[diffusion_key]["file"])},
        "gan": {"label": m2.LABELS[gan_key], "color": m2.COLORS[gan_key],
                "image": Image.open(png_paths[gan_key]), "cmap": None,
                "source": str(rep[gan_key]["file"])},
        "render_mechanism": "make_fig4_2_composite_magma_final.render_group_cube "
                             "(three-phase cutaway-cube render, categorical phase colors "
                             "unchanged -- model identity carried only by border/title)",
        "schematic_fn": schem_2d_to_3d,
        "schematic_source": "vector",
    }


def prepare_super_resolution(m3):
    """Section 4.3 -- ResShift vs SurVol paired 2D super-resolution outputs."""
    diffusion_key, gan_key = "resshift", "survol"
    _assert_locked_colors(m3, diffusion_key, gan_key, "super_resolution")

    survol_dir = m3.resolve_survol_dir()
    hr_files = m3.check_dir("hr", m3.HR_DIR, m3.EXPECTED_N, m3.EXPECTED_HR_SIZE)
    lr_files = m3.check_dir("lr", m3.LR_DIR, m3.EXPECTED_N, m3.EXPECTED_LR_SIZE)
    resshift_files = m3.check_dir("resshift", m3.RESSHIFT_DIR, m3.EXPECTED_N, m3.EXPECTED_HR_SIZE)
    survol_files = m3.check_dir("survol", survol_dir, m3.EXPECTED_N, m3.EXPECTED_HR_SIZE)

    hr_ids, survol_files_ordered, align_method = m3.build_id_maps(
        hr_files, lr_files, resshift_files, survol_files)
    log(f"[super_resolution] SurVol alignment method: {align_method}")

    rep_id, rep_score = m3.choose_representative_image(hr_ids)
    rep_idx = hr_ids.index(rep_id)
    log(f"[super_resolution] representative id = {rep_id} (score={rep_score})")

    hr_full = m3.load_gray(hr_files[rep_idx])
    lr_full = m3.load_gray(lr_files[rep_idx])
    resshift_full = m3.load_gray(resshift_files[rep_idx])
    survol_full = m3.load_gray(survol_files_ordered[rep_idx])

    # Deterministic, fixed-size, top-left crops of the same real representative
    # pair -- never manually chosen -- for the fully grounded LR->HR schematic.
    hr_crop = hr_full[0:96, 0:96]
    lr_crop_small = lr_full[0:24, 0:24]
    lr_crop = np.array(Image.fromarray(lr_crop_small).resize((96, 96), Image.NEAREST))

    return {
        "key": "super_resolution",
        "diffusion": {"label": m3.LABELS[diffusion_key], "color": m3.COLORS[diffusion_key],
                      "image": resshift_full, "cmap": "gray",
                      "source": str(resshift_files[rep_idx])},
        "gan": {"label": m3.LABELS[gan_key], "color": m3.COLORS[gan_key],
                "image": survol_full, "cmap": "gray",
                "source": str(survol_files_ordered[rep_idx])},
        "render_mechanism": "make_fig4_3_composite_magma_final grayscale PNG load (load_gray) "
                             "of the locked representative ResShift/SurVol outputs -- no 3D render",
        "schematic_fn": lambda ax: schem_super_resolution(ax, lr_crop, hr_crop),
        "schematic_source": f"real Section 4.3 representative crop (hr={hr_files[rep_idx]}, "
                             f"lr={lr_files[rep_idx]}, top-left 96x96/24x24 deterministic window)",
    }


def prepare_anisotropy(m4):
    """Section 4.4 -- AB-CDM vs SurVol 3D isosurfaces (COMMON100 glass)."""
    diffusion_key, gan_key = "diffusion", "gan"
    _assert_locked_colors(m4, diffusion_key, gan_key, "anisotropy")

    rep = {}
    render_groups = [diffusion_key, gan_key]
    for g in render_groups:
        volumes = m4.load_group_volumes(g)
        rows = m4.compute_group_descriptors(volumes)
        f, _dmin = m4.choose_representative(g, rows)
        rep[g] = {"file": f, "vol": volumes[f]}
        log(f"[anisotropy:{g}] representative file = {f}")

    parallel_scale = 0.72 * max(max(rep[g]["vol"].shape) for g in render_groups)
    raw_paths = {g: TMP / f"aniso_raw_{g}.png" for g in render_groups}
    png_paths = {g: TMP / f"aniso_render_{g}.png" for g in render_groups}
    for g in render_groups:
        m4.render_group_isosurface(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    m4.finalize_renders(raw_paths, png_paths)

    return {
        "key": "anisotropy",
        "diffusion": {"label": m4.LABELS[diffusion_key], "color": m4.COLORS[diffusion_key],
                      "image": Image.open(png_paths[diffusion_key]), "cmap": None,
                      "source": str(rep[diffusion_key]["file"])},
        "gan": {"label": m4.LABELS[gan_key], "color": m4.COLORS[gan_key],
                "image": Image.open(png_paths[gan_key]), "cmap": None,
                "source": str(rep[gan_key]["file"])},
        "render_mechanism": "make_fig4_4_composite_magma_final.render_group_isosurface "
                             "(PyVista marching-cubes phase-1 isosurface, solid model color, "
                             "raises loudly on failure -- no fallback)",
        "schematic_fn": schem_anisotropy,
        "schematic_source": "vector",
    }


def prepare_multiphase(m5):
    """Section 4.5 -- MicroGen3D vs SurVol 3D multiphase cutaways (Muller)."""
    diffusion_key, gan_key = "diffusion", "gan"
    _assert_locked_colors(m5, diffusion_key, gan_key, "multiphase")

    volumes_by_group, descriptors_by_group = {}, {}
    for g in m5.GROUPS:
        volumes = m5.load_group_volumes(g)
        volumes_by_group[g] = volumes
        rows = m5.compute_group_descriptors(volumes)
        descriptors_by_group[g] = rows
        direct_mean = {k: float(np.mean([r[k] for r in rows])) for k in ("pore", "active", "cbd")}
        direct_txyz = (float(np.mean([r["tx"] for r in rows])),
                       float(np.mean([r["ty"] for r in rows])),
                       float(np.mean([r["tz"] for r in rows])))
        m5.check_phase_fractions(g, direct_mean)
        m5.check_transitions(g, direct_txyz)

    real_row, real_mean_vec = m5.choose_representative_real(descriptors_by_group["real"])
    rep = {}
    for g in (diffusion_key, gan_key):
        r = m5.choose_representative_generated(g, descriptors_by_group[g], real_mean_vec)
        rep[g] = {"file": r["file"], "vol": volumes_by_group[g][r["file"]]}
        log(f"[multiphase:{g}] representative file = {r['file']}")

    render_groups = [diffusion_key, gan_key]
    parallel_scale = 0.82 * max(max(rep[g]["vol"].shape) for g in render_groups)
    raw_paths = {g: TMP / f"multiphase_raw_{g}.png" for g in render_groups}
    png_paths = {g: TMP / f"multiphase_render_{g}.png" for g in render_groups}
    for g in render_groups:
        m5.render_group_multiphase(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    m5.finalize_renders(raw_paths, png_paths)

    return {
        "key": "multiphase",
        "diffusion": {"label": m5.LABELS[diffusion_key], "color": m5.COLORS[diffusion_key],
                      "image": Image.open(png_paths[diffusion_key]), "cmap": None,
                      "source": str(rep[diffusion_key]["file"])},
        "gan": {"label": m5.LABELS[gan_key], "color": m5.COLORS[gan_key],
                "image": Image.open(png_paths[gan_key]), "cmap": None,
                "source": str(rep[gan_key]["file"])},
        "render_mechanism": "make_fig4_5_composite_magma_final.render_group_multiphase "
                             "(solid per-voxel-phase cutaway cube, categorical phase colors "
                             "unchanged -- model identity carried only by border/title)",
        "schematic_fn": schem_multiphase,
        "schematic_source": "vector",
    }


def prepare_topology(m6):
    """Section 4.6 -- PoreGen/DiffSci vs IPWGAN 3D topology renders."""
    diffusion_key, gan_key = "diffusion", "gan"
    _assert_locked_colors(m6, diffusion_key, gan_key, "topology")

    volumes_by_group = {}
    for g in m6.GROUPS:
        volumes = m6.load_group_volumes(g)
        volumes_by_group[g] = volumes
        fracs = [m6.pore_fraction(v) for v in volumes.values()]
        m6.check_pore_fraction(g, float(np.mean(fracs)))
        comp_means = m6.group_component_means(volumes)
        m6.check_component_sanity(g, comp_means)

    representative_files, _candidates = m6.select_representative_samples(volumes_by_group)

    rep = {}
    render_groups = [diffusion_key, gan_key]
    for g in render_groups:
        f = representative_files[g]
        v = volumes_by_group[g][f]
        stats = m6.component_stats(v)
        rep[g] = {"file": f, "vol": v, "labeled": stats["labeled"], "largest_id": stats["largest_id"]}
        log(f"[topology:{g}] representative file = {f}")

    parallel_scale = 0.82 * max(max(rep[g]["vol"].shape) for g in render_groups)
    raw_paths = {g: TMP / f"topology_raw_{g}.png" for g in render_groups}
    png_paths = {g: TMP / f"topology_render_{g}.png" for g in render_groups}
    for g in render_groups:
        m6.render_group_topology(rep[g]["labeled"], rep[g]["largest_id"], g, raw_paths[g],
                                  parallel_scale)
    m6.finalize_renders(raw_paths, png_paths)

    return {
        "key": "topology",
        "diffusion": {"label": m6.LABELS[diffusion_key], "color": m6.COLORS[diffusion_key],
                      "image": Image.open(png_paths[diffusion_key]), "cmap": None,
                      "source": str(rep[diffusion_key]["file"])},
        "gan": {"label": m6.LABELS[gan_key], "color": m6.COLORS[gan_key],
                "image": Image.open(png_paths[gan_key]), "cmap": None,
                "source": str(rep[gan_key]["file"])},
        "render_mechanism": "make_fig4_6_composite_magma_final.render_group_topology "
                             "(full pore topology; largest connected component and disconnected "
                             "fragments colored by topology class, not by model identity)",
        "schematic_fn": schem_topology,
        "schematic_source": "vector",
    }


def prepare_large_volume(m7):
    """Section 4.7 -- PoreDiT vs SurVol 512^3 pore-phase isosurfaces."""
    diffusion_key, gan_key = "poredit", "survol"
    _assert_locked_colors(m7, diffusion_key, gan_key, "large_volume")

    render_groups = [diffusion_key, gan_key]
    rep = {}
    for g in render_groups:
        m7.validate_visualization_folder(g, m7.SAMPLE_DIRS_512[g])
        vol, phi = m7.load_and_validate_representative(g, m7.REP_FILES[g])
        rep[g] = {"vol": vol, "phi": phi, "file": m7.REP_FILES[g]}
        log(f"[large_volume:{g}] representative file (locked ordinal {m7.REP_ORDINAL}) = "
            f"{m7.REP_FILES[g]}")
    m7.validate_qc_manifest({g: m7.REP_FILES[g] for g in render_groups})

    parallel_scale = 0.72 * max(rep[g]["vol"].shape[0] for g in render_groups)
    raw_paths = {g: TMP / f"largevol_raw_{g}.png" for g in render_groups}
    png_paths = {g: TMP / f"largevol_render_{g}.png" for g in render_groups}
    for g in render_groups:
        m7.render_group_isosurface(rep[g]["vol"], g, raw_paths[g], parallel_scale)
    m7.finalize_renders(raw_paths, png_paths)

    return {
        "key": "large_volume",
        "diffusion": {"label": m7.LABELS[diffusion_key], "color": m7.COLORS[diffusion_key],
                      "image": Image.open(png_paths[diffusion_key]), "cmap": None,
                      "source": str(rep[diffusion_key]["file"])},
        "gan": {"label": m7.LABELS[gan_key], "color": m7.COLORS[gan_key],
                "image": Image.open(png_paths[gan_key]), "cmap": None,
                "source": str(rep[gan_key]["file"])},
        "render_mechanism": "make_fig4_7_composite_magma_final.render_group_isosurface "
                             "(512^3 pore-phase isosurface, fixed block-occupancy downsampling, "
                             "locked ordinal-32 representative, raises loudly on failure)",
        "schematic_fn": schem_large_volume,
        "schematic_source": "vector",
    }


PREPARE_FUNCS = {
    "multiscale": prepare_multiscale,
    "2d_to_3d": prepare_2d_to_3d,
    "super_resolution": prepare_super_resolution,
    "anisotropy": prepare_anisotropy,
    "multiphase": prepare_multiphase,
    "topology": prepare_topology,
    "large_volume": prepare_large_volume,
}

# ============================================================================
# 7. TASK CARD BUILDER
# ============================================================================


def build_task_card(fig, xywh, title, task):
    x0, y0, w, h = xywh
    add_card(fig, xywh)

    title_y = y0 + h - 0.028
    fig.text(x0 + w / 2.0, title_y, title, ha="center", va="top",
              fontsize=10.2, fontweight="bold", color=TEXT)

    schem_h = 0.30 * h
    schem_y0 = title_y - 0.024 - schem_h
    schem_x0 = x0 + 0.06 * w
    schem_w = w - 0.12 * w
    ax_schem = schematic_axis(fig, [schem_x0, schem_y0, schem_w, schem_h])
    task["schematic_fn"](ax_schem)

    label_h = 0.05 * h
    img_top = schem_y0 - 0.03 * h
    label_y0 = img_top - label_h
    img_h = 0.34 * h
    img_y0 = label_y0 - 0.01 * h - img_h

    pad_x = 0.07 * w
    cell_gap = 0.05 * w
    cell_w = (w - 2 * pad_x - cell_gap) / 2.0
    left_x0 = x0 + pad_x
    right_x0 = left_x0 + cell_w + cell_gap

    for cell_x0, side in ((left_x0, "diffusion"), (right_x0, "gan")):
        info = task[side]
        fig.text(cell_x0 + cell_w / 2.0, label_y0 + label_h, info["label"],
                  ha="center", va="top", fontsize=8.4, fontweight="bold", color=info["color"])
        ax = fig.add_axes([cell_x0, img_y0, cell_w, img_h])
        if info["cmap"] is not None:
            ax.imshow(info["image"], cmap=info["cmap"], vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(info["image"], interpolation="bilinear")
        image_cell(ax, info["color"], lw=CELL_LW)


# ============================================================================
# 8. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    log(f"[paths] REPO = {REPO}")
    log(f"[paths] OUT  = {OUT}")

    # ---- Phase A: import + configure all seven Section modules -------------
    modules = {}
    for key in TOP_ROW + BOTTOM_ROW:
        script_path = SECTION_SCRIPTS[key]
        module = import_section_module(key, script_path)
        configure_section_paths(key, module)
        modules[key] = module

    # ---- Phase B: validate EVERY task's authoritative RA2 inputs before ----
    # any scientific loading/rendering starts, so a broken path on task 6 or
    # 7 is caught immediately rather than after tasks 1-5 already rendered.
    log("\n=== RA2 scientific-source preflight ===")
    all_checks = {}
    any_failed = False
    for key in TOP_ROW + BOTTOM_ROW:
        checks = validate_section_paths(key, modules[key])
        all_checks[key] = checks
        log(f"\n[{key}]")
        for label, ok, path in checks:
            status = "OK" if ok else "MISSING"
            log(f"  {label:<18} {status:<7} {path}")
            if not ok:
                any_failed = True

    if any_failed:
        lines = ["RA2 source preflight FAILED.", "Missing:"]
        for key, checks in all_checks.items():
            for label, ok, path in checks:
                if not ok:
                    lines.append(f"  [{key}] {label}: {path}")
        raise RuntimeError("\n".join(lines))

    log(f"\nPRECHECK PASSED: all {len(TOP_ROW) + len(BOTTOM_ROW)} tasks\n")

    # ---- Phase C: prepare/render all seven tasks ----------------------------
    tasks = {}
    audit_tasks = {}
    for key in TOP_ROW + BOTTOM_ROW:
        log(f"\n=== task: {key}  (source script: {SECTION_SCRIPTS[key].name}) ===")
        task = PREPARE_FUNCS[key](modules[key])
        tasks[key] = task
        log(f"[{key}] diffusion = '{task['diffusion']['label']}'  source = {task['diffusion']['source']}")
        log(f"[{key}] gan       = '{task['gan']['label']}'  source = {task['gan']['source']}")
        log(f"[{key}] render mechanism: {task['render_mechanism']}")
        log(f"[{key}] schematic source: {task['schematic_source']}")
        audit_tasks[key] = {
            "title": TASK_TITLES[key],
            "diffusion_label": task["diffusion"]["label"],
            "diffusion_source": task["diffusion"]["source"],
            "gan_label": task["gan"]["label"],
            "gan_source": task["gan"]["source"],
            "render_mechanism": task["render_mechanism"],
            "schematic_source": task["schematic_source"],
        }

    # ---- canvas --------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.patch.set_facecolor(BG)

    hub_cx, hub_cy = HUB_X0 + HUB_W / 2.0, HUB_Y0 + HUB_H / 2.0
    for key in TOP_ROW:
        cx, cy, cw, ch = TASK_CARDS[key]
        draw_connector(fig, hub_cx, hub_cy + HUB_H / 2.0, cx + cw / 2.0, cy)
    for key in BOTTOM_ROW:
        cx, cy, cw, ch = TASK_CARDS[key]
        draw_connector(fig, hub_cx, hub_cy - HUB_H / 2.0, cx + cw / 2.0, cy + ch)

    for key in TOP_ROW + BOTTOM_ROW:
        build_task_card(fig, TASK_CARDS[key], TASK_TITLES[key], tasks[key])

    add_card(fig, HUB_CARD, lw=HUB_LW)
    fig.text(hub_cx, hub_cy, "Task-oriented benchmark", ha="center", va="center",
              fontsize=13.0, fontweight="bold", color=TEXT)

    # =========================== SAVE ========================================
    png = OUT / f"{STEM}.png"
    pdf = OUT / f"{STEM}.pdf"
    svg = OUT / f"{STEM}.svg"

    fig.savefig(png, dpi=450, facecolor=BG)     # no bbox_inches -> geometry preserved
    fig.savefig(pdf, facecolor=BG)
    fig.savefig(svg, facecolor=BG)
    plt.close(fig)

    # =========================== AUDIT JSON ==================================
    metadata = {
        "script": Path(__file__).name,
        "ra2_roots": {k: str(v) for k, v in RA2_ROOTS.items()},
        "section_scripts": {k: str(v) for k, v in SECTION_SCRIPTS.items()},
        "task_order": {"top_row": TOP_ROW, "bottom_row": BOTTOM_ROW},
        "preflight": {
            key: [{"label": label, "ok": ok, "path": path} for label, ok, path in checks]
            for key, checks in all_checks.items()
        },
        "tasks": audit_tasks,
        "saved": {"png": str(png), "pdf": str(pdf), "svg": str(svg)},
        "log": RUN_LOG,
    }
    AUDIT_JSON.write_text(json.dumps(metadata, indent=2, default=str))

    log("\nSaved:")
    for p in (png, pdf, svg, AUDIT_JSON):
        log(" ", p.resolve())


if __name__ == "__main__":
    sys.exit(main())
