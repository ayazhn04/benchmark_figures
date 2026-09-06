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
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, to_rgb
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

# Stronger hub/title typography chrome (Requirement 4) -- a light tint,
# never a saturated fill, so contrast improves without competing with the
# diffusion/GAN cell colors.
HUB_FACE = "#F4EEFB"
TITLE_CHIP_FACE = "#F1E9FA"

# Locked model-family colors (identical to the "diffusion"/"gan" colors
# already used throughout Figures 4.1-4.7).
DIFFUSION = "#F2A93B"
GAN = "#772A8E"

# Neutral annotation color for schematic ROI/subvolume/bounding-box
# markers -- deliberately NOT orange or purple, since those are reserved
# throughout the paper for diffusion/GAN model identity and must never
# appear on a challenge-schematic annotation that has nothing to do with
# either model family.
ANNOTATION_COLOR = "#3B7EA1"

CARD_LW = 0.9
HUB_LW = 1.4
CELL_LW = 1.8
CONNECTOR_COLOR = "#9483B0"

# Restrained gray for the small in-schematic helper labels ("small FOV",
# "LR"/"HR", "phase 1", ...) -- support text, never competing with the
# task title or the model-name labels.
HELPER_GRAY = "#948C9E"
HELPER_FONTSIZE = 5.6

FIG_W, FIG_H = 17.8, 10.2

# ============================================================================
# 3. GEOMETRY -- hub + 3-top / 4-bottom task grid, all in figure fractions
# ============================================================================

MARGIN = 0.03
GAP = 0.022

# Deliberate breathing room between the hub and each task row (both the
# hub<->top-row gap and the hub<->bottom-row gap), so the hub reads as
# clearly central rather than squeezed between the rows. The connector
# "bus" lines (see draw_hub_tree) are routed through the middle of this
# gap.
HUB_GAP = 0.052

HUB_W, HUB_H = 0.168, 0.100
FULL_W = 1.0 - 2 * MARGIN

# ROW_H is derived so the full vertical budget (margin + row + hub-gap +
# hub + hub-gap + row + margin) exactly fills the page -- increasing
# HUB_GAP automatically (and only) shrinks the task rows, never overlaps.
ROW_H = (1.0 - 2 * MARGIN - 2 * HUB_GAP - HUB_H) / 2.0

BOTTOM_Y0 = MARGIN
TOP_Y0 = 1.0 - MARGIN - ROW_H

HUB_X0 = 0.5 - HUB_W / 2.0
HUB_Y0 = 0.5 - HUB_H / 2.0
HUB_CARD = [HUB_X0, HUB_Y0, HUB_W, HUB_H]

# Every one of the seven task panels shares one identical card width and
# height -- sized off the tighter bottom row (4 cards) so the 3-card top
# row is never stretched to fill the page width. The top row is instead
# centered on the page, with equal leftover margin on each side. Uniform
# panel size is prioritized over the top row spanning the full width.
N_BOTTOM = len(BOTTOM_ROW)
CARD_W = (FULL_W - (N_BOTTOM - 1) * GAP) / N_BOTTOM
CARD_H = ROW_H

# Inner horizontal padding shared identically by the schematic row and the
# render-cell row of every panel, so left/right whitespace reads as one
# consistent rhythm rather than two slightly different insets.
PANEL_INNER_PAD = 0.065


def _centered_row_cards(y0, keys):
    n = len(keys)
    row_w = n * CARD_W + (n - 1) * GAP
    x0_start = 0.5 - row_w / 2.0
    cards = []
    for i in range(n):
        x0 = x0_start + i * (CARD_W + GAP)
        cards.append([x0, y0, CARD_W, CARD_H])
    return cards


TOP_CARDS = dict(zip(TOP_ROW, _centered_row_cards(TOP_Y0, TOP_ROW)))
BOTTOM_CARDS = dict(zip(BOTTOM_ROW, _centered_row_cards(BOTTOM_Y0, BOTTOM_ROW)))
TASK_CARDS = {**TOP_CARDS, **BOTTOM_CARDS}

assert abs(HUB_Y0 - (BOTTOM_Y0 + ROW_H + HUB_GAP)) < 1e-9, \
    "hub<->bottom-row gap does not match HUB_GAP"
assert abs((HUB_Y0 + HUB_H) - (TOP_Y0 - HUB_GAP)) < 1e-9, \
    "hub<->top-row gap does not match HUB_GAP"

# Internal top-to-bottom slot ordering inside each task card (Requirement 2):
# the top row reads renders -> schematic -> title (title nearest the hub,
# which sits below the top row); the bottom row reads title -> schematic ->
# renders (title nearest the hub, which sits above the bottom row).
ORDER_TOP = ["renders", "schematic", "title"]
ORDER_BOTTOM = ["title", "schematic", "renders"]

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
        chk("real_folder", module.SAMPLE_DIRS_512["real"])
        chk("diffusion_rep_file", module.REP_FILES["poredit"])
        chk("gan_rep_file", module.REP_FILES["survol"])
        chk("real_rep_file", module.REP_FILES["real"])

    else:
        raise ValueError(f"unknown task key: {key}")

    return checks


def add_card(fig, xywh, lw=CARD_LW, facecolor=CARD_FACE):
    x, y, w, h = xywh
    fig.add_artist(Rectangle(
        (x, y), w, h, transform=fig.transFigure,
        facecolor=facecolor, edgecolor=CARD_EDGE, linewidth=lw, zorder=-50,
    ))


def image_cell(ax, color, lw=CELL_LW):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(color)


def schematic_axis(fig, xywh):
    """The dashed challenge-schematic box -- deliberately light/thin so it
    reads as secondary support for the title next to it, never competing
    with the model-render area for visual weight."""
    ax = fig.add_axes(xywh)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linestyle((0, (5, 4)))
        sp.set_linewidth(0.7)
        sp.set_color(SPINE)
    return ax


# ----------------------------------------------------------------------
# Shared internal coordinate system for every challenge schematic, so
# margins/arrow length/spacing/vertical centering/label placement read as
# one consistent design system rather than per-panel tuning:
#
#   | SCHEM_PAD | VISUAL | SCHEM_GAP | ARROW (SCHEM_ARROW_LEN) | SCHEM_GAP | VISUAL | SCHEM_PAD |
#
# with every visual box vertically centered on SCHEM_VCENTER, and a
# dedicated bottom SCHEM_LABEL_BAND reserved for helper labels so they
# never sit at the schematic's bare bottom edge (and thus never touch the
# dashed border).
# ----------------------------------------------------------------------
SCHEM_PAD = 0.05
SCHEM_GAP = 0.035
SCHEM_ARROW_LEN = 0.085
SCHEM_BOX_H = 0.62
SCHEM_LABEL_BAND = 0.20
SCHEM_TOP_PAD = 0.07
SCHEM_VCENTER = SCHEM_LABEL_BAND + (1.0 - SCHEM_LABEL_BAND - SCHEM_TOP_PAD) / 2.0
SCHEM_LABEL_Y = SCHEM_LABEL_BAND / 2.0


def helper_label(ax, x, text):
    """A small, restrained gray in-schematic caption ("small FOV", "LR",
    "phase 1", ...), always placed at the same y inside the shared bottom
    label band (SCHEM_LABEL_Y) -- never at the schematic's bare bottom
    edge, so it never touches the dashed border, and every label in the
    figure sits on one consistent baseline."""
    ax.text(x, SCHEM_LABEL_Y, text, ha="center", va="center", fontsize=HELPER_FONTSIZE,
            color=HELPER_GRAY, transform=ax.transAxes)


def schem_layout_row(ax, box_widths, box_h=SCHEM_BOX_H, arrows=True, arrow_color=SUBTEXT):
    """Lay out `len(box_widths)` visual boxes left-to-right, each `box_h`
    tall and vertically centered on SCHEM_VCENTER, as one group centered
    within the shared [SCHEM_PAD, 1-SCHEM_PAD] margin -- never stretched
    to fill leftover space. When `arrows` is True, a fixed-length arrow
    (SCHEM_ARROW_LEN, flanked by SCHEM_GAP on each side) connects every
    consecutive pair; when False (simultaneous/parallel views, not a
    transformation), boxes are simply separated by 2*SCHEM_GAP. Returns
    the [x0, y0, w, h] inset-box for each entry; arrows are drawn here."""
    n = len(box_widths)
    n_gaps = n - 1
    gap_w = (2 * SCHEM_GAP + SCHEM_ARROW_LEN) if arrows else 2 * SCHEM_GAP
    total_w = sum(box_widths) + n_gaps * gap_w
    avail_w = 1.0 - 2 * SCHEM_PAD
    x0 = SCHEM_PAD + max(0.0, (avail_w - total_w) / 2.0)
    y0 = SCHEM_VCENTER - box_h / 2.0
    boxes = []
    cursor = x0
    for i, bw in enumerate(box_widths):
        boxes.append([cursor, y0, bw, box_h])
        cursor += bw
        if i < n_gaps:
            if arrows:
                a0 = cursor + SCHEM_GAP
                a1 = a0 + SCHEM_ARROW_LEN
                draw_single_arrow(ax, a0, SCHEM_VCENTER, a1, SCHEM_VCENTER, color=arrow_color)
                cursor = a1 + SCHEM_GAP
            else:
                cursor += gap_w
    return boxes


CONNECTOR_ZORDER = -40  # above the white cards (-50) so nothing is clipped/hidden


def _connector_line(fig, points):
    """Plain polyline segment (no arrowhead) -- the shared trunk/bus part
    of a routed hub<->row connector tree."""
    xs, ys = zip(*points)
    fig.add_artist(Line2D(
        xs, ys, transform=fig.transFigure, color=CONNECTOR_COLOR,
        linewidth=1.2, alpha=0.9, solid_capstyle="round", zorder=CONNECTOR_ZORDER,
    ))


def _connector_drop(fig, x, y0, y1):
    """The final leg of a routed connector: a short straight drop from the
    shared bus line into the panel's title-side edge, arrowhead landing
    exactly on that edge."""
    arrow = FancyArrowPatch(
        (x, y0), (x, y1), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=7.5, shrinkA=0, shrinkB=0.5,
        linewidth=1.2, color=CONNECTOR_COLOR, alpha=0.9, zorder=CONNECTOR_ZORDER,
    )
    fig.add_artist(arrow)


def draw_hub_tree(fig, hub_cx, hub_edge_y, bus_y, panel_cxs, panel_edge_ys):
    """One consistent elbow-routed connector tree per row: a single trunk
    from the hub edge to a shared horizontal bus, then one short vertical
    drop (with arrowhead) from that bus into each panel's title-side edge.
    Replaces N separate spokes/arcs converging on the hub -- here only one
    line ever touches the hub on each side, so there is no clutter at the
    center, no ambiguity about routing, and every drop is a plain straight
    segment terminating cleanly on the panel border (Requirement: clean,
    consistent, elegantly routed connectors)."""
    _connector_line(fig, [(hub_cx, hub_edge_y), (hub_cx, bus_y)])
    _connector_line(fig, [(min(panel_cxs), bus_y), (max(panel_cxs), bus_y)])
    for cx, edge_y in zip(panel_cxs, panel_edge_ys):
        _connector_drop(fig, cx, bus_y, edge_y)


def draw_single_arrow(ax, x0, y0, x1, y1, color=SUBTEXT, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw, shrinkA=0, shrinkB=0))


def binary_slice_u8(vol, axis_index=None):
    """A real 2D cross-section of an already-loaded (binary pore/solid)
    Section 4.x reference volume, as a plain 0/255 uint8 image -- the exact
    voxels of the real representative sample, no smoothing or fabrication.
    Works regardless of the array's dtype (bool/int/float 0-1)."""
    idx = vol.shape[0] // 2 if axis_index is None else axis_index
    sl = np.asarray(vol[idx])
    return (sl.astype(bool).astype(np.uint8)) * 255


def choose_grounded_roi(slice01, size, occ_lo=0.20, occ_hi=0.80):
    """Deterministically choose a `size`x`size` ROI window of a real
    binary (0/1) 2D slice that actually contains both phases: scan a fixed
    grid of candidate windows (stride = size//2, row-major order -- never
    manually eyeballed), reject any candidate whose phase-1 occupancy
    falls outside [occ_lo, occ_hi], and among the surviving candidates
    pick the one whose occupancy is closest to the WHOLE slice's own
    global phase fraction (ties broken by scan order, so the result is
    fully deterministic and reproducible). Falls back to the closest-to-
    target candidate from the full grid only if literally none of them
    satisfy the occupancy bound. Returns (row0, col0, height, width,
    occupancy)."""
    h, w = slice01.shape
    size = max(4, min(size, h, w))
    global_frac = float(np.mean(slice01 > 0))
    stride = max(1, size // 2)
    candidates = []
    for r0 in range(0, h - size + 1, stride):
        for c0 in range(0, w - size + 1, stride):
            frac = float(np.mean(slice01[r0:r0 + size, c0:c0 + size] > 0))
            candidates.append((r0, c0, frac))
    valid = [c for c in candidates if occ_lo < c[2] < occ_hi]
    pool = valid if valid else candidates
    r0, c0, frac = min(pool, key=lambda c: (abs(c[2] - global_frac), c[0], c[1]))
    return r0, c0, size, size, frac


def choose_high_variance_roi(img2d, size, stride=None):
    """Deterministically choose a `size`x`size` window of a real 2D image
    with the highest local pixel variance among a fixed candidate grid --
    a standard structure-richness proxy -- so a schematic ROI is neither
    manually eyeballed nor a flat, uninformative patch. Ties broken by
    scan order (row-major) for full determinism. Returns (row0, col0)."""
    h, w = img2d.shape
    size = min(size, h, w)
    stride = stride or max(1, size // 2)
    best = None
    for r0 in range(0, h - size + 1, stride):
        for c0 in range(0, w - size + 1, stride):
            var = float(img2d[r0:r0 + size, c0:c0 + size].astype(np.float64).var())
            if best is None or var > best[2]:
                best = (r0, c0, var)
    return best[0], best[1]


def orthogonal_binary_slices(vol):
    """Three real orthogonal mid-slices of an already-loaded binary
    pore/solid Section 4.x reference volume -- one cut along each array
    axis of the SAME real sample (labeled XY/XZ/YZ, the standard labeling
    for the three canonical orthogonal cross-sections of a 3D array).
    Used to show real direction-dependent morphology directly, with no
    derived metric and no invented pattern."""
    d0, d1, d2 = vol.shape
    xy = np.asarray(vol[d0 // 2, :, :])
    xz = np.asarray(vol[:, d1 // 2, :])
    yz = np.asarray(vol[:, :, d2 // 2])
    return tuple(s.astype(bool).astype(np.uint8) for s in (xy, xz, yz))


def phase_mask_rgba(slice2d, phase_id, color_hex, bg_alpha=0.12):
    """A single real segmentation phase isolated from an already-loaded
    Section 4.x categorical slice: pixels of `phase_id` painted in that
    phase's own locked color, everything else left faint -- real labeled
    pixels only, never an invented/synthetic shape."""
    rgba = np.ones(slice2d.shape + (4,), dtype=np.float64)
    rgba[..., 3] = bg_alpha
    r, g, b = to_rgb(color_hex)
    mask = slice2d == phase_id
    rgba[mask, 0] = r
    rgba[mask, 1] = g
    rgba[mask, 2] = b
    rgba[mask, 3] = 1.0
    return rgba


# ============================================================================
# 5. CHALLENGE SCHEMATICS -- every schematic below is built either from a
# real crop/render of an already-loaded Section 4.x reference ("real") group
# volume, reusing that section's own locked rendering functions, or (where
# no dataset is needed) simple vector arrows/frames that support those real
# images. None fabricate a microstructure. See prepare_*() in section 6 for
# exactly which real data each schematic_fn closure captures.
# ============================================================================


def schem_multiscale(ax, large_img, small_img, crop_box_px):
    """Real Section 4.1 reference slice: a larger context crop with the
    exact deterministically-chosen ROI window highlighted, zoomed at right
    -- both real voxels of the already-loaded real representative volume
    (see choose_grounded_roi in prepare_multiscale for the ROI criterion).
    The ROI marker uses the neutral ANNOTATION_COLOR, never diffusion-
    orange/GAN-purple."""
    (lx0, ly0, lw, lh), (sx0, sy0, sw, sh) = schem_layout_row(ax, [0.40, 0.28])
    ax_large = ax.inset_axes([lx0, ly0, lw, lh])
    ax_large.imshow(large_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    cx0, cy0, cw, ch = crop_box_px
    ax_large.add_patch(Rectangle((cx0, cy0), cw, ch, fill=False, edgecolor=ANNOTATION_COLOR,
                                  linewidth=1.5))
    image_cell(ax_large, SUBTEXT, lw=1.0)
    ax_small = ax.inset_axes([sx0, sy0, sw, sh])
    ax_small.imshow(small_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    image_cell(ax_small, ANNOTATION_COLOR, lw=1.2)
    helper_label(ax, lx0 + lw / 2.0, "larger context")
    helper_label(ax, sx0 + sw / 2.0, "small FOV")


def schem_2d_to_3d(ax, slice_img, phase_cmap, cube_img):
    """A real Section 4.2 reference 2D slice (categorical phase colors) ->
    a real cutaway cube rendered from that same real reference volume."""
    (sx0, sy0, sw, sh), (cx0, cy0, cw, ch) = schem_layout_row(ax, [0.30, 0.34])
    ax_slice = ax.inset_axes([sx0, sy0, sw, sh])
    ax_slice.imshow(slice_img, cmap=phase_cmap, vmin=0, vmax=2, interpolation="nearest")
    image_cell(ax_slice, SUBTEXT, lw=1.0)
    ax_cube = ax.inset_axes([cx0, cy0, cw, ch])
    ax_cube.imshow(cube_img, interpolation="bilinear")
    image_cell(ax_cube, SUBTEXT, lw=1.0)
    helper_label(ax, sx0 + sw / 2.0, "2D slice")
    helper_label(ax, cx0 + cw / 2.0, "3D volume")


def schem_super_resolution(ax, lr_crop, hr_crop):
    """LR crop -> HR crop, the exact same real physical region of the
    already-loaded Section 4.3 representative pair (see
    prepare_super_resolution: the HR crop is hr[4*y0:4*y1, 4*x0:4*x1] for
    whatever LR crop is lr[y0:y1, x0:x1] -- never independently cropped).
    The LR crop is shown nearest-upsampled to the same display size as the
    native-resolution HR crop, with no artificial blur -- the real
    resolution gap alone makes the LR crop visibly blockier."""
    (lx0, ly0, lw, lh), (hx0, hy0, hw, hh) = schem_layout_row(ax, [0.34, 0.34])
    ax_lr = ax.inset_axes([lx0, ly0, lw, lh])
    ax_lr.imshow(lr_crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    image_cell(ax_lr, SUBTEXT, lw=1.1)
    ax_hr = ax.inset_axes([hx0, hy0, hw, hh])
    ax_hr.imshow(hr_crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    image_cell(ax_hr, SUBTEXT, lw=1.1)
    helper_label(ax, lx0 + lw / 2.0, "LR")
    helper_label(ax, hx0 + hw / 2.0, "HR")


def schem_anisotropy(ax, xy_slice, xz_slice, yz_slice, structure_color):
    """Three real orthogonal mid-slices (XY/XZ/YZ) of the SAME already-
    loaded real Section 4.4 reference volume, side by side as parallel
    views (no arrows -- these are simultaneous views, not a sequence).
    Because the material is anisotropic, its morphology looks visibly
    different depending on viewing direction -- a direct, grounded cue,
    with no quantitative chart and no invented fiber pattern."""
    cmap = ListedColormap(["#FFFFFF", structure_color])
    boxes = schem_layout_row(ax, [0.26, 0.26, 0.26], arrows=False)
    for (bx0, by0, bw, bh), sl, lab in zip(boxes, (xy_slice, xz_slice, yz_slice),
                                            ("XY", "XZ", "YZ")):
        axb = ax.inset_axes([bx0, by0, bw, bh])
        axb.imshow(sl, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        image_cell(axb, SUBTEXT, lw=1.0)
        helper_label(ax, bx0 + bw / 2.0, lab)


PHASE_BOX_W = 0.11
PHASE_PLUS_GAP = 0.025
COMBINED_BOX_W = 0.22


def schem_multiphase(ax, phase_imgs, combined_img):
    """phase 1 + phase 2 + phase 3 -> combined: three real phase-isolated
    crops of the reference volume's own slice (pore/active/CBD, Figure
    4.5's own categorical colors), equal size/spacing with breathing room
    around each '+', combining into the real 3D multiphase reference
    render -- all vertically centered on the same SCHEM_VCENTER."""
    y0 = SCHEM_VCENTER - SCHEM_BOX_H / 2.0
    group_w = 3 * PHASE_BOX_W + 2 * PHASE_PLUS_GAP
    total_w = group_w + 2 * SCHEM_GAP + SCHEM_ARROW_LEN + COMBINED_BOX_W
    x0 = SCHEM_PAD + max(0.0, (1.0 - 2 * SCHEM_PAD - total_w) / 2.0)

    xs = []
    cursor = x0
    for i in range(3):
        xs.append(cursor)
        cursor += PHASE_BOX_W + (PHASE_PLUS_GAP if i < 2 else 0.0)

    for i, x in enumerate(xs):
        axp = ax.inset_axes([x, y0, PHASE_BOX_W, SCHEM_BOX_H])
        axp.imshow(phase_imgs[i], interpolation="nearest")
        image_cell(axp, SUBTEXT, lw=0.8)
        helper_label(ax, x + PHASE_BOX_W / 2.0, f"phase {i + 1}")
        if i < 2:
            plus_x = x + PHASE_BOX_W + PHASE_PLUS_GAP / 2.0
            ax.text(plus_x, SCHEM_VCENTER, "+", ha="center", va="center", fontsize=13,
                    fontweight="bold", color=SUBTEXT, transform=ax.transAxes)

    arrow_x0 = xs[-1] + PHASE_BOX_W + SCHEM_GAP
    arrow_x1 = arrow_x0 + SCHEM_ARROW_LEN
    draw_single_arrow(ax, arrow_x0, SCHEM_VCENTER, arrow_x1, SCHEM_VCENTER, color=SUBTEXT)

    comb_x0 = arrow_x1 + SCHEM_GAP
    ax_c = ax.inset_axes([comb_x0, y0, COMBINED_BOX_W, SCHEM_BOX_H])
    ax_c.imshow(combined_img, interpolation="bilinear")
    image_cell(ax_c, SUBTEXT, lw=1.0)
    helper_label(ax, comb_x0 + COMBINED_BOX_W / 2.0, "combined")


TOPOLOGY_PART_W = 0.19
TOPOLOGY_PLUS_GAP = 0.03
TOPOLOGY_FULL_W = 0.26


def schem_topology(ax, backbone_img, fragments_img, full_img):
    """connected backbone + fragments -> full pore network: the real
    reference sample's own connected-component decomposition (largest
    component alone, disconnected fragments alone, and the two together)
    -- teaching what topology/connectivity means in this benchmark as a
    composition, not a "bad -> good" transformation. All three renders
    come from the exact same real representative sample."""
    y0 = SCHEM_VCENTER - SCHEM_BOX_H / 2.0
    group_w = 2 * TOPOLOGY_PART_W + TOPOLOGY_PLUS_GAP
    total_w = group_w + 2 * SCHEM_GAP + SCHEM_ARROW_LEN + TOPOLOGY_FULL_W
    x0 = SCHEM_PAD + max(0.0, (1.0 - 2 * SCHEM_PAD - total_w) / 2.0)

    x_backbone = x0
    x_fragments = x_backbone + TOPOLOGY_PART_W + TOPOLOGY_PLUS_GAP
    for x, img, lab in ((x_backbone, backbone_img, "connected backbone"),
                        (x_fragments, fragments_img, "fragments")):
        axb = ax.inset_axes([x, y0, TOPOLOGY_PART_W, SCHEM_BOX_H])
        axb.imshow(img, interpolation="bilinear")
        image_cell(axb, SUBTEXT, lw=0.9)
        helper_label(ax, x + TOPOLOGY_PART_W / 2.0, lab)
    plus_x = x_backbone + TOPOLOGY_PART_W + TOPOLOGY_PLUS_GAP / 2.0
    ax.text(plus_x, SCHEM_VCENTER, "+", ha="center", va="center", fontsize=13,
            fontweight="bold", color=SUBTEXT, transform=ax.transAxes)

    arrow_x0 = x_fragments + TOPOLOGY_PART_W + SCHEM_GAP
    arrow_x1 = arrow_x0 + SCHEM_ARROW_LEN
    draw_single_arrow(ax, arrow_x0, SCHEM_VCENTER, arrow_x1, SCHEM_VCENTER, color=SUBTEXT)

    full_x0 = arrow_x1 + SCHEM_GAP
    ax_full = ax.inset_axes([full_x0, y0, TOPOLOGY_FULL_W, SCHEM_BOX_H])
    ax_full.imshow(full_img, interpolation="bilinear")
    image_cell(ax_full, SUBTEXT, lw=1.0)
    helper_label(ax, full_x0 + TOPOLOGY_FULL_W / 2.0, "full network")


def schem_large_volume(ax, small_img, large_img):
    """A small real crop of the reference 512^3 volume -> the full real
    reference volume rendered the same way, with a neutral-colored
    wireframe bounding box on the large render marking exactly where the
    small subvolume originates -- so the small-in-large correspondence is
    explicit rather than two disconnected renders. Never orange/purple."""
    (sx0, sy0, sw, sh), (lx0, ly0, lw, lh) = schem_layout_row(ax, [0.26, 0.34])
    ax_small = ax.inset_axes([sx0, sy0, sw, sh])
    ax_small.imshow(small_img, interpolation="bilinear")
    image_cell(ax_small, SUBTEXT, lw=1.0)
    ax_large = ax.inset_axes([lx0, ly0, lw, lh])
    ax_large.imshow(large_img, interpolation="bilinear")
    bx0, by0, bw, bh = 0.04, 0.04, 0.30, 0.30
    ax_large.plot([bx0, bx0, bx0 + bw, bx0 + bw, bx0], [by0, by0 + bh, by0 + bh, by0, by0],
                  color=ANNOTATION_COLOR, linewidth=1.6, transform=ax_large.transAxes,
                  solid_capstyle="round")
    image_cell(ax_large, SUBTEXT, lw=1.0)
    helper_label(ax, sx0 + sw / 2.0, "small subvolume")
    helper_label(ax, lx0 + lw / 2.0, "large domain")


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

    # Grounded multiscale schematic: a real mid-slice of the already-loaded
    # real reference volume, small-window-vs-larger-context, both real
    # voxels of the same real sample (never a separate finalize_renders
    # call -- this never touches the diffusion/gan crop union above). The
    # small ROI is chosen deterministically (choose_grounded_roi) so it
    # actually contains both phases, rather than a plain center crop that
    # can land almost entirely in one phase.
    real_vol = rep["real"]["vol"]
    large_img = binary_slice_u8(real_vol)
    large_bin01 = (large_img > 0).astype(np.uint8)
    crop_size = max(4, int(min(large_bin01.shape) * 0.14))
    r0, c0, ch, cw, roi_frac = choose_grounded_roi(large_bin01, crop_size)
    small_img = large_img[r0:r0 + ch, c0:c0 + cw]
    crop_box_px = (c0, r0, cw, ch)
    log(f"[multiscale] deterministic ROI phase-1 occupancy = {roi_frac:.4f} "
        f"(whole-slice occupancy = {float(np.mean(large_bin01)):.4f})")

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
        "schematic_fn": lambda ax: schem_multiscale(ax, large_img, small_img, crop_box_px),
        "schematic_source": f"real Section 4.1 reference representative mid-slice "
                             f"({rep['real']['file']}), deterministic both-phases ROI "
                             f"(occupancy {roi_frac:.4f}, window 0.20-0.80 bound)",
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

    # Grounded 2D->3D schematic: a real reference 2D slice -> a real cutaway
    # cube rendered from that same real reference volume. Rendered and
    # finalized in its own independent call (own alpha-crop bbox) so the
    # diffusion/gan cube images above are byte-for-byte unaffected.
    real_vol = rep["real"]["vol"]
    slice_img = real_vol[real_vol.shape[0] // 2]
    real_raw = TMP / "2d3d_raw_real.png"
    real_png = TMP / "2d3d_render_real.png"
    m2.render_group_cube(real_vol, "real", real_raw, parallel_scale)
    m2.finalize_renders({"real": real_raw}, {"real": real_png})
    cube_img = Image.open(real_png)

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
        "schematic_fn": lambda ax: schem_2d_to_3d(ax, slice_img, m2.PHASE_CMAP, cube_img),
        "schematic_source": f"real Section 4.2 reference representative mid-slice and cutaway "
                             f"cube ({rep['real']['file']}), rendered by the section's own "
                             f"render_group_cube",
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

    # Same real physical ROI in both LR and HR -- never cropped
    # independently. SR_FACTOR is read directly from the section's own
    # locked EXPECTED_HR_SIZE/EXPECTED_LR_SIZE (800/200 = 4x), so an LR
    # window lr[y0:y1, x0:x1] always corresponds to exactly
    # hr[4*y0:4*y1, 4*x0:4*x1]. The LR window location is chosen
    # deterministically (highest local pixel variance among a fixed grid
    # of candidates -- choose_high_variance_roi) rather than a fixed
    # corner, so it lands on a structure-rich, informative region. Display:
    # the native LR crop is nearest-upsampled to the same canvas size as
    # the native HR crop, with NO artificial blur -- the real resolution
    # gap alone makes LR visibly blockier.
    sr_factor = m3.EXPECTED_HR_SIZE[0] // m3.EXPECTED_LR_SIZE[0]
    lr_size = 50
    lr_r0, lr_c0 = choose_high_variance_roi(lr_full, lr_size)
    lr_crop_native = lr_full[lr_r0:lr_r0 + lr_size, lr_c0:lr_c0 + lr_size]
    display_size = lr_size * sr_factor
    lr_crop = np.array(Image.fromarray(lr_crop_native).resize((display_size, display_size),
                                                                Image.NEAREST))
    hr_r0, hr_c0 = lr_r0 * sr_factor, lr_c0 * sr_factor
    hr_crop = hr_full[hr_r0:hr_r0 + display_size, hr_c0:hr_c0 + display_size]

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
                             f"lr={lr_files[rep_idx]}), the exact same physical ROI in both "
                             f"(lr[{lr_r0}:{lr_r0 + lr_size},{lr_c0}:{lr_c0 + lr_size}] <-> "
                             f"hr[{hr_r0}:{hr_r0 + display_size},{hr_c0}:{hr_c0 + display_size}]), "
                             f"chosen by highest local variance -- no artificial blur",
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

    # Grounded anisotropy schematic: three real orthogonal mid-slices
    # (XY/XZ/YZ) of the SAME real reference representative volume. Because
    # the glass-fiber structure is genuinely anisotropic, its morphology
    # looks visibly different depending on viewing direction -- shown
    # directly via real cross-sections, with no derived metric/chart and
    # no invented pattern.
    real_volumes = m4.load_group_volumes("real")
    real_rows = m4.compute_group_descriptors(real_volumes)
    real_file, _ = m4.choose_representative("real", real_rows)
    real_vol = real_volumes[real_file]
    xy_slice, xz_slice, yz_slice = orthogonal_binary_slices(real_vol)

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
        "schematic_fn": lambda ax: schem_anisotropy(ax, xy_slice, xz_slice, yz_slice,
                                                      m4.COLORS["real"]),
        "schematic_source": f"real Section 4.4 reference representative volume ({real_file}), "
                             f"three real orthogonal mid-slices (XY/XZ/YZ)",
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

    # Grounded multiphase schematic: three real phase-isolated crops of the
    # real reference sample's own mid-slice (pore/active/CBD, Figure 4.5's
    # own locked categorical colors), combining into that same real
    # sample's real 3D multiphase cutaway render (own independent
    # finalize_renders call -- never mixed into the diffusion/gan crop
    # union above).
    real_file = real_row["file"]
    real_vol = volumes_by_group["real"][real_file]
    real_slice = real_vol[real_vol.shape[0] // 2]
    phase_imgs = {k: phase_mask_rgba(real_slice, k, m5.PHASE_COLORS[k]) for k in (0, 1, 2)}

    combined_raw = TMP / "multiphase_raw_real_combined.png"
    combined_png = TMP / "multiphase_render_real_combined.png"
    m5.render_group_multiphase(real_vol, "real", combined_raw, parallel_scale)
    m5.finalize_renders({"real_combined": combined_raw}, {"real_combined": combined_png})
    combined_img = Image.open(combined_png)

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
        "schematic_fn": lambda ax: schem_multiphase(ax, phase_imgs, combined_img),
        "schematic_source": f"real Section 4.5 reference representative mid-slice, phase-isolated "
                             f"({real_file}), combined with the section's own real 3D "
                             f"render_group_multiphase render",
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

    # Grounded topology schematic: the real reference sample's own
    # connected-component labeling (already computed by the locked
    # component_stats), decomposed into three real renders of the SAME
    # sample -- the largest component alone ("connected backbone"), the
    # disconnected fragments alone (largest component removed), and the
    # full labeling (both together) -- teaching what topology/connectivity
    # means here as a composition (backbone + fragments -> full network),
    # not a "bad -> good" transformation. All three via the section's own
    # render_group_topology, in an independent finalize_renders call so
    # the diffusion/gan renders above are unaffected.
    real_file = representative_files["real"]
    real_vol = volumes_by_group["real"][real_file]
    real_stats = m6.component_stats(real_vol)
    labeled, largest_id = real_stats["labeled"], real_stats["largest_id"]
    fragments_labeled = labeled.copy()
    backbone_labeled = np.zeros_like(labeled)
    if largest_id is not None:
        fragments_labeled[fragments_labeled == largest_id] = 0
        backbone_labeled = np.where(labeled == largest_id, labeled, 0)

    backbone_raw = TMP / "topology_raw_real_backbone.png"
    frag_raw = TMP / "topology_raw_real_fragments.png"
    full_raw = TMP / "topology_raw_real_full.png"
    backbone_png = TMP / "topology_render_real_backbone.png"
    frag_png = TMP / "topology_render_real_fragments.png"
    full_png = TMP / "topology_render_real_full.png"
    m6.render_group_topology(backbone_labeled, largest_id if largest_id is not None else -1,
                              "real", backbone_raw, parallel_scale)
    m6.render_group_topology(fragments_labeled, -1, "real", frag_raw, parallel_scale)
    m6.render_group_topology(labeled, largest_id, "real", full_raw, parallel_scale)
    m6.finalize_renders({"backbone": backbone_raw, "fragments": frag_raw, "full": full_raw},
                         {"backbone": backbone_png, "fragments": frag_png, "full": full_png})
    backbone_img = Image.open(backbone_png)
    fragments_img = Image.open(frag_png)
    full_img = Image.open(full_png)

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
        "schematic_fn": lambda ax: schem_topology(ax, backbone_img, fragments_img, full_img),
        "schematic_source": f"real Section 4.6 reference representative sample ({real_file}), "
                             f"its own connected-component labeling decomposed into backbone "
                             f"(largest component alone), fragments (largest removed), and the "
                             f"full network (both together)",
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

    # Grounded large-volume schematic: a small real crop of the real 512^3
    # reference volume -> the full real reference volume, both rendered by
    # the section's own render_group_isosurface (own independent
    # finalize_renders call -- never mixed into the diffusion/gan crop
    # union above).
    m7.validate_visualization_folder("real", m7.SAMPLE_DIRS_512["real"])
    real_vol, _real_phi = m7.load_and_validate_representative("real", m7.REP_FILES["real"])
    n = real_vol.shape[0]
    small_n = n // 4
    small_vol = real_vol[:small_n, :small_n, :small_n]

    small_raw = TMP / "largevol_raw_real_small.png"
    large_raw = TMP / "largevol_raw_real_large.png"
    small_png = TMP / "largevol_render_real_small.png"
    large_png = TMP / "largevol_render_real_large.png"
    m7.render_group_isosurface(small_vol, "real", small_raw, 0.72 * small_n)
    m7.render_group_isosurface(real_vol, "real", large_raw, 0.72 * n)
    m7.finalize_renders({"small": small_raw, "large": large_raw},
                         {"small": small_png, "large": large_png})
    small_img = Image.open(small_png)
    large_img = Image.open(large_png)

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
        "schematic_fn": lambda ax: schem_large_volume(ax, small_img, large_img),
        "schematic_source": f"real Section 4.7 reference representative volume "
                             f"({m7.REP_FILES['real']}), a real {small_n}^3 corner crop vs. "
                             f"the full real {n}^3 volume, both via render_group_isosurface",
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


SLOT_FRAC = {"title": 0.15, "schematic": 0.26, "renders": 0.59}

# Two explicit shared gap constants, not per-panel tuning: title and
# schematic are always adjacent in `order` and sit close together (the
# title reads as naming the schematic right next to it), while schematic
# and renders always get the SAME larger separation in both rows -- the
# gap magnitude never depends on row direction, only which pair of slots
# is adjacent.
TITLE_SCHEMATIC_GAP = 0.014
SCHEMATIC_MODEL_GAP = 0.030

# The "model unit" rhythm (MODEL NAME / small fixed gap / MODEL RENDER),
# expressed as fractions of the renders slot's own height so it is
# pixel-identical on all 14 model cells (7 panels x diffusion/GAN)
# regardless of row direction or neighboring slot.
MODEL_NAME_TOP_INSET = 0.11    # breathing room above the model-name text
MODEL_NAME_LINE_H = 0.15       # band reserved for the model-name text itself
MODEL_NAME_RENDER_GAP = 0.025  # small fixed gap between the name and its render


def _slot_gap(key_a, key_b):
    return TITLE_SCHEMATIC_GAP if {key_a, key_b} == {"title", "schematic"} else SCHEMATIC_MODEL_GAP


def _stack_slots(y0, h, order):
    """Top-to-bottom slot geometry inside a card spanning [y0, y0+h], per
    `order` (a top-to-bottom list of 'renders'/'schematic'/'title').
    Returns {slot_key: (slot_y0, slot_h)}, slot_y0 being each slot's
    bottom-left y (matplotlib's own convention for add_axes/Rectangle)."""
    pad = 0.030 * h
    total_gap = _slot_gap(order[0], order[1]) + _slot_gap(order[1], order[2])
    avail = h - 2 * pad - total_gap * h
    positions = {}
    cursor_top = y0 + h - pad
    for i, key in enumerate(order):
        slot_h = SLOT_FRAC[key] * avail
        slot_bottom = cursor_top - slot_h
        positions[key] = (slot_bottom, slot_h)
        if i < len(order) - 1:
            cursor_top = slot_bottom - _slot_gap(key, order[i + 1]) * h
    return positions


def draw_model_unit(fig, cell_x0, cell_w, ry0, rh, info):
    """One reusable 'model unit' -- MODEL NAME, [small fixed gap], MODEL
    RENDER -- used identically for all 14 model cells (7 panels x
    diffusion/GAN), so the name always visually belongs to its render
    rather than floating above it, on the exact same rhythm everywhere."""
    img_h = rh * (1.0 - MODEL_NAME_TOP_INSET - MODEL_NAME_LINE_H - MODEL_NAME_RENDER_GAP)
    img_y0 = ry0
    name_top_y = ry0 + rh - MODEL_NAME_TOP_INSET * rh
    fig.text(cell_x0 + cell_w / 2.0, name_top_y, info["label"],
              ha="center", va="top", fontsize=8.4, fontweight="bold", color=info["color"])
    ax = fig.add_axes([cell_x0, img_y0, cell_w, img_h])
    if info["cmap"] is not None:
        ax.imshow(info["image"], cmap=info["cmap"], vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(info["image"], interpolation="bilinear")
    image_cell(ax, info["color"], lw=CELL_LW)


def build_task_card(fig, xywh, title, task, order):
    """Render one task card with the top-to-bottom internal slot order
    given by `order` -- ORDER_TOP (renders, schematic, title) for the top
    row or ORDER_BOTTOM (title, schematic, renders) for the bottom row, so
    the task title always lands nearest the hub (Requirement 2)."""
    x0, y0, w, h = xywh
    add_card(fig, xywh)
    slots = _stack_slots(y0, h, order)

    # ---- title slot: bold, larger, framed in a light contrast chip -------
    ty0, th = slots["title"]
    chip_w, chip_h = 0.88 * w, 0.80 * th
    chip_x0 = x0 + (w - chip_w) / 2.0
    chip_y0 = ty0 + (th - chip_h) / 2.0
    fig.add_artist(Rectangle(
        (chip_x0, chip_y0), chip_w, chip_h, transform=fig.transFigure,
        facecolor=TITLE_CHIP_FACE, edgecolor=CARD_EDGE, linewidth=0.9, zorder=-40,
    ))
    fig.text(x0 + w / 2.0, ty0 + th / 2.0, title, ha="center", va="center",
              fontsize=11.5, fontweight="bold", color=TEXT)

    # ---- schematic slot: shrunk relative to renders, secondary in weight --
    sy0, sh = slots["schematic"]
    schem_x0 = x0 + PANEL_INNER_PAD * w
    schem_w = w - 2 * PANEL_INNER_PAD * w
    ax_schem = schematic_axis(fig, [schem_x0, sy0, schem_w, sh])
    task["schematic_fn"](ax_schem)

    # ---- renders slot: two identical "model units" side by side ----------
    ry0, rh = slots["renders"]
    pad_x = PANEL_INNER_PAD * w
    cell_gap = 0.05 * w
    cell_w = (w - 2 * pad_x - cell_gap) / 2.0
    left_x0 = x0 + pad_x
    right_x0 = left_x0 + cell_w + cell_gap

    for cell_x0, side in ((left_x0, "diffusion"), (right_x0, "gan")):
        draw_model_unit(fig, cell_x0, cell_w, ry0, rh, task[side])


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

    # One consistent elbow-routed connector tree per row (trunk -> bus ->
    # per-panel drop), landing on each panel's title-side edge -- see
    # draw_hub_tree for why this replaces the previous per-task arcs.
    top_bus_y = (hub_cy + HUB_H / 2.0 + TOP_Y0) / 2.0
    top_cxs = [TASK_CARDS[k][0] + TASK_CARDS[k][2] / 2.0 for k in TOP_ROW]
    top_edge_ys = [TASK_CARDS[k][1] for k in TOP_ROW]
    draw_hub_tree(fig, hub_cx, hub_cy + HUB_H / 2.0, top_bus_y, top_cxs, top_edge_ys)

    bottom_bus_y = (hub_cy - HUB_H / 2.0 + (BOTTOM_Y0 + ROW_H)) / 2.0
    bottom_cxs = [TASK_CARDS[k][0] + TASK_CARDS[k][2] / 2.0 for k in BOTTOM_ROW]
    bottom_edge_ys = [TASK_CARDS[k][1] + TASK_CARDS[k][3] for k in BOTTOM_ROW]
    draw_hub_tree(fig, hub_cx, hub_cy - HUB_H / 2.0, bottom_bus_y, bottom_cxs, bottom_edge_ys)

    for key in TOP_ROW:
        build_task_card(fig, TASK_CARDS[key], TASK_TITLES[key], tasks[key], ORDER_TOP)
    for key in BOTTOM_ROW:
        build_task_card(fig, TASK_CARDS[key], TASK_TITLES[key], tasks[key], ORDER_BOTTOM)

    add_card(fig, HUB_CARD, lw=HUB_LW, facecolor=HUB_FACE)
    fig.text(hub_cx, hub_cy, "Task-oriented\nbenchmark", ha="center", va="center",
              fontsize=15.5, fontweight="bold", color=TEXT, linespacing=1.35)

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
