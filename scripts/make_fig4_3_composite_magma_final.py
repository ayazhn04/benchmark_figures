#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4.3 composite (magma, final)
Limited-resolution reconstruction / paired 2D x4 super-resolution.
Bicubic x4 vs ResShift vs SurVol, HR reference vs LR input, on DRSRD1 sandstone.

Same visual contract as Figures 4.1/4.2 (make_fig4_1_composite_magma_final.py,
make_fig4_2_composite_magma_final.py): same cards, fonts, panel-label style,
export block. No tight_layout / constrained_layout / bbox_inches.

Unlike 4.1/4.2, this script never reimplements the official evaluator's
phase-threshold, bicubic, radial-TPCF, or radial-PSD logic from a guess --
it dynamically imports section43_full_metrics_2d.py itself and calls its
actual functions, so panels b/c/e are built from the exact official
definitions rather than an approximation of them. If those functions can't
be identified unambiguously, the affected panel is refused rather than
faked (see EVALUATOR_OVERRIDES below for a manual escape hatch).
"""

from __future__ import annotations

import importlib.util
import inspect
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

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIG
# ============================================================================

PROJECT = Path(__file__).resolve().parents[1]  # -> /home/ra2/4.3

RESSHIFT_ROOT = PROJECT / "resshift_superres_4_3"
HR_DIR = RESSHIFT_ROOT / "data" / "processed" / "2d_x4_sandstone" / "test50" / "hr"
LR_DIR = RESSHIFT_ROOT / "data" / "processed" / "2d_x4_sandstone" / "test50" / "lr"
RESSHIFT_DIR = RESSHIFT_ROOT / "outputs" / "samples" / "final_50_2d_resshift_release"

SURVOL_ROOT = PROJECT / "survol_gan_4_3"
SURVOL_DIR_EXPECTED = SURVOL_ROOT / "outputs" / "samples" / "survol_xy_selected_epoch7_test50" / "predictions"

METRICS_DIR = PROJECT / "section43_superres_final_metrics"
EVALUATOR_PATH = METRICS_DIR / "section43_full_metrics_2d.py"
MAIN_TABLE_CSV = METRICS_DIR / "main_table_candidates.csv"
PER_IMAGE_CSV = METRICS_DIR / "per_image_metrics.csv"

OUT = PROJECT / "paper_figures_4_3" / "figure_4_3_composite_magma_final"
LOGS = PROJECT / "logs"
STEM = "fig4_3_composite_magma_final"

# Forbidden sources -- never touched regardless of what discovery finds.
FORBIDDEN_TOKENS = ("sr3", "withsr3", "ema2000", "ema5000", "ema50000",
                     "3dx4sandstone", "final50_3d", "final503d")

EXPECTED_HR_SIZE = (800, 800)
EXPECTED_LR_SIZE = (200, 200)
EXPECTED_N = 50

PHASE_THRESHOLD = 128   # per task spec; the >= vs > direction is resolved by
MAX_LAG = 256           # calling the evaluator's own function, never guessed.

ROI_SIZE = 192          # HR-grid pixels (= 48 LR pixels at x4)
ROI_STRIDE = 16

# Manual override escape hatch: if evaluator-function auto-detection finds
# more than one candidate (or the wrong one), fill these in with the exact
# names printed under "[evaluator] found function: ..." and re-run. Left as
# None, auto-detection is used and must be unambiguous.
EVALUATOR_OVERRIDES = {
    "threshold": None,   # e.g. "binarize_phase"
    "bicubic": None,     # e.g. "bicubic_upsample"
    "tpcf": None,        # e.g. "tpcf_radial_phase1"
    "psd": None,         # e.g. "psd_radial_phase1"
}

FIG_W, FIG_H = 17.8, 10.2

# ============================================================================
# 1b. OFFICIAL SANITY-CHECK VALUES (Table 4.8 / main_table_candidates.csv)
# ============================================================================

SANITY_TABLE = {
    "bicubic": {"psnr": 25.9781, "ssim": 0.646337, "phase_fraction_error": 0.002221,
                "iou": 0.970074, "dice": 0.984810, "local_porosity_w32": 0.009520,
                "interface_density_error": 0.006880, "tpcf_mae": 0.003190,
                "psd_mae": 2.2361e-06, "lr_consistency_psnr": 36.0320},
    "resshift": {"psnr": 27.0727, "ssim": 0.643310, "phase_fraction_error": 0.004876,
                 "iou": 0.977443, "dice": 0.988593, "local_porosity_w32": 0.005684,
                 "interface_density_error": 0.000889, "tpcf_mae": 0.007296,
                 "psd_mae": 6.1070e-07, "lr_consistency_psnr": 38.4415},
    "survol": {"psnr": 27.3478, "ssim": 0.637886, "phase_fraction_error": 0.000473,
               "iou": 0.978670, "dice": 0.989220, "local_porosity_w32": 0.003833,
               "interface_density_error": 0.002404, "tpcf_mae": 0.000745,
               "psd_mae": 5.3788e-07, "lr_consistency_psnr": 38.8326},
}
SANITY_REL_TOL = 0.15  # 15% relative tolerance -- these are cross-checks, not exact re-derivations

# Official CSV method-key -> internal key.
CSV_METHOD_MAP = {
    "real_hr": "hr",
    "nearest_x4": "nearest",
    "bicubic_x4": "bicubic",
    "diffusion_resshift_release_0ft": "resshift",
    "gan_survol_xy_epoch7": "survol",
}

# ============================================================================
# 2. STYLE -- identical contract to Figures 4.1/4.2
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

REFERENCE = "#8C93A1"
RESSHIFT = "#F2A93B"
SURVOL = "#772A8E"
LR_COLOR = "#C0C4CC"
BICUBIC_COLOR = "#738196"

COLORS = {"lr": LR_COLOR, "hr": REFERENCE, "bicubic": BICUBIC_COLOR,
          "resshift": RESSHIFT, "survol": SURVOL}
LABELS = {"lr": "LR input", "hr": "HR reference", "bicubic": "Bicubic ×4",
          "resshift": "ResShift", "survol": "SurVol"}
LINESTYLES = {"hr": "-", "bicubic": (0, (2, 1.6)), "resshift": "-", "survol": (0, (5.5, 2.2))}
LINEWIDTHS = {"hr": 1.85, "bicubic": 1.6, "resshift": 1.85, "survol": 1.95}
TITLE_COLOR = {"lr": SUBTEXT, "hr": TEXT, "bicubic": BICUBIC_COLOR,
               "resshift": RESSHIFT, "survol": SURVOL}

CARD_LW = 0.8
CELL_LW = 1.6

PANEL_A_COLS = ["lr", "hr", "bicubic", "resshift", "survol"]
PANEL_B_COLS = ["lr", "hr", "bicubic", "resshift", "survol"]
PANEL_C_COLS = ["bicubic", "resshift", "survol"]
LEARNED_METHODS = ["resshift", "survol"]
BAR_METHODS = ["bicubic", "resshift", "survol"]

# ============================================================================
# 3. GEOMETRY (figure fractions) -- adapted from the 4.1/4.2 contract for a
# five-card layout: full comparison / zoom+error / metrics+curves.
# ============================================================================

card_a = [0.045, 0.675, 0.910, 0.245]
card_b = [0.045, 0.395, 0.565, 0.215]
card_c = [0.630, 0.395, 0.325, 0.215]
card_d = [0.045, 0.055, 0.440, 0.270]
card_e = [0.505, 0.055, 0.450, 0.270]

PANEL_LABEL_OFFSET = 0.030

assert abs((card_b[0] + card_b[2]) + 0.020 - card_c[0]) < 1e-9, "b/c gap mismatch"
assert abs(card_b[1] - card_c[1]) < 1e-12, "b/c y mismatch"
assert abs(card_b[3] - card_c[3]) < 1e-12, "b/c height mismatch"
assert abs((card_c[0] + card_c[2]) - (card_a[0] + card_a[2])) < 1e-9, "a/c right edge mismatch"
assert abs((card_d[0] + card_d[2]) + 0.020 - card_e[0]) < 1e-9, "d/e gap mismatch"
assert abs(card_d[1] - card_e[1]) < 1e-12, "d/e y mismatch"
assert abs(card_d[3] - card_e[3]) < 1e-12, "d/e height mismatch"
assert abs((card_e[0] + card_e[2]) - (card_a[0] + card_a[2])) < 1e-9, "a/e right edge mismatch"
assert abs(card_a[0] - card_b[0]) < 1e-12, "a/b left edge mismatch"
assert abs(card_a[0] - card_d[0]) < 1e-12, "a/d left edge mismatch"

# ============================================================================
# 4. GENERIC HELPERS
# ============================================================================


def log(*a):
    print(*a, flush=True)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _canon_id(x) -> str:
    """Canonical form of an image id for cross-source matching. IDs like
    '0901' are the ground truth (from filenames); a CSV column of the same
    IDs is frequently read back by pandas as int64, silently dropping the
    leading zero ('901'). Comparing raw strings in that case makes every
    lookup fail without raising -- every score collapses to the same
    fallback value and downstream code silently picks the first row. This
    strips leading zeros from any purely-numeric id on both sides so '0901'
    and 901 compare equal, while leaving non-numeric ids untouched."""
    s = str(x).strip()
    return str(int(s)) if re.fullmatch(r"-?\d+", s) else s


def _path_is_forbidden(p) -> bool:
    s = _norm(str(p))
    return any(tok in s for tok in FORBIDDEN_TOKENS)


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
    log(msg + "  -> skipped. Available:", list(df.columns))
    return None


def pick_row(df: pd.DataFrame, method_col: str, method_key: str):
    """Row whose method column, after CSV_METHOD_MAP normalisation, equals
    method_key. Also tolerates the raw CSV string itself (e.g. 'bicubic')."""
    raw_keys = [k for k, v in CSV_METHOD_MAP.items() if v == method_key]
    mask = df[method_col].astype(str).str.strip().isin(raw_keys)
    sub = df[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


# ============================================================================
# 5. EVALUATOR INTROSPECTION -- dynamically import the real
# section43_full_metrics_2d.py and reuse its actual functions instead of
# reimplementing a guess of its threshold/bicubic/TPCF/PSD conventions.
# ============================================================================


def load_evaluator_module():
    if not EVALUATOR_PATH.exists():
        raise FileNotFoundError(
            f"Official evaluator not found: {EVALUATOR_PATH}\n"
            f"This script requires the real evaluator to reuse its exact "
            f"phase-threshold, bicubic, TPCF and PSD conventions -- it does "
            f"not reimplement a guess of them."
        )
    spec = importlib.util.spec_from_file_location("section43_full_metrics_2d", str(EVALUATOR_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module_functions(mod):
    return {name: fn for name, fn in inspect.getmembers(mod, inspect.isfunction)
            if fn.__module__ == mod.__name__}


def _find_candidates(funcs, keywords, exclude=()):
    out = []
    for name, fn in funcs.items():
        ln = _norm(name)
        if any(k in ln for k in keywords) and not any(e in ln for e in exclude):
            out.append((name, fn))
    return out


def _resolve_function(funcs, capability, keywords, exclude=()):
    override = EVALUATOR_OVERRIDES.get(capability)
    if override:
        if override not in funcs:
            raise RuntimeError(f"[evaluator] EVALUATOR_OVERRIDES['{capability}'] = '{override}' "
                                f"is not a function in {EVALUATOR_PATH.name}")
        log(f"[evaluator] {capability}: using manual override '{override}'")
        return funcs[override]

    cands = _find_candidates(funcs, keywords, exclude)
    if len(cands) == 1:
        name, fn = cands[0]
        log(f"[evaluator] {capability}: auto-resolved to '{name}'")
        return fn
    if len(cands) == 0:
        return None
    names = [n for n, _ in cands]
    raise RuntimeError(
        f"[evaluator] {capability}: {len(cands)} ambiguous candidates found: {names}\n"
        f"Set EVALUATOR_OVERRIDES['{capability}'] to the correct one and re-run."
    )


def _try_calls(fn, variants, sanity_check, what):
    """Try several (args, kwargs) call signatures against fn until one both
    succeeds and passes sanity_check(result); log which one worked."""
    errors = []
    for args, kwargs in variants:
        try:
            result = fn(*args, **kwargs)
            if sanity_check(result):
                return result
            errors.append(f"args={args} kwargs={kwargs} -> failed sanity check")
        except Exception as exc:
            errors.append(f"args={args} kwargs={kwargs} -> {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"[evaluator] could not call '{fn.__name__}' for {what} with any known signature:\n  "
        + "\n  ".join(errors)
    )


class Evaluator:
    """Thin adaptive wrapper around the real evaluator module's functions."""

    def __init__(self, mod):
        self.mod = mod
        funcs = _module_functions(mod)
        log(f"[evaluator] {len(funcs)} module-level functions found in {EVALUATOR_PATH.name}:")
        for name, fn in funcs.items():
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(?)"
            log(f"[evaluator]   {name}{sig}")

        self.threshold_fn = _resolve_function(
            funcs, "threshold",
            keywords=["threshold", "binariz", "binaris", "tophase", "phasemap", "getphase", "phase1"])
        self.bicubic_fn = _resolve_function(funcs, "bicubic", keywords=["bicubic"])
        self.tpcf_fn = _resolve_function(
            funcs, "tpcf", keywords=["tpcf", "twopoint", "s2radial", "radialtpcf", "tpc"])
        self.psd_fn = _resolve_function(
            funcs, "psd", keywords=["psd", "powerspectrum", "radialpsd", "spectraldensity"])

        if self.threshold_fn is None:
            raise RuntimeError(
                "[evaluator] no phase-threshold/binarization function could be found. "
                "This script refuses to guess the >=128 vs >128 convention -- set "
                "EVALUATOR_OVERRIDES['threshold'] to the correct function name."
            )
        if self.bicubic_fn is None:
            raise RuntimeError(
                "[evaluator] no bicubic-upsampling function could be found. Set "
                "EVALUATOR_OVERRIDES['bicubic'] to the correct function name."
            )
        if self.tpcf_fn is None:
            log("[evaluator] WARNING: no radial-TPCF function found -> panel e TPCF will show "
                "'metric unavailable' rather than a fabricated curve.")
        if self.psd_fn is None:
            log("[evaluator] WARNING: no radial-PSD function found -> panel e PSD will show "
                "'metric unavailable' rather than a fabricated curve.")

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        def ok(r):
            a = np.asarray(r)
            return a.shape == gray.shape and a.size > 0
        variants = [
            ((gray,), {}),
            ((gray,), {"threshold": PHASE_THRESHOLD}),
            ((gray, PHASE_THRESHOLD), {}),
        ]
        r = _try_calls(self.threshold_fn, variants, ok, "phase binarization")
        return np.asarray(r).astype(bool)

    def bicubic(self, lr_gray: np.ndarray, out_size=EXPECTED_HR_SIZE) -> np.ndarray:
        def ok(r):
            a = np.asarray(r)
            return a.ndim == 2 and a.shape[0] >= out_size[0] * 0.9

        variants = [
            ((lr_gray,), {"scale": 4}),
            ((lr_gray, 4), {}),
            ((lr_gray,), {"scale_factor": 4}),
            ((lr_gray, out_size), {}),
            ((lr_gray,), {"size": out_size}),
            ((lr_gray,), {}),
        ]
        r = _try_calls(self.bicubic_fn, variants, ok, "bicubic upsampling")
        arr = np.asarray(r)
        if arr.shape != out_size:
            img = Image.fromarray(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr)
            img = img.resize((out_size[1], out_size[0]), Image.BICUBIC)
            arr = np.array(img)
        return arr

    def tpcf(self, binary: np.ndarray):
        if self.tpcf_fn is None:
            return None
        def ok(r):
            return r is not None
        variants = [
            ((binary,), {"max_lag": MAX_LAG}),
            ((binary, MAX_LAG), {}),
            ((binary,), {}),
        ]
        r = _try_calls(self.tpcf_fn, variants, ok, "radial TPCF")
        return _unpack_curve(r, MAX_LAG)

    def psd(self, binary: np.ndarray):
        if self.psd_fn is None:
            return None
        def ok(r):
            return r is not None
        variants = [
            ((binary,), {"max_lag": MAX_LAG}),
            ((binary, MAX_LAG), {}),
            ((binary,), {}),
        ]
        r = _try_calls(self.psd_fn, variants, ok, "radial PSD")
        return _unpack_curve(r, MAX_LAG)


def _unpack_curve(result, max_len):
    """Normalise an evaluator curve-function's return value to (x, y)."""
    if isinstance(result, tuple) and len(result) == 2:
        x, y = np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
        return x, y
    y = np.asarray(result, dtype=float).ravel()
    x = np.arange(y.size, dtype=float)
    return x, y


# ============================================================================
# 6. IMAGE / PATH RESOLUTION
# ============================================================================


def resolve_survol_dir() -> Path:
    if SURVOL_DIR_EXPECTED.exists() and not _path_is_forbidden(SURVOL_DIR_EXPECTED):
        n = len(list(SURVOL_DIR_EXPECTED.glob("*.png")))
        if n == EXPECTED_N:
            log(f"[data:survol] using expected folder: {SURVOL_DIR_EXPECTED} ({n} PNGs)")
            return SURVOL_DIR_EXPECTED
        log(f"[data:survol] expected folder exists but has {n} PNGs (need {EXPECTED_N}) -> searching")

    search_root = SURVOL_ROOT / "outputs"
    if not search_root.exists():
        raise FileNotFoundError(f"[data:survol] expected SurVol outputs root not found: {search_root}")

    # Match on the directory's own name, not the full path string -- matching
    # the full path would also catch e.g. a "predictions" subfolder nested
    # under an already-matching parent, double-counting the same real
    # candidate and manufacturing a false ambiguity.
    dir_candidates = [p for p in search_root.rglob("*")
                      if p.is_dir() and "survolxyselectedepoch7test50" in _norm(p.name)
                      and not _path_is_forbidden(p)]
    log(f"[data:survol] expected folder not usable -> candidates under {search_root}: "
        f"{[str(p) for p in dir_candidates]}")

    resolved = []
    seen = set()
    for c in dir_candidates:
        pred = c / "predictions" if (c / "predictions").is_dir() else c
        if pred in seen:
            continue
        seen.add(pred)
        n = len(list(pred.glob("*.png")))
        if n == EXPECTED_N:
            resolved.append(pred)

    if len(resolved) == 1:
        log(f"[data:survol] resolved unambiguous SurVol folder: {resolved[0]}")
        return resolved[0]

    raise FileNotFoundError(
        f"[data:survol] could not unambiguously resolve the SurVol epoch-7 test50 prediction "
        f"folder.\nExpected: {SURVOL_DIR_EXPECTED}\n"
        f"Directory name candidates: {[str(p) for p in dir_candidates]}\n"
        f"Candidates with exactly {EXPECTED_N} PNGs: {[str(p) for p in resolved]}\n"
        f"Refusing to silently use a different GAN checkpoint."
    )


def list_pngs(d: Path):
    if not d.exists():
        raise FileNotFoundError(f"Expected folder not found: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
    if not files:
        raise FileNotFoundError(f"No PNG files found in {d}")
    return files


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)


def check_dir(label, d, expected_n, expected_size):
    files = list_pngs(d)
    log(f"[data:{label}] folder = {d}")
    log(f"[data:{label}] n_images = {len(files)}")
    log(f"[data:{label}] first files: {[f.name for f in files[:3]]}")
    if len(files) != expected_n:
        raise RuntimeError(f"[data:{label}] expected exactly {expected_n} images, found {len(files)}")
    shapes = set()
    for f in files[:3]:
        arr = load_gray(f)
        shapes.add(arr.shape)
        log(f"[data:{label}]   {f.name}: shape={arr.shape}")
    for shp in shapes:
        if shp != expected_size:
            raise RuntimeError(f"[data:{label}] expected shape {expected_size}, found {shp}")
    return files


def build_id_maps(hr_files, lr_files, resshift_files, survol_files):
    """Align by basename where possible; fall back to per_image_metrics.csv
    file-reference columns for SurVol if its basenames don't match HR's."""
    hr_ids = [p.stem for p in hr_files]
    lr_ids = [p.stem for p in lr_files]
    resshift_ids = [p.stem for p in resshift_files]

    if lr_ids != hr_ids:
        raise RuntimeError(f"[align] LR ids do not match HR ids.\nHR[:5]={hr_ids[:5]}\nLR[:5]={lr_ids[:5]}")
    if resshift_ids != hr_ids:
        raise RuntimeError(f"[align] ResShift ids do not match HR ids.\n"
                            f"HR[:5]={hr_ids[:5]}\nResShift[:5]={resshift_ids[:5]}")
    log("[align] HR / LR / ResShift ids match exactly.")

    survol_by_stem = {p.stem: p for p in survol_files}
    if set(survol_by_stem) == set(hr_ids) and len(survol_files) == len(hr_ids):
        log("[align] SurVol ids match HR ids directly by basename.")
        survol_ordered = [survol_by_stem[i] for i in hr_ids]
        return hr_ids, survol_ordered, "direct basename match"

    log("[align] SurVol basenames do not directly match HR ids -> trying per_image_metrics.csv "
        "file-reference columns")
    if not PER_IMAGE_CSV.exists():
        raise RuntimeError(f"[align] cannot align SurVol: basenames differ and {PER_IMAGE_CSV} is missing.")
    df = pd.read_csv(PER_IMAGE_CSV)
    survol_file_col = None
    for c in df.columns:
        cn = _norm(c)
        if ("survol" in cn or "gan" in cn) and ("file" in cn or "path" in cn or "image" in cn):
            survol_file_col = c
            break
    if survol_file_col is None:
        raise RuntimeError(
            f"[align] no SurVol file-reference column found in {PER_IMAGE_CSV}. "
            f"Available columns: {list(df.columns)}"
        )
    id_col = pick_col(df, ["image_id", "id", "sample_id", "stem"], "image id", required=False)
    if id_col is None:
        raise RuntimeError(f"[align] no image-id column found in {PER_IMAGE_CSV} to pair with "
                            f"'{survol_file_col}'. Available columns: {list(df.columns)}")

    id_to_survol_basename = {}
    for _, row in df.iterrows():
        iid = _canon_id(row[id_col])
        basename = Path(str(row[survol_file_col])).name
        id_to_survol_basename[iid] = basename

    survol_ordered = []
    for iid in hr_ids:
        canon = _canon_id(iid)
        key = canon if canon in id_to_survol_basename else None
        if key is None:
            for cand in id_to_survol_basename:
                if _norm(cand) == _norm(canon) or _norm(cand).endswith(_norm(canon)):
                    key = cand
                    break
        if key is None:
            raise RuntimeError(f"[align] could not find a SurVol file reference for HR id '{iid}' "
                                f"in {PER_IMAGE_CSV} column '{survol_file_col}'.")
        basename = id_to_survol_basename[key]
        stem = Path(basename).stem
        if stem not in survol_by_stem:
            raise RuntimeError(f"[align] {PER_IMAGE_CSV} references SurVol file '{basename}' for id "
                                f"'{iid}', but no such file exists in the resolved SurVol folder.")
        survol_ordered.append(survol_by_stem[stem])

    log(f"[align] SurVol aligned to HR ids via {PER_IMAGE_CSV.name} column '{survol_file_col}'.")
    return hr_ids, survol_ordered, f"per_image_metrics.csv column '{survol_file_col}'"


# ============================================================================
# 7. CARD / LABEL / AXIS PRIMITIVES -- reused verbatim from Figures 4.1/4.2
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
# 8. SANITY CHECKS AGAINST main_table_candidates.csv
# ============================================================================


def load_main_table():
    if not MAIN_TABLE_CSV.exists():
        raise FileNotFoundError(f"Official main table not found: {MAIN_TABLE_CSV}")
    df = pd.read_csv(MAIN_TABLE_CSV)
    log(f"[metrics] {MAIN_TABLE_CSV.name}: {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


def sanity_check_main_table(df):
    method_col = pick_col(df, ["method", "model", "name", "key", "candidate"], "method")
    metric_candidates = {
        "psnr": ["psnr", "psnr_mean"],
        "ssim": ["ssim", "ssim_mean"],
        "phase_fraction_error": ["phase_fraction_error", "phase_fraction_abs_error", "phasefractionerror"],
        "iou": ["iou", "iou_mean"],
        "dice": ["dice", "dice_mean"],
        "local_porosity_w32": ["local_porosity_w32", "local_porosity_w32_mae", "local_porosity_w32_mean"],
        "interface_density_error": ["interface_density_error", "interface_density_abs_error",
                                     "desc_abs_error__interface_density"],
        "tpcf_mae": ["tpcf_mae", "radial_tpcf_mae", "curve_error__tpcf_radial_phase1_mae"],
        "psd_mae": ["psd_mae", "radial_psd_mae", "curve_error__psd_radial_phase1_mae"],
        "lr_consistency_psnr": ["lr_consistency_psnr", "lr_consistency"],
    }
    resolved_cols = {k: pick_col(df, v, k, required=False) for k, v in metric_candidates.items()}

    for method_key, expected in SANITY_TABLE.items():
        row = pick_row(df, method_col, method_key)
        if row is None:
            raise RuntimeError(f"[sanity-check] no row for method '{method_key}' in {MAIN_TABLE_CSV}")
        bad = []
        for metric, exp_val in expected.items():
            col = resolved_cols.get(metric)
            if col is None:
                continue
            got = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(got):
                continue
            denom = max(abs(exp_val), 1e-12)
            if abs(got - exp_val) / denom > SANITY_REL_TOL:
                bad.append((metric, float(got), exp_val))
        if bad:
            detail = ", ".join(f"{m}: got {g:.6g} vs expected {e:.6g}" for m, g, e in bad)
            raise RuntimeError(
                f"[sanity-check] method '{method_key}' deviates from the official Table 4.8 "
                f"values by more than {SANITY_REL_TOL:.0%} relative ({detail}). This suggests "
                f"the wrong metric directory or method mapping was used."
            )
        log(f"[sanity-check] method '{method_key}' OK against Table 4.8 (tol {SANITY_REL_TOL:.0%})")
    return resolved_cols, method_col


# ============================================================================
# 9. REPRESENTATIVE IMAGE SELECTION
# ============================================================================

REP_METRIC_CANDIDATES = {
    "psnr": ["psnr"],
    "phase_fraction_abs_error": ["phase_fraction_abs_error", "phase_fraction_error"],
    "local_porosity_w32_map_mae": ["local_porosity_w32_map_mae", "local_porosity_w32_mae"],
    "desc_abs_error__interface_density": ["desc_abs_error__interface_density", "interface_density_error"],
    "curve_error__tpcf_radial_phase1_mae": ["curve_error__tpcf_radial_phase1_mae", "tpcf_mae"],
    "curve_error__psd_radial_phase1_mae": ["curve_error__psd_radial_phase1_mae", "psd_mae"],
}


def choose_representative_image(hr_ids):
    if not PER_IMAGE_CSV.exists():
        raise FileNotFoundError(f"Cannot select a representative image: {PER_IMAGE_CSV} not found.")
    df = pd.read_csv(PER_IMAGE_CSV)
    log(f"[rep] {PER_IMAGE_CSV.name}: {df.shape[0]} rows, columns: {list(df.columns)}")

    method_col = pick_col(df, ["method", "model", "name", "key", "candidate"], "method", required=False)
    id_col = pick_col(df, ["image_id", "id", "sample_id", "stem"], "image id")

    df["_method"] = df[method_col].map(CSV_METHOD_MAP) if method_col else None
    df["_id"] = df[id_col].map(_canon_id)

    resolved_metrics = {}
    for metric, cands in REP_METRIC_CANDIDATES.items():
        col = pick_col(df, cands, metric, required=False)
        if col is not None:
            resolved_metrics[metric] = col
    if not resolved_metrics:
        raise RuntimeError(f"[rep] none of the expected representative-selection metrics were found "
                            f"in {PER_IMAGE_CSV}. Available columns: {list(df.columns)}")
    log(f"[rep] using metrics: {list(resolved_metrics.keys())}")

    ids = [i for i in hr_ids]
    canon_ids = [_canon_id(i) for i in ids]
    scores = np.zeros(len(ids))
    n_terms = np.zeros(len(ids))
    any_match = False

    for method_key in LEARNED_METHODS:
        sub = df[df["_method"] == method_key].set_index("_id")
        for metric, col in resolved_metrics.items():
            vals_by_id = pd.to_numeric(sub[col], errors="coerce") if col in sub.columns else None
            if vals_by_id is None or vals_by_id.dropna().empty:
                continue
            arr = vals_by_id.reindex(canon_ids).to_numpy(float)
            if np.all(np.isnan(arr)):
                log(f"[rep] WARNING: metric '{metric}' for method '{method_key}' matched none of the "
                    f"HR image ids (even after leading-zero canonicalisation) -- skipping this term "
                    f"rather than letting it silently zero out the whole score")
                continue
            any_match = True
            med = np.nanmedian(arr)
            mad = np.nanmedian(np.abs(arr - med))
            spread = mad * 1.4826 if mad > 0 else (np.nanstd(arr) or 1.0)
            spread = spread if spread > 1e-12 else 1.0
            z = np.nan_to_num(np.abs((arr - med) / spread), nan=0.0)
            scores += z
            n_terms += (~np.isnan(arr)).astype(float)

    if not any_match:
        raise RuntimeError(
            f"[rep] none of the representative-selection metrics could be matched to any HR image id "
            f"in {PER_IMAGE_CSV} (column '{id_col}') -- refusing to silently fall back to the first "
            f"image. Check that this column actually contains the same image ids as the HR filenames."
        )

    n_terms[n_terms == 0] = 1.0
    combined = scores / n_terms
    best_idx = int(np.argmin(combined))
    best_id = ids[best_idx]
    best_canon = canon_ids[best_idx]

    log(f"[rep] selected representative image id = {best_id}  (score = {combined[best_idx]:.4f})")
    for method_key in LEARNED_METHODS:
        sub = df[(df["_method"] == method_key) & (df["_id"] == best_canon)]
        if not sub.empty:
            vals = {m: float(pd.to_numeric(sub.iloc[0][c], errors="coerce"))
                    for m, c in resolved_metrics.items() if c in sub.columns}
            log(f"[rep]   {method_key}: {vals}")
    return best_id, float(combined[best_idx])


# ============================================================================
# 10. ROI SELECTION (objective, boundary-density based)
# ============================================================================


def boundary_map(binary: np.ndarray) -> np.ndarray:
    b = np.zeros_like(binary, dtype=bool)
    b[:-1, :] |= binary[:-1, :] != binary[1:, :]
    b[1:, :] |= binary[:-1, :] != binary[1:, :]
    b[:, :-1] |= binary[:, :-1] != binary[:, 1:]
    b[:, 1:] |= binary[:, :-1] != binary[:, 1:]
    return b


def choose_roi(hr_binary: np.ndarray):
    h, w = hr_binary.shape
    bmap = boundary_map(hr_binary)
    xs = [x for x in range(0, w - ROI_SIZE + 1, ROI_STRIDE) if x != 0 and x != (w - ROI_SIZE)]
    ys = [y for y in range(0, h - ROI_SIZE + 1, ROI_STRIDE) if y != 0 and y != (h - ROI_SIZE)]
    if not xs or not ys:
        xs = list(range(0, w - ROI_SIZE + 1, ROI_STRIDE))
        ys = list(range(0, h - ROI_SIZE + 1, ROI_STRIDE))

    best = None
    for y0 in ys:
        row_sum = bmap[y0:y0 + ROI_SIZE, :].astype(np.int32)
        for x0 in xs:
            score = row_sum[:, x0:x0 + ROI_SIZE].sum() / (ROI_SIZE * ROI_SIZE)
            if best is None or score > best[0]:
                best = (score, x0, y0)

    score, x0, y0 = best
    log(f"[roi] selected ROI: x0={x0} y0={y0} width={ROI_SIZE} height={ROI_SIZE} "
        f"boundary_density={score:.4f}")
    return x0, y0, ROI_SIZE, ROI_SIZE, score


# ============================================================================
# 11. PANEL A / B -- image strips
# ============================================================================


def draw_image_strip(fig, card, cols, images_full, titles_on_top, roi=None):
    x_, y_, w_, h_ = card
    pad_x, pad_top, pad_bot = 0.014, 0.034 if titles_on_top else 0.016, 0.014
    gap_x = 0.014
    n = len(cols)

    free_w_in = (w_ - 2 * pad_x - (n - 1) * gap_x) * FIG_W / n
    free_h_in = (h_ - pad_top - pad_bot) * FIG_H
    cell_in = min(free_w_in, free_h_in)
    cell_w, cell_h = cell_in / FIG_W, cell_in / FIG_H

    grid_w = n * cell_w + (n - 1) * gap_x
    gx0 = x_ + (w_ - grid_w) / 2.0
    gy0 = y_ + pad_bot + max(0.0, (h_ - pad_top - pad_bot - cell_h) / 2.0)

    axes = {}
    for i, key in enumerate(cols):
        cx = gx0 + i * (cell_w + gap_x)
        if titles_on_top:
            fig.text(cx + cell_w / 2.0, gy0 + cell_h + 0.010, LABELS[key],
                     ha="center", va="bottom", fontsize=9.2, fontweight="bold",
                     color=TITLE_COLOR[key])
        a = fig.add_axes([cx, gy0, cell_w, cell_h])
        img = images_full[key]
        interp = "nearest" if key == "lr" else "bilinear"
        a.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation=interp)
        image_cell(a, COLORS[key])
        axes[key] = a

        if roi is not None and key == "hr":
            x0, y0, rw, rh, _ = roi
            a.add_patch(Rectangle((x0, y0), rw, rh, fill=False,
                                   edgecolor=TEXT, linewidth=1.3, zorder=10))
    return axes


# ============================================================================
# 12. PANEL C -- binary phase-error maps
# ============================================================================


def draw_phase_error_maps(fig, card, cols, hr_binary_roi, pred_binary_by_key):
    x_, y_, w_, h_ = card
    pad_x, pad_top, pad_bot = 0.018, 0.036, 0.020
    gap_x = 0.020
    n = len(cols)

    free_w_in = (w_ - 2 * pad_x - (n - 1) * gap_x) * FIG_W / n
    free_h_in = (h_ - pad_top - pad_bot) * FIG_H
    cell_in = min(free_w_in, free_h_in)
    cell_w, cell_h = cell_in / FIG_W, cell_in / FIG_H

    grid_w = n * cell_w + (n - 1) * gap_x
    gx0 = x_ + (w_ - grid_w) / 2.0
    gy0 = y_ + pad_bot

    for i, key in enumerate(cols):
        cx = gx0 + i * (cell_w + gap_x)
        fig.text(cx + cell_w / 2.0, gy0 + cell_h + 0.010, LABELS[key],
                 ha="center", va="bottom", fontsize=9.0, fontweight="bold",
                 color=TITLE_COLOR[key])

        pred = pred_binary_by_key[key]
        mismatch = pred != hr_binary_roi
        rgb = np.empty(mismatch.shape + (3,), dtype=np.uint8)
        light = np.array([0xF3, 0xF0, 0xF8], dtype=np.uint8)
        mcolor = np.array([int(COLORS[key][1:3], 16), int(COLORS[key][3:5], 16),
                            int(COLORS[key][5:7], 16)], dtype=np.uint8)
        rgb[~mismatch] = light
        rgb[mismatch] = mcolor

        a = fig.add_axes([cx, gy0, cell_w, cell_h])
        a.imshow(rgb, interpolation="nearest")
        image_cell(a, COLORS[key])

        pct = 100.0 * mismatch.mean()
        a.text(0.03, 0.03, f"ROI mismatch = {pct:.2f}%", transform=a.transAxes,
               fontsize=6.4, color=TEXT, ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4))

    legend_y = y_ + h_ - 0.014
    fig.add_artist(Rectangle((gx0, legend_y - 0.006), 0.011, 0.010, transform=fig.transFigure,
                              facecolor="#F3F0F8", edgecolor=SPINE, linewidth=0.6, zorder=101))
    fig.text(gx0 + 0.016, legend_y, "correct phase", ha="left", va="center",
             fontsize=6.6, color=SUBTEXT, zorder=101)
    fig.add_artist(Rectangle((gx0 + 0.105, legend_y - 0.006), 0.011, 0.010, transform=fig.transFigure,
                              facecolor=SUBTEXT, edgecolor=SPINE, linewidth=0.6, zorder=101))
    fig.text(gx0 + 0.121, legend_y, "phase mismatch (method color)", ha="left", va="center",
             fontsize=6.6, color=SUBTEXT, zorder=101)


# ============================================================================
# 13. PANEL D -- representative metric bars
# ============================================================================

BAR_METRIC_SPECS = [
    ("psnr", "PSNR ↑", None),
    ("ssim", "SSIM ↑", None),
    ("phase_fraction_error", "Phase-fraction error ↓", "sci"),
    ("interface_density_error", "Interface-density error ↓", "sci"),
]


def plot_metric_bars(fig, card, main_df, resolved_cols, method_col):
    x_, y_, w_, h_ = card
    pl, pr, pt, pb = 0.048, 0.018, 0.040, 0.040
    gx, gy = 0.040, 0.052

    plot_w = (w_ - pl - pr - gx) / 2.0
    plot_h = (h_ - pt - pb - gy) / 2.0

    std_candidates = {
        "psnr": ["psnr_std", "psnr_sd"],
        "ssim": ["ssim_std", "ssim_sd"],
        "phase_fraction_error": ["phase_fraction_error_std", "phase_fraction_abs_error_std"],
        "interface_density_error": ["interface_density_error_std", "desc_abs_error__interface_density_std"],
    }

    for i, (metric, title, fmt) in enumerate(BAR_METRIC_SPECS):
        r, c = divmod(i, 2)
        x = x_ + pl + c * (plot_w + gx)
        y = y_ + h_ - pt - (r + 1) * plot_h - r * gy
        ax = fig.add_axes([x, y, plot_w, plot_h])
        style_axis(ax, grid=False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_title(title, pad=4.0, color=TEXT, fontweight="bold")

        col = resolved_cols.get(metric)
        if col is None:
            ax.text(0.5, 0.5, "metric unavailable", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color=SUBTEXT)
            continue

        std_col = pick_col(main_df, std_candidates.get(metric, []), f"{metric}_std", required=False)

        vals, errs = [], []
        for m in BAR_METHODS:
            row = pick_row(main_df, method_col, m)
            v = float(pd.to_numeric(row[col], errors="coerce")) if row is not None else np.nan
            vals.append(v)
            if row is not None and std_col is not None:
                errs.append(float(pd.to_numeric(row[std_col], errors="coerce")))
            else:
                errs.append(0.0)

        xs = np.arange(len(BAR_METHODS))
        ax.bar(xs, vals, yerr=errs, width=0.56, color=[COLORS[m] for m in BAR_METHODS],
               edgecolor="black", linewidth=0.7,
               error_kw=dict(ecolor=TEXT, elinewidth=1.0, capsize=2.5))
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in BAR_METHODS], fontsize=7.2)
        if fmt == "sci":
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))
        ax.margins(y=0.18)


# ============================================================================
# 14. PANEL E -- radial TPCF / PSD curves
# ============================================================================


def plot_curve_group(ax, curves, title, xlabel, ylabel):
    style_axis(ax)
    ax.set_title(title, pad=4.5, color=TEXT, fontweight="bold")
    ax.set_xlabel(xlabel, color=SUBTEXT)
    ax.set_ylabel(ylabel, color=SUBTEXT)

    if curves is None:
        ax.text(0.5, 0.5, "metric unavailable\n(no matching evaluator function)",
                transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=SUBTEXT)
        return

    for key in ["hr", "bicubic", "resshift", "survol"]:
        if key not in curves:
            continue
        x, y, std = curves[key]["x"], curves[key]["y"], curves[key].get("std")
        if key == "hr" and std is not None and np.any(std > 0):
            ax.fill_between(x, y - std, y + std, color=COLORS[key], alpha=0.20, linewidth=0, zorder=1)
        ax.plot(x, y, color=COLORS[key], linestyle=LINESTYLES[key], linewidth=LINEWIDTHS[key],
                 solid_capstyle="round", label=LABELS[key], zorder=3 if key == "hr" else 4)

    ax.margins(x=0.02, y=0.06)
    leg = ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=SPINE,
                     framealpha=0.94, handlelength=2.3, borderpad=0.45, labelspacing=0.3)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(8)


def compute_group_curves(evaluator, binaries_by_key):
    """binaries_by_key: {key: [binary_img, ...]} for hr/bicubic/resshift/survol."""
    if evaluator.tpcf_fn is None and evaluator.psd_fn is None:
        return None, None

    tpcf_curves, psd_curves = {}, {}
    for key, binaries in binaries_by_key.items():
        tpcf_ys, psd_ys = [], []
        x_tpcf = x_psd = None
        for b in binaries:
            if evaluator.tpcf_fn is not None:
                x, y = evaluator.tpcf(b)
                x_tpcf = x if x_tpcf is None else x_tpcf
                tpcf_ys.append(y)
            if evaluator.psd_fn is not None:
                x, y = evaluator.psd(b)
                x_psd = x if x_psd is None else x_psd
                psd_ys.append(y)
        if tpcf_ys:
            arr = np.array(tpcf_ys)
            tpcf_curves[key] = {"x": x_tpcf, "y": arr.mean(axis=0),
                                 "std": arr.std(axis=0) if key == "hr" else None}
        if psd_ys:
            arr = np.array(psd_ys)
            psd_curves[key] = {"x": x_psd, "y": arr.mean(axis=0),
                                "std": arr.std(axis=0) if key == "hr" else None}

    return (tpcf_curves if tpcf_curves else None), (psd_curves if psd_curves else None)


# ============================================================================
# 15. BUILD
# ============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"[paths] PROJECT       = {PROJECT}")
    log(f"[paths] HR_DIR        = {HR_DIR}")
    log(f"[paths] LR_DIR        = {LR_DIR}")
    log(f"[paths] RESSHIFT_DIR  = {RESSHIFT_DIR}")
    log(f"[paths] METRICS_DIR   = {METRICS_DIR}")

    for p in (RESSHIFT_DIR,):
        if _path_is_forbidden(p):
            raise RuntimeError(f"[guard] refusing forbidden path: {p}")
    survol_dir = resolve_survol_dir()

    hr_files = check_dir("hr", HR_DIR, EXPECTED_N, EXPECTED_HR_SIZE)
    lr_files = check_dir("lr", LR_DIR, EXPECTED_N, EXPECTED_LR_SIZE)
    resshift_files = check_dir("resshift", RESSHIFT_DIR, EXPECTED_N, EXPECTED_HR_SIZE)
    survol_files = check_dir("survol", survol_dir, EXPECTED_N, EXPECTED_HR_SIZE)

    hr_ids, survol_files_ordered, align_method = build_id_maps(
        hr_files, lr_files, resshift_files, survol_files)
    log(f"[align] SurVol alignment method: {align_method}")

    evaluator = Evaluator(load_evaluator_module())
    log(f"[evaluator] PHASE_THRESHOLD (nominal) = {PHASE_THRESHOLD}, MAX_LAG = {MAX_LAG}")

    main_df = load_main_table()
    resolved_cols, method_col = sanity_check_main_table(main_df)

    # ---- representative image ------------------------------------------
    rep_id, rep_score = choose_representative_image(hr_ids)
    rep_idx = hr_ids.index(rep_id)

    hr_full = load_gray(hr_files[rep_idx])
    lr_full = load_gray(lr_files[rep_idx])
    resshift_full = load_gray(resshift_files[rep_idx])
    survol_full = load_gray(survol_files_ordered[rep_idx])
    bicubic_full = evaluator.bicubic(lr_full, out_size=hr_full.shape)

    images_full = {"lr": lr_full, "hr": hr_full, "bicubic": bicubic_full,
                   "resshift": resshift_full, "survol": survol_full}

    # ---- ROI --------------------------------------------------------------
    hr_binary_full = evaluator.binarize(hr_full)
    x0, y0, rw, rh, roi_score = choose_roi(hr_binary_full)
    roi = (x0, y0, rw, rh, roi_score)

    def crop(img):
        return img[y0:y0 + rh, x0:x0 + rw]

    lr_x0, lr_y0, lr_w, lr_h = x0 // 4, y0 // 4, rw // 4, rh // 4
    lr_crop = lr_full[lr_y0:lr_y0 + lr_h, lr_x0:lr_x0 + lr_w]
    lr_crop_big = np.array(Image.fromarray(lr_crop).resize((rw, rh), Image.NEAREST))

    images_roi = {"lr": lr_crop_big, "hr": crop(hr_full), "bicubic": crop(bicubic_full),
                  "resshift": crop(resshift_full), "survol": crop(survol_full)}

    hr_binary_roi = crop(hr_binary_full)
    pred_binary_roi = {k: evaluator.binarize(images_roi[k]) for k in PANEL_C_COLS}

    # ---- panel-e curves (full 50-image sets) -------------------------------
    log("[panel-e] computing radial TPCF/PSD curves from the 50-image official test set "
        "using the evaluator's own functions ...")
    binaries_by_key = {"hr": [], "bicubic": [], "resshift": [], "survol": []}
    for i in range(EXPECTED_N):
        hr_i = load_gray(hr_files[i])
        binaries_by_key["hr"].append(evaluator.binarize(hr_i))
        lr_i = load_gray(lr_files[i])
        bic_i = evaluator.bicubic(lr_i, out_size=hr_i.shape)
        binaries_by_key["bicubic"].append(evaluator.binarize(bic_i))
        binaries_by_key["resshift"].append(evaluator.binarize(load_gray(resshift_files[i])))
        binaries_by_key["survol"].append(evaluator.binarize(load_gray(survol_files_ordered[i])))
    tpcf_curves, psd_curves = compute_group_curves(evaluator, binaries_by_key)

    # ---- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for card in (card_a, card_b, card_c, card_d, card_e):
        add_card(fig, card)

    add_panel_label(fig, card_a[0] + 0.007, card_a[1] + card_a[3] + PANEL_LABEL_OFFSET, "a)")
    add_panel_label(fig, card_b[0] + 0.007, card_b[1] + card_b[3] + PANEL_LABEL_OFFSET, "b)")
    add_panel_label(fig, card_c[0] + 0.007, card_c[1] + card_c[3] + PANEL_LABEL_OFFSET, "c)")
    add_panel_label(fig, card_d[0] + 0.007, card_d[1] + card_d[3] + PANEL_LABEL_OFFSET, "d)")
    add_panel_label(fig, card_e[0] + 0.007, card_e[1] + card_e[3] + PANEL_LABEL_OFFSET, "e)")

    draw_image_strip(fig, card_a, PANEL_A_COLS, images_full, titles_on_top=True, roi=roi)
    draw_image_strip(fig, card_b, PANEL_B_COLS, images_roi, titles_on_top=True, roi=None)
    draw_phase_error_maps(fig, card_c, PANEL_C_COLS, hr_binary_roi, pred_binary_roi)
    plot_metric_bars(fig, card_d, main_df, resolved_cols, method_col)

    ex_, ey_, ew_, eh_ = card_e
    e_pl, e_pr, e_pt, e_pb, e_gx = 0.045, 0.020, 0.040, 0.046, 0.055
    e_plot_w = (ew_ - e_pl - e_pr - e_gx) / 2.0
    e_plot_h = eh_ - e_pt - e_pb
    ax_tpcf = fig.add_axes([ex_ + e_pl, ey_ + e_pb, e_plot_w, e_plot_h])
    ax_psd = fig.add_axes([ex_ + e_pl + e_plot_w + e_gx, ey_ + e_pb, e_plot_w, e_plot_h])
    plot_curve_group(ax_tpcf, tpcf_curves, "Radial TPCF", "Lag / radius (pixels)", r"$S_2(r)$")
    plot_curve_group(ax_psd, psd_curves, "Radial PSD", "Radial frequency bin", "Power")

    # =========================== SAVE ========================================
    png = OUT / f"{STEM}.png"
    pdf = OUT / f"{STEM}.pdf"
    svg = OUT / f"{STEM}.svg"
    tif_out = OUT / f"{STEM}.tiff"

    fig.savefig(png, dpi=450, facecolor=BG)
    fig.savefig(pdf, facecolor=BG)
    fig.savefig(svg, facecolor=BG)
    Image.open(png).convert("RGB").save(tif_out, compression="tiff_lzw", dpi=(450, 450))
    plt.close(fig)

    log("\nSaved:")
    for p in (png, pdf, svg, tif_out):
        log(" ", p.resolve())


if __name__ == "__main__":
    sys.exit(main())
