#!/usr/bin/env python3
"""
validation.py

Validates a trilingual (English / Russian / Kazakh) readability assessment
system by comparing human readability judgments against automated
readability scores.

Usage:
    python validation.py <annotations.csv> <readability_scores.csv> [--output-dir ./results]

Inputs
------
annotations.csv:
    sentence_id, lang, sentence_text, human_rating (1-5), notes
    human_rating: 5 = Very Easy ... 1 = Very Difficult

readability_scores.csv:
    lang, sentence, readability_score (0-100), interpretation,
    tree_depth, char_density, avg_syllables, pos_ratio, token_count
    readability_score: 100 = Very Easy ... 0 = Very Difficult

Outputs (written to --output-dir)
----------------------------------
validation_summary.txt   - full console report, also printed to stdout
fig1_scatter_by_language.png  - human vs. automated score, one panel per language
fig2_distribution_comparison.png - histogram overlay of the two scales, per language
fig3_residual_plot.png   - residuals (human - predicted) vs. predicted score
outliers.csv             - rows where |human_scaled - readability_score| > 20
validation_report.json   - structured machine-readable results

Design notes / assumptions (documented since the spec left them open):
- The three correlation figures are saved as three separate PNG files
  (fig1/fig2/fig3) rather than one combined image, since they show very
  different things and are far more usable separately.
- Merge key is (lang, whitespace-normalized sentence_text) rather than text
  alone, to avoid accidental cross-language collisions.
- Confidence intervals for Pearson r and Spearman rho are computed by
  case-resampling bootstrap (2000 resamples) rather than the Fisher
  z-transform, since it is valid for both statistics without extra
  assumptions.
- "error" in outliers.csv / the residual plot is defined as
  (human_scaled - readability_score): positive means the algorithm rated
  the sentence as harder than the human did.
- Outliers are sorted by absolute error, descending (worst mismatches
  first), while keeping the signed error in the "error" column.
- A language's mean error is flagged as a systematic bias if
  |mean error| > 3 points (a small tolerance around zero to avoid flagging
  noise from tiny sample sizes).
"""

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

REQUIRED_ANNOTATION_COLS = ["sentence_id", "lang", "sentence_text", "human_rating"]
REQUIRED_SCORE_COLS = [
    "lang", "sentence", "readability_score", "interpretation",
    "tree_depth", "char_density", "avg_syllables", "pos_ratio", "token_count",
]
FEATURE_COLS = ["tree_depth", "char_density", "avg_syllables", "pos_ratio"]

OUTLIER_THRESHOLD = 20.0          # points, on the 0-100 scale
UNMATCHED_WARN_PCT = 5.0          # percent of annotation rows
BIAS_THRESHOLD = 3.0              # points, for flagging systematic bias
MIN_N_FOR_CORRELATION = 3
N_BOOTSTRAP = 2000
CI_LEVEL = 95
RANDOM_SEED = 42

LANG_ORDER = ["en", "ru", "kk"]
LANG_LABELS = {"en": "English", "ru": "Russian", "kk": "Kazakh"}
LANG_COLORS = {"en": "#4C72B0", "ru": "#DD8452", "kk": "#55A868"}


# ----------------------------------------------------------------------------
# Small utility: tee print output to console AND collect it for the .txt report
# ----------------------------------------------------------------------------

class Reporter:
    """Prints to stdout while also collecting everything into a buffer,
    so the exact console transcript can be written to validation_summary.txt."""

    def __init__(self):
        self._lines = []

    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        sep = kwargs.get("sep")
        if sep is not None:
            text = sep.join(str(a) for a in args)
        print(text)
        self._lines.append(text)

    def write_to_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")


# ----------------------------------------------------------------------------
# Loading & validation
# ----------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, required: list, source_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required column(s): {', '.join(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


def load_csvs(annotations_path: str, scores_path: str):
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not os.path.isfile(scores_path):
        raise FileNotFoundError(f"Readability scores file not found: {scores_path}")

    ann_df = pd.read_csv(annotations_path, encoding="utf-8")
    scores_df = pd.read_csv(scores_path, encoding="utf-8")

    require_columns(ann_df, REQUIRED_ANNOTATION_COLS, os.path.basename(annotations_path))
    require_columns(scores_df, REQUIRED_SCORE_COLS, os.path.basename(scores_path))

    if "notes" not in ann_df.columns:
        ann_df["notes"] = ""

    return ann_df, scores_df


def normalize_text(text) -> str:
    """Strip leading/trailing whitespace and collapse internal runs of
    whitespace to a single space. Case-sensitive (no case folding)."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_lang(lang) -> str:
    if pd.isna(lang):
        return ""
    return str(lang).strip().lower()


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------

def merge_datasets(ann_df: pd.DataFrame, scores_df: pd.DataFrame, reporter: Reporter):
    ann = ann_df.copy()
    scores = scores_df.copy()

    ann["_lang_norm"] = ann["lang"].apply(normalize_lang)
    scores["_lang_norm"] = scores["lang"].apply(normalize_lang)

    ann["_text_norm"] = ann["sentence_text"].apply(normalize_text)
    scores["_text_norm"] = scores["sentence"].apply(normalize_text)

    # Guard against fan-out: if the same (lang, normalized sentence) appears
    # more than once in the scores file, keep the first occurrence and warn.
    dup_mask = scores.duplicated(subset=["_lang_norm", "_text_norm"], keep="first")
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        reporter.print(
            f"[WARN] {n_dupes} duplicate (lang, sentence) entr{'y' if n_dupes==1 else 'ies'} "
            f"found in readability_scores.csv; keeping the first occurrence of each."
        )
    scores = scores[~dup_mask]

    merged = ann.merge(
        scores,
        on=["_lang_norm", "_text_norm"],
        how="left",
        suffixes=("", "_score"),
        indicator=True,
    )

    unmatched = merged[merged["_merge"] == "left_only"]
    n_unmatched = len(unmatched)
    n_total = len(ann)
    pct_unmatched = (n_unmatched / n_total * 100.0) if n_total else 0.0

    if n_unmatched:
        reporter.print(f"[WARN] {n_unmatched} of {n_total} annotation rows "
                        f"({pct_unmatched:.1f}%) had no matching sentence in "
                        f"readability_scores.csv and will be skipped.")
        example_ids = unmatched["sentence_id"].astype(str).head(5).tolist()
        reporter.print(f"        Example unmatched sentence_id(s): {', '.join(example_ids)}")
    if pct_unmatched > UNMATCHED_WARN_PCT:
        reporter.print(
            f"[WARN] Unmatched rate ({pct_unmatched:.1f}%) exceeds the "
            f"{UNMATCHED_WARN_PCT:.0f}% threshold -- check for text/formatting "
            f"mismatches between the two files."
        )

    matched = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

    if len(matched) == 0:
        raise RuntimeError("No matching sentences found between annotations.csv "
                            "and readability_scores.csv. Aborting.")

    return matched, n_unmatched, pct_unmatched


# ----------------------------------------------------------------------------
# Scale alignment / cleaning
# ----------------------------------------------------------------------------

def scale_and_clean(df: pd.DataFrame, reporter: Reporter):
    df = df.copy()
    df["human_scaled"] = (pd.to_numeric(df["human_rating"], errors="coerce") - 1.0) * 25.0
    df["readability_score"] = pd.to_numeric(df["readability_score"], errors="coerce")

    for col in FEATURE_COLS + ["token_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    check_cols = ["human_scaled", "readability_score"]
    bad_mask = df[check_cols].apply(
        lambda s: ~np.isfinite(s.astype(float))
    ).any(axis=1)
    n_bad = int(bad_mask.sum())
    if n_bad:
        reporter.print(f"[WARN] Dropping {n_bad} row(s) with NaN/inf values in "
                        f"human_rating or readability_score.")
    df = df[~bad_mask].reset_index(drop=True)

    df["error"] = df["human_scaled"] - df["readability_score"]
    df["abs_error"] = df["error"].abs()

    return df


# ----------------------------------------------------------------------------
# Correlation metrics
# ----------------------------------------------------------------------------

def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = np.arange(n)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        xs, ys = x[sample], y[sample]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = stat_fn(xs, ys)
        if np.isfinite(val):
            boots.append(val)
    if len(boots) < 10:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return (lo, hi)


def classify_strength(r: float) -> str:
    ar = abs(r) if np.isfinite(r) else 0.0
    if ar >= 0.7:
        return "Strong"
    elif ar >= 0.5:
        return "Moderate"
    else:
        return "Weak"


def compute_group_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    result = {"n": n}

    if n < MIN_N_FOR_CORRELATION:
        result["warning"] = (
            f"Insufficient data (n={n}) for reliable correlation statistics "
            f"(need at least {MIN_N_FOR_CORRELATION})."
        )
        for k in ["pearson_r", "pearson_p", "pearson_ci", "spearman_rho",
                  "spearman_p", "spearman_ci", "mae", "rmse", "r2",
                  "mean_error", "strength", "pearson_significant",
                  "spearman_significant"]:
            result[k] = None
        return result

    x = df["human_scaled"].to_numpy(dtype=float)   # "actual"
    y = df["readability_score"].to_numpy(dtype=float)  # "predicted"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pear_r, pear_p = stats.pearsonr(x, y)
        spear_rho, spear_p = stats.spearmanr(x, y)

    pear_ci = bootstrap_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0])
    spear_ci = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b)[0])

    errors = x - y
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mean_error = float(np.mean(errors))

    result.update({
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "pearson_ci": pear_ci,
        "pearson_significant": bool(pear_p <= 0.05),
        "spearman_rho": float(spear_rho),
        "spearman_p": float(spear_p),
        "spearman_ci": spear_ci,
        "spearman_significant": bool(spear_p <= 0.05),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_error": mean_error,
        "strength": classify_strength(pear_r),
    })
    return result


def compute_all_metrics(df: pd.DataFrame) -> dict:
    results = {"overall": compute_group_metrics(df)}
    for lang in LANG_ORDER:
        sub = df[df["_lang_norm"] == lang]
        if len(sub) == 0:
            continue
        results[lang] = compute_group_metrics(sub)
    # Include any languages present that weren't in LANG_ORDER
    for lang in df["_lang_norm"].unique():
        if lang not in results:
            sub = df[df["_lang_norm"] == lang]
            results[lang] = compute_group_metrics(sub)
    return results


# ----------------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------------

def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def fmt_ci(ci, decimals=3):
    if ci is None:
        return "n/a"
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def sig_note(p):
    if p is None or not np.isfinite(p):
        return ""
    return "" if p <= 0.05 else "  (not statistically significant)"


def print_summary_table(results: dict, reporter: Reporter):
    reporter.print("=" * 78)
    reporter.print("READABILITY VALIDATION SUMMARY")
    reporter.print("=" * 78)

    row_order = ["overall"] + [l for l in LANG_ORDER if l in results] + \
                [l for l in results if l not in ["overall"] + LANG_ORDER]

    header = f"{'Group':<10}{'n':>5}{'Pearson r':>12}{'p':>10}{'Spearman rho':>14}{'p':>10}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'Strength':>10}"
    reporter.print(header)
    reporter.print("-" * len(header))
    for key in row_order:
        m = results[key]
        label = "Overall" if key == "overall" else LANG_LABELS.get(key, key)
        if m.get("warning"):
            reporter.print(f"{label:<10}{m['n']:>5}   {m['warning']}")
            continue
        reporter.print(
            f"{label:<10}{m['n']:>5}"
            f"{fmt(m['pearson_r']):>12}"
            f"{fmt(m['pearson_p']):>10}"
            f"{fmt(m['spearman_rho']):>14}"
            f"{fmt(m['spearman_p']):>10}"
            f"{fmt(m['mae'],1):>9}"
            f"{fmt(m['rmse'],1):>9}"
            f"{fmt(m['r2']):>9}"
            f"{m['strength']:>10}"
        )
    reporter.print("")
    reporter.print(f"95% confidence intervals (bootstrap, n={N_BOOTSTRAP} resamples):")
    for key in row_order:
        m = results[key]
        if m.get("warning"):
            continue
        label = "Overall" if key == "overall" else LANG_LABELS.get(key, key)
        reporter.print(
            f"  {label:<10} Pearson r CI: {fmt_ci(m['pearson_ci'])}{sig_note(m['pearson_p'])}   "
            f"Spearman rho CI: {fmt_ci(m['spearman_ci'])}{sig_note(m['spearman_p'])}"
        )
    reporter.print("")


# ----------------------------------------------------------------------------
# Outliers
# ----------------------------------------------------------------------------

def build_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outliers = df[df["abs_error"] > OUTLIER_THRESHOLD].copy()
    outliers = outliers.sort_values("abs_error", ascending=False)

    cols = ["sentence_id", "lang", "sentence_text", "human_rating", "human_scaled",
            "readability_score", "error", "tree_depth", "char_density",
            "avg_syllables", "pos_ratio"]
    cols = [c for c in cols if c in outliers.columns]
    return outliers[cols]


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def _langs_present(df):
    return [l for l in LANG_ORDER if l in df["_lang_norm"].unique()] or \
           sorted(df["_lang_norm"].unique())


def plot_scatter_by_language(df: pd.DataFrame, results: dict, output_dir: str):
    langs = _langs_present(df)
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, lang in zip(axes, langs):
        sub = df[df["_lang_norm"] == lang]
        color = LANG_COLORS.get(lang, "#888888")
        sizes = 20 + 6 * sub["token_count"].fillna(sub["token_count"].median() if "token_count" in sub else 10)
        ax.scatter(sub["human_scaled"], sub["readability_score"],
                   s=sizes, c=color, alpha=0.65, edgecolors="white", linewidths=0.5)
        ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, label="Perfect agreement (y=x)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Readability Score (human, scaled)")
        ax.set_ylabel("Readability Score (automated)")

        m = results.get(lang, {})
        if m.get("warning"):
            title = f"{LANG_LABELS.get(lang, lang)}\n(n={m['n']}, insufficient data)"
        else:
            title = (f"{LANG_LABELS.get(lang, lang)}  (n={m['n']})\n"
                     f"r={fmt(m['pearson_r'])}, p={fmt(m['pearson_p'])}")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Human vs. Automated Readability Scores by Language", y=1.02, fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_scatter_by_language.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distribution_comparison(df: pd.DataFrame, output_dir: str):
    langs = _langs_present(df)
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    bins = np.linspace(0, 100, 21)
    for ax, lang in zip(axes, langs):
        sub = df[df["_lang_norm"] == lang]
        ax.hist(sub["human_scaled"], bins=bins, alpha=0.5, color="tab:blue", label="Human (scaled)")
        ax.hist(sub["readability_score"], bins=bins, alpha=0.5, color="tab:orange", label="Automated")

        h_mean, h_med = sub["human_scaled"].mean(), sub["human_scaled"].median()
        r_mean, r_med = sub["readability_score"].mean(), sub["readability_score"].median()
        ax.axvline(h_mean, color="tab:blue", linestyle="-", linewidth=1.5)
        ax.axvline(h_med, color="tab:blue", linestyle=":", linewidth=1.5)
        ax.axvline(r_mean, color="tab:orange", linestyle="-", linewidth=1.5)
        ax.axvline(r_med, color="tab:orange", linestyle=":", linewidth=1.5)

        ax.set_title(f"{LANG_LABELS.get(lang, lang)} (n={len(sub)})")
        ax.set_xlabel("Readability Score (0-100)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution: Human vs. Automated Scores (solid=mean, dotted=median)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_distribution_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residuals(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    normal = df[df["abs_error"] <= OUTLIER_THRESHOLD]
    outliers = df[df["abs_error"] > OUTLIER_THRESHOLD]

    for lang in _langs_present(df):
        sub = normal[normal["_lang_norm"] == lang]
        ax.scatter(sub["readability_score"], sub["error"],
                   c=LANG_COLORS.get(lang, "#888888"), alpha=0.6,
                   label=LANG_LABELS.get(lang, lang), s=35, edgecolors="white", linewidths=0.4)

    if len(outliers):
        ax.scatter(outliers["readability_score"], outliers["error"],
                   facecolors="none", edgecolors="red", s=110, linewidths=1.6,
                   label=f"Outlier (|error| > {OUTLIER_THRESHOLD:.0f})")

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(OUTLIER_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-OUTLIER_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Readability Score (automated, predicted)")
    ax.set_ylabel("Error (human_scaled - automated)")
    ax.set_title("Residual Plot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_residual_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Per-language insights
# ----------------------------------------------------------------------------

def recommendation_for(m: dict) -> str:
    if m.get("warning"):
        return "Not enough data to make a recommendation."
    r = m["pearson_r"]
    if r >= 0.7:
        rec = "Strong alignment -- no tuning needed."
    elif r >= 0.5:
        rec = "Moderate alignment -- minor tuning may improve results."
    else:
        rec = "Weak alignment -- consider widening reference ranges or reweighting features."
    if r < 0.6:
        rec += " (r < 0.6: consider widening reference ranges or reweighting features.)"
    if abs(m["mean_error"]) > BIAS_THRESHOLD:
        direction = "over-rates (predicts easier than humans judge)" if m["mean_error"] < 0 \
            else "under-rates (predicts harder than humans judge)"
        rec += f" Algorithm shows a systematic bias: it consistently {direction} " \
               f"(mean error = {m['mean_error']:.1f} points)."
    return rec


def feature_ranges(sub: pd.DataFrame) -> dict:
    out = {}
    for col in FEATURE_COLS:
        if col in sub.columns and sub[col].notna().any():
            out[col] = {"mean": float(sub[col].mean()), "std": float(sub[col].std(ddof=0))}
    return out


def top_errors(sub: pd.DataFrame, k=3) -> list:
    top = sub.sort_values("abs_error", ascending=False).head(k)
    records = []
    for _, row in top.iterrows():
        records.append({
            "sentence_id": row.get("sentence_id"),
            "sentence_text": row.get("sentence_text"),
            "human_scaled": float(row["human_scaled"]),
            "readability_score": float(row["readability_score"]),
            "error": float(row["error"]),
        })
    return records


def print_per_language_insights(df: pd.DataFrame, results: dict, reporter: Reporter) -> dict:
    reporter.print("=" * 78)
    reporter.print("PER-LANGUAGE INSIGHTS")
    reporter.print("=" * 78)

    insights = {}
    for lang in _langs_present(df):
        sub = df[df["_lang_norm"] == lang]
        m = results.get(lang, {})
        reporter.print(f"\n-- {LANG_LABELS.get(lang, lang)} (n={len(sub)}) --")

        rec = recommendation_for(m)
        reporter.print(f"  Recommendation: {rec}")

        te = top_errors(sub)
        reporter.print("  Largest mismatches:")
        if te:
            for e in te:
                text_preview = str(e["sentence_text"])
                if len(text_preview) > 60:
                    text_preview = text_preview[:57] + "..."
                reporter.print(
                    f"    id={e['sentence_id']}  error={e['error']:+.1f}  "
                    f"human={e['human_scaled']:.0f}  auto={e['readability_score']:.1f}  "
                    f"\"{text_preview}\""
                )
        else:
            reporter.print("    (none)")

        franges = feature_ranges(sub)
        reporter.print("  Feature ranges (mean +/- std):")
        for feat, stats_ in franges.items():
            reporter.print(f"    {feat}: {stats_['mean']:.2f} +/- {stats_['std']:.2f}")

        insights[lang] = {
            "n": len(sub),
            "recommendation": rec,
            "top_errors": te,
            "feature_ranges": franges,
        }

    reporter.print("")
    return insights


# ----------------------------------------------------------------------------
# JSON report
# ----------------------------------------------------------------------------

def build_json_report(results, insights, n_unmatched, pct_unmatched, n_dropped_nan, n_total_annotations):
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    report = {
        "match_summary": {
            "total_annotation_rows": n_total_annotations,
            "unmatched_rows": n_unmatched,
            "unmatched_pct": pct_unmatched,
            "rows_dropped_for_nan_inf": n_dropped_nan,
        },
        "metrics": results,
        "insights": insights,
    }
    return clean(report)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a trilingual readability system against human ratings."
    )
    parser.add_argument("annotations_csv", help="Path to annotations.csv")
    parser.add_argument("readability_scores_csv", help="Path to readability_scores.csv")
    parser.add_argument("--output-dir", default="./results", help="Directory for output files")
    parser.add_argument("--no-json", action="store_true", help="Skip writing validation_report.json")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    reporter = Reporter()
    sns.set_theme(style="whitegrid")

    try:
        ann_df, scores_df = load_csvs(args.annotations_csv, args.readability_scores_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    n_total_annotations = len(ann_df)

    try:
        matched, n_unmatched, pct_unmatched = merge_datasets(ann_df, scores_df, reporter)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    n_before_clean = len(matched)
    df = scale_and_clean(matched, reporter)
    n_dropped_nan = n_before_clean - len(df)

    if len(df) == 0:
        print("[ERROR] No matching sentences found (all rows dropped during cleaning).",
              file=sys.stderr)
        sys.exit(1)

    results = compute_all_metrics(df)
    print_summary_table(results, reporter)

    outliers = build_outliers(df)
    outliers_path = os.path.join(args.output_dir, "outliers.csv")
    outliers.to_csv(outliers_path, index=False, encoding="utf-8")
    reporter.print(f"Outlier report: {len(outliers)} sentence(s) with |error| > "
                    f"{OUTLIER_THRESHOLD:.0f} points -> {outliers_path}")
    reporter.print("")

    fig1_path = plot_scatter_by_language(df, results, args.output_dir)
    fig2_path = plot_distribution_comparison(df, args.output_dir)
    fig3_path = plot_residuals(df, args.output_dir)
    reporter.print("Saved figures:")
    reporter.print(f"  {fig1_path}")
    reporter.print(f"  {fig2_path}")
    reporter.print(f"  {fig3_path}")
    reporter.print("")

    insights = print_per_language_insights(df, results, reporter)

    summary_path = os.path.join(args.output_dir, "validation_summary.txt")
    reporter.write_to_file(summary_path)

    if not args.no_json:
        report = build_json_report(results, insights, n_unmatched, pct_unmatched,
                                    n_dropped_nan, n_total_annotations)
        json_path = os.path.join(args.output_dir, "validation_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
