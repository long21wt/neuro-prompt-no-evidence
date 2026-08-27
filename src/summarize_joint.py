import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid", "vibrant"])
except Exception:
    pass

os.environ["PATH"] += os.pathsep + "texlive/2025/bin/x86_64-linux"
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.figsize": (16, 9),
    "text.latex.preamble": r"\usepackage{mathptmx}",
})


CONDITIONS = ["c1", "c2", "c4"]
CLASSES = ["mdd", "control"]
SPLITS = ["train", "dev", "test"]

COND_LABELS = [
    r"$\textsc{text(csv)}$",
    r"$\textsc{text(csv)}$" + "\n" + r"$+ \textsc{prompt(MRI)}$",
    r"$\textsc{text(csv, parcel)}$" + "\n"
        + r"$+ \textsc{prompt(MRI)}$" + "\n"
        + r"$+ \textsc{plot(MRI)}$",
]
COND_KEYS = ["p_mdd_c1", "p_mdd_c2", "p_mdd_c4"]


def parse_filename(path):
    # joint_<model>_<cond>_<class>_<split>.jsonl -> dict, or None if not parseable.
    stem = Path(path).stem
    if not stem.startswith("joint_"):
        return None
    rest = stem[len("joint_"):]

    split = next((s for s in SPLITS if rest.endswith("_" + s)), None)
    if split is None:
        return None
    rest = rest[: -(len(split) + 1)]

    cls = next((c for c in CLASSES if rest.endswith("_" + c)), None)
    if cls is None:
        return None
    rest = rest[: -(len(cls) + 1)]

    cond = next((c for c in CONDITIONS if rest.endswith("_" + c)), None)
    if cond is None:
        return None
    rest = rest[: -(len(cond) + 1)]

    return {"model": rest, "condition": cond, "true_class": cls, "split": split}


def _coerce_pred_label(rec, true_class):
    for key in ("pred_label", "pred", "predicted_label"):
        v = rec.get(key)
        if v is None:
            continue
        v = str(v).strip().lower()
        if "control" in v:
            return "control"
        if "major" in v or "depress" in v or v == "mdd":
            return "mdd"
        return v
    return None


def _coerce_correct(rec, pred, true_class):
    if "correct" in rec and rec["correct"] is not None:
        try:
            return int(bool(rec["correct"]))
        except Exception:
            pass
    if pred is None:
        return None
    return int(pred == true_class)


def load_jsonl_records(path, meta):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pred = _coerce_pred_label(r, meta["true_class"])
            rows.append({
                **meta,
                "filename": r.get("filename"),
                "p_mdd_norm": r.get("p_mdd_norm"),
                "p_control_norm": r.get("p_control_norm"),
                "p_major_at_label": r.get("p_major_at_label"),
                "p_control_at_label": r.get("p_control_at_label"),
                "log_p_prefix": r.get("log_p_prefix"),
                "log_p_joint_mdd": r.get("log_p_joint_mdd"),
                "log_p_joint_ctrl": r.get("log_p_joint_ctrl"),
                "pred_label": pred,
                "correct": _coerce_correct(r, pred, meta["true_class"]),
            })
    return rows


def load_all(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "joint_*.jsonl")))
    print(f"Found {len(paths)} joint_*.jsonl files in {input_dir}")
    if not paths:
        raise SystemExit("No joint_*.jsonl files found.")

    rows = []
    for p in paths:
        meta = parse_filename(p)
        if meta is None:
            print(f"  unparseable: {os.path.basename(p)}")
            continue
        recs = load_jsonl_records(p, meta)
        rows.extend(recs)
        print(f"{len(recs):4d}  <-  {os.path.basename(p)}")

    df = pd.DataFrame(rows)
    df["p_mdd_norm"] = pd.to_numeric(df["p_mdd_norm"], errors="coerce")
    return df.dropna(subset=["p_mdd_norm", "filename"])


def pair_across_conditions(df_model):
    pivot_p = (
        df_model.pivot_table(
            index=["filename", "true_class", "split"],
            columns="condition",
            values="p_mdd_norm",
            aggfunc="first",
        )
        .rename(columns={c: f"p_mdd_{c}" for c in CONDITIONS})
        .reset_index()
    )
    pivot_corr = (
        df_model.pivot_table(
            index=["filename", "true_class", "split"],
            columns="condition",
            values="correct",
            aggfunc="first",
        )
        .rename(columns={c: f"correct_{c}" for c in CONDITIONS})
        .reset_index()
    )
    merged = pivot_p.merge(pivot_corr, on=["filename", "true_class", "split"], how="outer")
    paired = merged.dropna(subset=COND_KEYS).copy()
    paired["delta_c2_c1"] = paired["p_mdd_c2"] - paired["p_mdd_c1"]
    paired["delta_c4_c1"] = paired["p_mdd_c4"] - paired["p_mdd_c1"]
    paired["delta_c4_c2"] = paired["p_mdd_c4"] - paired["p_mdd_c2"]
    return paired, merged


def compute_ece(probs, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - probs[mask].mean())
    return float(ece)


def ece_brier_at_condition(df_model, cond):
    # MDD = positive class for F1/precision/recall (matches the paper).
    sub = df_model[(df_model["condition"] == cond) & df_model["correct"].notna()].copy()
    if sub.empty:
        return {"n": 0}
    p_mdd = sub["p_mdd_norm"].to_numpy()
    p_hat = np.where(p_mdd >= 0.5, p_mdd, 1.0 - p_mdd)
    correct = sub["correct"].to_numpy().astype(float)

    true_mdd = (sub["true_class"] == "mdd").to_numpy()
    pred_mdd = (sub["pred_label"] == "mdd").to_numpy()
    tp = int((true_mdd & pred_mdd).sum())
    fp = int((~true_mdd & pred_mdd).sum())
    fn = int((true_mdd & ~pred_mdd).sum())
    tn = int((~true_mdd & ~pred_mdd).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "n": int(len(sub)),
        "acc": float(correct.mean()),
        "f1_mdd": float(f1),
        "precision_mdd": float(prec),
        "recall_mdd": float(rec),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "ece_15": compute_ece(p_hat, correct, 15),
        "ece_10": compute_ece(p_hat, correct, 10),
        "brier": float(np.mean((p_hat - correct) ** 2)),
    }


def summarize_paired(paired, cohort):
    n = len(paired)
    # breakpoint()  # used this while chasing the wilcoxon NaN issue
    nan = float("nan")
    out = {"cohort": cohort, "n": n}
    for col in ("p_mdd_c1", "p_mdd_c2", "p_mdd_c4",
                "delta_c2_c1", "delta_c4_c1", "delta_c4_c2"):
        out["mean_" + col] = float(paired[col].mean()) if n else nan
        out["std_" + col] = float(paired[col].std()) if n else nan
    for c in CONDITIONS:
        ccol = f"correct_{c}"
        if ccol in paired and paired[ccol].notna().any():
            out[f"acc_{c}_paired"] = float(paired[ccol].mean())
    return out


def print_paired_summary(s):
    print()
    print(f"paired summary ({s['cohort']}, n={s['n']})")
    for cond in ("c1", "c2", "c4"):
        m = s[f"mean_p_mdd_{cond}"]
        sd = s[f"std_p_mdd_{cond}"]
        print(f"  P(MDD) {cond.upper()}: {m:.3f} +/- {sd:.3f}")
    for a, b in (("c2", "c1"), ("c4", "c1"), ("c4", "c2")):
        m = s[f"mean_delta_{a}_{b}"]
        sd = s[f"std_delta_{a}_{b}"]
        print(f"  delta {a.upper()}-{b.upper()}: {m:+.3f} +/- {sd:.3f}")
    accs = {c: s.get(f"acc_{c}_paired") for c in CONDITIONS}
    if any(v is not None for v in accs.values()):
        bits = " ".join(f"{c.upper()}={v:.3f}" for c, v in accs.items() if v is not None)
        print(f"  acc (paired): {bits}")


def plot_three_way(paired, output_dir, model, cohort_tag, show_decision_boundary=False):
    n = len(paired)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [0, 1, 2]
    means = [paired[k].mean() for k in COND_KEYS]
    stds = [paired[k].std() for k in COND_KEYS]
    d_c2_c1 = means[1] - means[0]
    d_c4_c1 = means[2] - means[0]
    band_lo = [max(0.0, m - s) for m, s in zip(means, stds)]
    band_hi = [min(1.0, m + s) for m, s in zip(means, stds)]

    ax.fill_between(xs, band_lo, band_hi, color="black", alpha=0.12, label=r"$\pm 1$ SD")
    ax.plot(
        xs, means, color="black", linewidth=2.5, marker="o", markersize=7, zorder=6,
        label=(
            f"{cohort_tag} cohort, $n={n}$\n"
            + rf"$\delta_{{C_2-C_1}}={d_c2_c1:+.3f}$,"
            + rf" $\delta_{{C_4-C_1}}={d_c4_c1:+.3f}$"
        ),
    )
    if show_decision_boundary:
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.4,
                   label="Decision boundary (0.5)")

    ax.set_xticks(xs)
    ax.set_xticklabels(COND_LABELS, fontsize=15)
    ax.set_ylabel(r"$\hat{P}(\mathrm{MDD})$", fontsize=15)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, borderaxespad=0, frameon=True,
              columnspacing=1.2, handlelength=1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(
        os.path.join(output_dir, f"fig_joint_three_way_{model}_{cohort_tag}.pdf"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close()


def plot_three_way_overlay(paired_mdd, paired_ctl, output_dir, model):
    if len(paired_mdd) == 0 and len(paired_ctl) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [0, 1, 2]
    series = [
        ("MDD", paired_mdd, "#CC3311", "-", "o"),
        ("Control", paired_ctl, "#0077BB", "--", "s"),
    ]
    for name, df_c, color, ls, marker in series:
        if len(df_c) == 0:
            continue
        means = [df_c[k].mean() for k in COND_KEYS]
        stds = [df_c[k].std() for k in COND_KEYS]
        d_c2_c1 = means[1] - means[0]
        d_c4_c1 = means[2] - means[0]
        lower = [max(0.0, m - s) for m, s in zip(means, stds)]
        upper = [min(1.0, m + s) for m, s in zip(means, stds)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.10)
        ax.plot(
            xs, means, color=color, linewidth=2.5, linestyle=ls,
            marker=marker, markersize=7, zorder=6,
            label=(rf"{name} ($n={len(df_c)}$): "
                   rf"$\delta_{{C_2-C_1}}={d_c2_c1:+.3f}$, "
                   rf"$\delta_{{C_4-C_1}}={d_c4_c1:+.3f}$"),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(COND_LABELS, fontsize=15)
    ax.set_ylabel(r"$\hat{P}(\mathrm{MDD})$", fontsize=15)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=1, borderaxespad=0, frameon=True, handlelength=2.4)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.savefig(
        os.path.join(output_dir, f"fig_joint_overlay_{model}.pdf"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close()


def plot_three_way_dual(paired_mdd, paired_ctl, output_dir, model):
    if len(paired_mdd) == 0 and len(paired_ctl) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 9.0), sharex=True,
                             gridspec_kw={"hspace": 0.10})
    xs = [0, 1, 2]
    cohorts = [
        ("MDD", paired_mdd, "#CC3311", "-", "o"),
        ("Control", paired_ctl, "#0077BB", "--", "s"),
    ]
    panels = [
        (axes[0], r"$\hat{P}(\mathrm{MDD})$", False),
        (axes[1], r"$\hat{P}(\mathrm{Control})$", True),
    ]
    for panel_idx, (ax, ylabel, flip) in enumerate(panels):
        for cname, df_c, color, ls, marker in cohorts:
            if len(df_c) == 0:
                continue
            means_raw = [df_c[k].mean() for k in COND_KEYS]
            stds = [df_c[k].std() for k in COND_KEYS]
            means = [1 - m for m in means_raw] if flip else means_raw
            band_lo = [max(0.0, m - s) for m, s in zip(means, stds)]
            band_hi = [min(1.0, m + s) for m, s in zip(means, stds)]

            ax.fill_between(xs, band_lo, band_hi, color=color, alpha=0.10)
            d21 = means[1] - means[0]
            d41 = means[2] - means[0]
            label = (rf"{cname} ($n={len(df_c)}$): "
                     rf"$\delta_{{C_2-C_1}}={d21:+.3f}$, "
                     rf"$\delta_{{C_4-C_1}}={d41:+.3f}$")
            ax.plot(xs, means, color=color, linewidth=2.5, linestyle=ls,
                    marker=marker, markersize=7, zorder=6,
                    label=label if panel_idx == 0 else None)

        ax.set_xticks(xs)
        ax.set_xticklabels(COND_LABELS, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(-0.02, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.00),
                   ncol=1, frameon=True, fontsize=18, handlelength=2.4)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(
        os.path.join(output_dir, f"fig_joint_dual_{model}.pdf"),
        dpi=300, bbox_inches="tight", pad_inches=0.05,
    )
    plt.close()


def reliability_bins(probs, correct, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, counts, confs, accs = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        centres.append(0.5 * (lo + hi))
        if mask.sum() == 0:
            counts.append(0)
            confs.append(np.nan)
            accs.append(np.nan)
        else:
            counts.append(int(mask.sum()))
            confs.append(float(probs[mask].mean()))
            accs.append(float(correct[mask].mean()))
    return (np.asarray(centres), np.asarray(counts),
            np.asarray(confs), np.asarray(accs))


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate joint-inference results into tables and figures.")
    p.add_argument("--input_dir", default="./results_joint_v3")
    p.add_argument("--output_dir", default="./summary_joint")
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional model basename filter (default: all discovered).")
    p.add_argument("--reliability_bins", type=int, nargs="+", default=[10, 15],
                   help="Bin counts for reliability diagrams; one PDF per count.")
    p.add_argument("--log_to_file", action="store_true",
                   help="Tee stdout to summary_joint.log inside --output_dir.")
    args = p.parse_args()
    return args


def main():
    args = parse_args()

    log_path = None
    if args.log_to_file:
        log_path = os.path.join(args.output_dir, "summary_joint.log")
        sys.stdout = open(log_path, "w")
        sys.stderr = sys.stdout

    df = load_all(args.input_dir)
    print(f"\nTotal records: {len(df)}")
    print(f"Models:        {sorted(df['model'].unique())}")

    target_models = args.models or sorted(df["model"].unique())
    all_summary_rows = []
    n_skipped = 0  # TODO: count skips when we add per-model filtering

    for model in target_models:
        df_m = df[df["model"] == model].copy()
        if df_m.empty:
            print(f"\n[skip] no records for model={model}")
            continue

        print(f"\n{model}: {len(df_m)} records")
        for cond in CONDITIONS:
            n_mdd = len(df_m[(df_m["condition"] == cond) & (df_m["true_class"] == "mdd")])
            n_ctl = len(df_m[(df_m["condition"] == cond) & (df_m["true_class"] == "control")])
            print(f"  {cond.upper()}: mdd={n_mdd}  control={n_ctl}")

        paired, _ = pair_across_conditions(df_m)
        paired_mdd = paired[paired["true_class"] == "mdd"].copy()
        paired_ctl = paired[paired["true_class"] == "control"].copy()

        sum_mdd = summarize_paired(paired_mdd, "MDD")
        sum_ctl = summarize_paired(paired_ctl, "Control")
        sum_all = summarize_paired(paired, "All")
        print_paired_summary(sum_mdd)
        print_paired_summary(sum_ctl)
        print_paired_summary(sum_all)

        ece_table = {c: ece_brier_at_condition(df_m, c) for c in CONDITIONS}

        paired.to_csv(os.path.join(args.output_dir, f"paired_{model}.csv"), index=False)
        pd.DataFrame([
            {**sum_mdd, "model": model},
            {**sum_ctl, "model": model},
            {**sum_all, "model": model},
        ]).to_csv(
            os.path.join(args.output_dir, f"paired_summary_{model}.csv"),
            index=False, float_format="%.4f",
        )
        pd.DataFrame([{"model": model, "condition": c, **r}
                      for c, r in ece_table.items()]).to_csv(
            os.path.join(args.output_dir, f"ece_table_{model}.csv"),
            index=False, float_format="%.4f",
        )

        plot_three_way(paired_mdd, args.output_dir, model, "mdd")
        plot_three_way(paired_ctl, args.output_dir, model, "control")
        plot_three_way_overlay(paired_mdd, paired_ctl, args.output_dir, model)
        plot_three_way_dual(paired_mdd, paired_ctl, args.output_dir, model)

        all_summary_rows.extend([
            {**sum_mdd, "model": model},
            {**sum_ctl, "model": model},
            {**sum_all, "model": model},
        ])

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(
            os.path.join(args.output_dir, "paired_summary_all.csv"),
            index=False, float_format="%.4f",
        )

    if log_path:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    # cheap: make sure output_dir exists before main does anything with paths
    _a = parse_args()
    os.makedirs(_a.output_dir, exist_ok=True)
    main()
