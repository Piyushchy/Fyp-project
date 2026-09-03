# ============================================================
# TEXT EMOTION MODULE - REPORT
# ============================================================
#
# Turns results/benchmark.json into the figures used by the
# README. Run after benchmark.py:
#
#   python report.py
#
# Produces, in assets/:
#   model_comparison.png  quality / size / speed, one axis each
#   per_class_f1.png      per-emotion F1 for all three models
#   confusion_matrix.png  best model, row-normalised
#   usage.png             a rendering of the predict.py session
# ============================================================

import argparse
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from config import (
    ASSETS_DIR,
    ID2LABEL,
    MODELS,
    MODULE_DIR,
    RESULTS_DIR,
    RESULTS_FILE,
    ensure_directories,
)

MODEL_EPOCHS = {key: settings["epochs"] for key, settings in MODELS.items()}
MODEL_LR = {key: settings["learning_rate"] for key, settings in MODELS.items()}


# ============================================================
# PALETTE
# ============================================================
#
# Colour is keyed to the MODEL, not to the metric, so the same
# model keeps the same hue in every panel. Three categorical
# slots, validated for colour-vision deficiency across all
# pairs. Every bar is directly labelled, so no reading of the
# chart depends on telling two hues apart.
# ============================================================

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

MODEL_COLORS = {
    "tinybert": "#2a78d6",
    "distilbert": "#eb6834",
    "mobilebert": "#1baf7a",
}

# Single-hue sequential ramp for the heatmaps, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues",
    ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"],
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


# ============================================================
# SHARED AXIS STYLING
# ============================================================

def style_axis(axis, x_label):
    """Recessive chrome: hairline grid, no box, muted ticks."""

    axis.set_xlabel(x_label, fontsize=9, color=INK_SECONDARY, labelpad=8)

    axis.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    axis.yaxis.grid(False)
    axis.set_axisbelow(True)

    for side in ("top", "right", "bottom"):
        axis.spines[side].set_visible(False)

    axis.spines["left"].set_color(BASELINE)
    axis.spines["left"].set_linewidth(1.0)

    axis.tick_params(length=0, labelsize=9)


def labelled_bars(axis, results, values, x_label, labels, headroom_factor=1.28):
    """One horizontal bar per model, each labelled with its value."""

    names = [result["display_name"].split(" (")[0] for result in results]
    colors = [MODEL_COLORS[result["model_key"]] for result in results]

    positions = np.arange(len(results))[::-1]

    axis.barh(positions, values, height=0.42, color=colors, zorder=2)

    axis.set_yticks(positions)
    axis.set_yticklabels(names, fontsize=9.5, color=INK_PRIMARY)

    headroom = max(values) * headroom_factor

    axis.set_xlim(0, headroom)

    for position, value, label in zip(positions, values, labels):
        axis.text(
            value + headroom * 0.02,
            position,
            label,
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )

    style_axis(axis, x_label)


# ============================================================
# FIGURE 1 - QUALITY, SIZE, SPEED
# ============================================================

def figure_comparison(payload, path):

    results = payload["results"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 6.2))

    figure.suptitle(
        "Compact transformers on MTEB EmotionClassification",
        fontsize=13,
        color=INK_PRIMARY,
        x=0.055,
        ha="left",
        y=0.975,
    )

    test_size = int(np.array(results[0]["confusion_matrix"]).sum())

    figure.text(
        0.055, 0.925,
        f"Test split, {test_size} sentences  -  "
        f"CPU inference, {payload['torch_threads']} threads  -  "
        f"size ratios are against BERT-base "
        f"({payload['bert_base_parameters']/1e6:.0f}M), which is not trained here",
        fontsize=9,
        color=INK_SECONDARY,
        ha="left",
    )

    labelled_bars(
        axes[0][0], results,
        [r["metrics"]["accuracy"] * 100 for r in results],
        "Accuracy (%)  -  higher is better",
        [f"{r['metrics']['accuracy']*100:.1f}%" for r in results],
    )

    labelled_bars(
        axes[0][1], results,
        [r["metrics"]["macro_f1"] * 100 for r in results],
        "Macro-F1 (%)  -  higher is better",
        [f"{r['metrics']['macro_f1']*100:.1f}%" for r in results],
    )

    labelled_bars(
        axes[1][0], results,
        [r["size"]["parameters"] / 1e6 for r in results],
        "Parameters (millions)  -  lower is better",
        [f"{r['size']['parameters']/1e6:.1f}M "
         f"({r['size']['smaller_than_bert_base']:.1f}x under BERT-base)"
         for r in results],
        headroom_factor=1.9,
    )

    labelled_bars(
        axes[1][1], results,
        [r["latency"]["median_ms"] for r in results],
        "Median CPU latency, one sentence (ms)  -  lower is better",
        [f"{r['latency']['median_ms']:.1f} ms" for r in results],
    )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    figure.savefig(path, dpi=160)

    plt.close(figure)

    print(f"wrote {path}")


# ============================================================
# FIGURE 2 - PER-CLASS F1
# ============================================================

def figure_per_class(payload, path):

    results = payload["results"]

    emotions = list(ID2LABEL.values())

    matrix = np.array([
        [result["metrics"]["per_class"][name]["f1"] * 100 for name in emotions]
        for result in results
    ])

    figure, axis = plt.subplots(figsize=(9, 3.4))

    image = axis.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")

    axis.set_xticks(range(len(emotions)))
    axis.set_xticklabels(
        [f"{name}\n(n={results[0]['metrics']['per_class'][name]['support']})"
         for name in emotions],
        fontsize=9,
        color=INK_SECONDARY,
    )

    axis.set_yticks(range(len(results)))
    axis.set_yticklabels(
        [result["display_name"].split(" (")[0] for result in results],
        fontsize=9.5,
        color=INK_PRIMARY,
    )

    # Cell values carry the magnitude; the ramp is a secondary cue.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row][column]
            axis.text(
                column, row, f"{value:.1f}",
                ha="center", va="center",
                fontsize=9.5,
                color="#ffffff" if value > 50 else INK_PRIMARY,
            )

    axis.set_title(
        "Per-emotion F1 (%)  -  test split",
        fontsize=12,
        color=INK_PRIMARY,
        loc="left",
        pad=12,
    )

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.tick_params(length=0)

    # 2px surface gap between cells.
    axis.set_xticks(np.arange(-0.5, len(emotions), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(results), 1), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="minor", length=0)

    bar = figure.colorbar(image, ax=axis, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, colors=INK_MUTED)

    figure.tight_layout()

    figure.savefig(path, dpi=160)

    plt.close(figure)

    print(f"wrote {path}")


# ============================================================
# FIGURE 3 - CONFUSION MATRIX
# ============================================================

def figure_confusion(payload, path):

    best = max(payload["results"], key=lambda r: r["metrics"]["macro_f1"])

    counts = np.array(best["confusion_matrix"], dtype=float)

    # Row-normalise: with supports from 66 to 695, raw counts
    # would just redraw the class imbalance.
    normalised = counts / counts.sum(axis=1, keepdims=True) * 100

    emotions = list(ID2LABEL.values())

    figure, axis = plt.subplots(figsize=(6.6, 5.6))

    axis.imshow(normalised, cmap=SEQUENTIAL, vmin=0, vmax=100)

    axis.set_xticks(range(len(emotions)))
    axis.set_xticklabels(emotions, fontsize=9, color=INK_SECONDARY, rotation=30, ha="right")

    axis.set_yticks(range(len(emotions)))
    axis.set_yticklabels(emotions, fontsize=9, color=INK_SECONDARY)

    axis.set_xlabel("predicted", fontsize=9.5, color=INK_SECONDARY, labelpad=8)
    axis.set_ylabel("actual", fontsize=9.5, color=INK_SECONDARY, labelpad=8)

    for row in range(len(emotions)):
        for column in range(len(emotions)):
            percentage = normalised[row][column]
            axis.text(
                column, row,
                f"{percentage:.0f}%\n{int(counts[row][column])}",
                ha="center", va="center",
                fontsize=8,
                color="#ffffff" if percentage > 50 else INK_PRIMARY,
            )

    axis.set_title(
        f"{best['display_name']}  -  confusion matrix (row %)",
        fontsize=12,
        color=INK_PRIMARY,
        loc="left",
        pad=12,
    )

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.tick_params(length=0)

    axis.set_xticks(np.arange(-0.5, len(emotions), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(emotions), 1), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="minor", length=0)

    figure.tight_layout()

    figure.savefig(path, dpi=160)

    plt.close(figure)

    print(f"wrote {path}")


# ============================================================
# FIGURE 4 - USAGE
# ============================================================
#
# Runs predict.py for real and renders whatever it printed, so
# the image in the README cannot drift from the actual output.
# ============================================================

USAGE_COMMANDS = [
    ["predict.py", "i cant believe you remembered my birthday"],
    ["predict.py", "--distribution", "i am terrified of what happens tomorrow"],
]

TERMINAL_BACKGROUND = "#12120f"
TERMINAL_TEXT = "#e8e6df"
TERMINAL_PROMPT = "#1baf7a"
TERMINAL_RESULT = "#7fb2f0"


def capture(command):
    """Run one predict.py invocation and return prompt + output lines."""

    completed = subprocess.run(
        [sys.executable, *command],
        cwd=MODULE_DIR,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"predict.py failed: {completed.stderr.strip()[-500:]}"
        )

    quoted = [
        f'"{part}"' if " " in part else part
        for part in command
    ]

    return f"$ python {' '.join(quoted)}", completed.stdout.rstrip("\n").split("\n")


def figure_usage(path):

    lines = []

    for command in USAGE_COMMANDS:

        prompt, output = capture(command)

        lines.append(("prompt", prompt))

        for line in output:
            lines.append(("result" if "->" in line else "output", line))

        lines.append(("output", ""))

    height = 0.235 * len(lines) + 0.55

    figure = plt.figure(figsize=(11, height), facecolor=TERMINAL_BACKGROUND)

    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_facecolor(TERMINAL_BACKGROUND)
    axis.set_axis_off()

    colors = {
        "prompt": TERMINAL_PROMPT,
        "result": TERMINAL_RESULT,
        "output": TERMINAL_TEXT,
    }

    for index, (kind, line) in enumerate(lines):
        axis.text(
            0.018,
            1 - (index + 1.1) * (0.235 / height),
            line,
            fontsize=10.5,
            family="monospace",
            color=colors[kind],
            va="top",
            transform=axis.transAxes,
        )

    figure.savefig(path, dpi=160, facecolor=TERMINAL_BACKGROUND)

    plt.close(figure)

    print(f"wrote {path}")


# ============================================================
# MARKDOWN TABLES
# ============================================================
#
# The README quotes these tables verbatim. Generating them from
# benchmark.json means the documented numbers cannot drift away
# from the numbers the models actually produced.
# ============================================================

def write_tables(payload, path):

    results = payload["results"]

    emotions = list(ID2LABEL.values())

    lines = ["## Comparison", ""]

    lines.append("| Model | Params | Disk | Accuracy | Macro-F1 | "
                 "Macro-P | Macro-R | Latency | vs BERT-base |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['display_name']} "
            f"| {result['size']['parameters']/1e6:.1f}M "
            f"| {result['size']['disk_mb']:.0f} MB "
            f"| {metrics['accuracy']*100:.2f}% "
            f"| {metrics['macro_f1']*100:.2f}% "
            f"| {metrics['macro_precision']*100:.2f}% "
            f"| {metrics['macro_recall']*100:.2f}% "
            f"| {result['latency']['median_ms']:.1f} ms "
            f"| {result['size']['smaller_than_bert_base']:.1f}x smaller |"
        )

    lines += ["", "## Per-emotion F1 (%)", ""]

    lines.append("| Model | " + " | ".join(emotions) + " |")
    lines.append("|---" * (len(emotions) + 1) + "|")

    for result in results:
        scores = result["metrics"]["per_class"]
        lines.append(
            f"| {result['display_name'].split(' (')[0]} | "
            + " | ".join(f"{scores[name]['f1']*100:.1f}" for name in emotions)
            + " |"
        )

    lines.append(
        "| _test rows_ | "
        + " | ".join(
            str(results[0]["metrics"]["per_class"][name]["support"])
            for name in emotions
        )
        + " |"
    )

    lines += ["", "## Training cost", ""]

    lines.append("| Model | Epochs | LR | Train time (CPU) | Throughput |")
    lines.append("|---|---|---|---|---|")

    for result in results:
        seconds = result["training_seconds"] or 0
        lines.append(
            f"| {result['display_name'].split(' (')[0]} "
            f"| {MODEL_EPOCHS[result['model_key']]} "
            f"| {MODEL_LR[result['model_key']]:g} "
            f"| {seconds/60:.0f} min "
            f"| {result['throughput_sentences_per_second']:.0f} sentences/s |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {path}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--usage-only",
        action="store_true",
        help="Regenerate only the usage image (no benchmark needed).",
    )

    arguments = parser.parse_args()

    ensure_directories()

    if not arguments.usage_only:

        if not RESULTS_FILE.exists():
            sys.exit(f"{RESULTS_FILE} not found. Run: python benchmark.py")

        with open(RESULTS_FILE) as handle:
            payload = json.load(handle)

        figure_comparison(payload, ASSETS_DIR / "model_comparison.png")
        figure_per_class(payload, ASSETS_DIR / "per_class_f1.png")
        figure_confusion(payload, ASSETS_DIR / "confusion_matrix.png")

        write_tables(payload, RESULTS_DIR / "tables.md")

    figure_usage(ASSETS_DIR / "usage.png")


if __name__ == "__main__":
    main()
