import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEYWORDS = {
    "def",
    "return",
    "if",
    "for",
    "while",
    "class",
    "import",
    "from",
    "else",
    "elif",
    "try",
    "except",
    "with",
}


def sanitize_float(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def clean_token(token):
    return token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def safe_ylim_from_series(series_list, min_span=1e-4, pad_ratio=0.12):
    values = [v for series in series_list for v in series]
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if high - low < min_span:
        center = 0.5 * (high + low)
        half = 0.5 * min_span
        return max(0.0, center - half), center + half
    pad = (high - low) * pad_ratio
    return max(0.0, low - pad), high + pad


def classify_token(token):
    stripped = token.strip()
    if "\n" in token or token == "\n":
        return "newline"
    if "```" in token:
        return "fence"
    if stripped in KEYWORDS:
        return "keyword"
    if stripped in {".", ",", ":", ";", "(", ")", "[", "]", "{", "}"}:
        return "punct"
    if token in {"    ", "  ", "\t"} or token.startswith("    "):
        return "indent"
    return "other"


def gather_sample_paths(base_dir):
    global_paths = sorted(base_dir.glob("sample_*/sample_*_attn.json"))
    layer_paths = sorted(base_dir.glob("sample_*/sample_*_layer_combined_scores.json"))
    layer_map = {
        path.parent.name: path
        for path in layer_paths
    }
    samples = []
    for global_path in global_paths:
        sample_name = global_path.parent.name
        samples.append(
            {
                "name": sample_name,
                "global_json": global_path,
                "layer_json": layer_map.get(sample_name),
            }
        )
    return samples


def parse_global_sample(global_data):
    trajectory = global_data.get("trajectory", [])
    return {
        "tokens": [step.get("token", "") for step in trajectory],
        "mean_scores": [sanitize_float(step.get("mean_score", 0.0)) for step in trajectory],
        "combined_scores": [sanitize_float(step.get("combined_score", step.get("focus_score", 0.0))) for step in trajectory],
        "code_scores": [sanitize_float(step.get("code_combined_score", 0.0)) for step in trajectory],
        "cut_positions": list(global_data.get("absolute_cut_positions", [])),
        "block_sizes": list(global_data.get("block_sizes", [])),
    }


def parse_layer_sample(layer_data):
    layer_matrix = []
    layer_code_matrix = []
    layer_mean_matrix = []
    for layer in layer_data.get("layers", []):
        layer_mean_matrix.append([sanitize_float(step.get("mean_score", 0.0)) for step in layer.get("trajectory", [])])
        layer_matrix.append([sanitize_float(step.get("combined_score", 0.0)) for step in layer.get("trajectory", [])])
        layer_code_matrix.append([sanitize_float(step.get("code_combined_score", 0.0)) for step in layer.get("trajectory", [])])
    return {
        "combined": np.array(layer_matrix, dtype=np.float32) if layer_matrix else np.zeros((0, 0), dtype=np.float32),
        "code": np.array(layer_code_matrix, dtype=np.float32) if layer_code_matrix else np.zeros((0, 0), dtype=np.float32),
        "mean": np.array(layer_mean_matrix, dtype=np.float32) if layer_mean_matrix else np.zeros((0, 0), dtype=np.float32),
    }


def plot_global_scores(sample_name, sample, save_path):
    tokens = sample["tokens"]
    xs = np.arange(len(tokens))
    mean_scores = sample["mean_scores"]
    combined_scores = sample["combined_scores"]
    code_scores = sample["code_scores"]

    fig_width = max(14, len(tokens) * 0.08)
    plt.figure(figsize=(fig_width, 5.5))
    plt.plot(xs, mean_scores, color="#7f7f7f", linewidth=1.2, marker=".", markersize=2.5, label="Mean Score")
    plt.plot(xs, combined_scores, color="#d62728", linewidth=1.4, marker=".", markersize=3.0, label="Combined Score")
    plt.plot(xs, code_scores, color="#2ca02c", linewidth=1.4, marker=".", markersize=3.0, label="Code-only Combined")

    for cut in sample["cut_positions"]:
        plt.axvline(cut - 0.5, color="#17becf", linestyle="--", linewidth=1.0, alpha=0.75)

    if tokens and len(tokens) <= 256:
        plt.xticks(xs, [clean_token(tok) for tok in tokens], rotation=90, fontsize=8, fontfamily="monospace")
    else:
        tick_step = max(1, len(tokens) // 32)
        ticks = list(range(0, len(tokens), tick_step))
        plt.xticks(ticks, ticks)
        plt.xlabel("Generated Token Position")

    plt.ylabel("Score")
    plt.title(f"{sample_name}: Global Score Trajectory")
    plt.ylim(*safe_ylim_from_series([mean_scores, combined_scores, code_scores]))
    plt.xlim(-0.5, max(len(tokens) - 0.5, 0.5))
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_layer_heatmap(sample_name, layer_bundle, cut_positions, save_path, use_code=False):
    matrix = layer_bundle["code" if use_code else "combined"]
    if matrix.size == 0:
        return

    plt.figure(figsize=(max(14, matrix.shape[1] * 0.08), max(6, matrix.shape[0] * 0.28)))
    vmax = np.percentile(matrix, 99) if matrix.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="magma", interpolation="nearest", vmin=0.0, vmax=vmax)
    for cut in cut_positions:
        plt.axvline(cut - 0.5, color="#7fffd4", linestyle="--", linewidth=0.9, alpha=0.7)
    plt.colorbar(label="Score")
    plt.xlabel("Generated Token Position")
    plt.ylabel("Layer")
    suffix = "Code-only Combined" if use_code else "Combined"
    plt.title(f"{sample_name}: Layer Heatmap ({suffix})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_cut_alignment(sample_name, combined_scores, cut_positions, save_path, window=8):
    aligned = []
    for cut in cut_positions:
        if cut - window >= 0 and cut + window < len(combined_scores):
            aligned.append(combined_scores[cut - window:cut + window + 1])
    if not aligned:
        return

    aligned = np.array(aligned, dtype=np.float32)
    mean_curve = aligned.mean(axis=0)
    std_curve = aligned.std(axis=0)
    xs = np.arange(-window, window + 1)

    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, mean_curve, color="#d62728", linewidth=1.8, label="Mean aligned combined score")
    plt.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, color="#d62728", alpha=0.18, label="±1 std")
    plt.axvline(0, color="black", linestyle="--", linewidth=1.1, label="Cut point")
    plt.xlabel("Relative Position to Cut")
    plt.ylabel("Combined Score")
    plt.title(f"{sample_name}: Cut-aligned Score Curve")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_cut_windows(sample_name, tokens, combined_scores, cut_positions, save_path, window=8, max_cuts=8):
    valid_cuts = [cut for cut in cut_positions if cut - window >= 0 and cut + window < len(combined_scores)]
    valid_cuts = valid_cuts[:max_cuts]
    if not valid_cuts:
        return

    rows = len(valid_cuts)
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(3, rows * 2.2)), sharex=False)
    if rows == 1:
        axes = [axes]

    for ax, cut in zip(axes, valid_cuts):
        local_scores = combined_scores[cut - window:cut + window + 1]
        local_tokens = tokens[cut - window:cut + window + 1]
        xs = np.arange(-window, window + 1)
        ax.plot(xs, local_scores, color="#d62728", marker="o", markersize=2.5, linewidth=1.2)
        ax.axvline(0, color="#17becf", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"cut={cut}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([clean_token(tok) for tok in local_tokens], rotation=90, fontsize=7, fontfamily="monospace")

    axes[-1].set_xlabel("Relative Token Position")
    plt.suptitle(f"{sample_name}: Local Cut Windows", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def compute_block_metrics(sample):
    combined_scores = sample["combined_scores"]
    block_sizes = sample["block_sizes"]
    metrics = []
    start = 0
    for block_index, size in enumerate(block_sizes):
        end = min(start + size, len(combined_scores))
        block_scores = combined_scores[start:end]
        if not block_scores:
            break
        cut_score = combined_scores[end] if end < len(combined_scores) else block_scores[-1]
        metrics.append(
            {
                "block_index": block_index,
                "block_size": size,
                "mean_score": float(np.mean(block_scores)),
                "max_score": float(np.max(block_scores)),
                "cut_score": cut_score,
                "start": start,
                "end": end,
            }
        )
        start = end
    return metrics


def plot_block_metrics(sample_name, block_metrics, save_path):
    if not block_metrics:
        return

    xs = [item["block_index"] for item in block_metrics]
    block_sizes = [item["block_size"] for item in block_metrics]
    mean_scores = [item["mean_score"] for item in block_metrics]
    max_scores = [item["max_score"] for item in block_metrics]
    cut_scores = [item["cut_score"] for item in block_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xs, block_sizes, color="#1f77b4", marker="o", linewidth=1.4)
    axes[0].set_ylabel("Block Size")
    axes[0].set_title(f"{sample_name}: Block Size and Score Summary")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)

    axes[1].plot(xs, mean_scores, color="#7f7f7f", marker="o", linewidth=1.2, label="Block Mean")
    axes[1].plot(xs, max_scores, color="#d62728", marker="o", linewidth=1.2, label="Block Max")
    axes[1].plot(xs, cut_scores, color="#2ca02c", marker="o", linewidth=1.2, label="Cut Score")
    axes[1].set_xlabel("Block Index")
    axes[1].set_ylabel("Combined Score")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_block_scatter(dataset_block_metrics, save_path):
    if not dataset_block_metrics:
        return

    block_sizes = [item["block_size"] for item in dataset_block_metrics]
    max_scores = [item["max_score"] for item in dataset_block_metrics]
    cut_scores = [item["cut_score"] for item in dataset_block_metrics]

    plt.figure(figsize=(8, 5.5))
    plt.scatter(block_sizes, max_scores, color="#d62728", alpha=0.75, label="Block Max")
    plt.scatter(block_sizes, cut_scores, color="#2ca02c", alpha=0.75, label="Cut Score")
    plt.xlabel("Block Size")
    plt.ylabel("Combined Score")
    plt.title("Dataset: Block Size vs Score")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_structural_boxplot(structural_scores, save_path):
    labels = ["newline", "punct", "keyword", "fence", "indent", "other"]
    data = [structural_scores[label] for label in labels if structural_scores[label]]
    used_labels = [label for label in labels if structural_scores[label]]
    if not data:
        return

    plt.figure(figsize=(9, 5.2))
    plt.boxplot(data, labels=used_labels, showfliers=False)
    plt.ylabel("Combined Score")
    plt.title("Dataset: Structural Token Score Distribution")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_structural_lift(structural_scores, save_path):
    labels = ["newline", "punct", "keyword", "fence", "indent", "other"]
    means = []
    used_labels = []
    overall = structural_scores.get("all", [])
    if not overall:
        return
    overall_mean = float(np.mean(overall))
    for label in labels:
        scores = structural_scores[label]
        if not scores:
            continue
        used_labels.append(label)
        means.append(float(np.mean(scores) / (overall_mean + 1e-12)))

    plt.figure(figsize=(9, 5.2))
    plt.bar(used_labels, means, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"][:len(means)])
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    plt.ylabel("Mean Score / Global Mean")
    plt.title("Dataset: Structural Token Lift")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_dataset_cut_alignment(all_aligned, save_path):
    if not all_aligned:
        return
    aligned = np.array(all_aligned, dtype=np.float32)
    window = (aligned.shape[1] - 1) // 2
    xs = np.arange(-window, window + 1)
    mean_curve = aligned.mean(axis=0)
    std_curve = aligned.std(axis=0)

    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, mean_curve, color="#d62728", linewidth=1.8, label="Dataset mean")
    plt.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, color="#d62728", alpha=0.18, label="±1 std")
    plt.axvline(0, color="black", linestyle="--", linewidth=1.1, label="Cut point")
    plt.xlabel("Relative Position to Cut")
    plt.ylabel("Combined Score")
    plt.title("Dataset: Cut-aligned Combined Score")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_dataset_layer_summary(layer_peak_rows, save_path):
    if not layer_peak_rows:
        return

    layer_ids = [row["layer"] for row in layer_peak_rows]
    peak_pm1 = [row["peak_pm1"] for row in layer_peak_rows]
    ratios = [row["ratio"] for row in layer_peak_rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(layer_ids, peak_pm1, color="#1f77b4")
    axes[0].set_ylabel("Peak@cut±1 Rate")
    axes[0].set_title("Dataset: Layer Boundary Alignment Summary")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)

    axes[1].bar(layer_ids, ratios, color="#d62728")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cut / Neighbor Mean")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def compute_layer_summary(layer_bundle, cut_positions):
    matrix = layer_bundle["combined"]
    if matrix.size == 0:
        return []

    rows = []
    for layer_idx in range(matrix.shape[0]):
        scores = matrix[layer_idx].tolist()
        peak_at = 0
        peak_pm1 = 0
        ratios = []
        local_ranks = []
        valid = 0
        for cut in cut_positions:
            if not (2 <= cut < len(scores) - 2):
                continue
            valid += 1
            window = scores[cut - 2:cut + 3]
            local_max = max(window)
            argmax = window.index(local_max) - 2
            if argmax == 0:
                peak_at += 1
            if abs(argmax) <= 1:
                peak_pm1 += 1
            neighbors = scores[max(0, cut - 4):cut] + scores[cut + 1:min(len(scores), cut + 5)]
            if neighbors:
                ratios.append(scores[cut] / (sum(neighbors) / len(neighbors) + 1e-12))
            local_ranks.append(sum(1 for val in window if val <= scores[cut]) / len(window))
        if valid == 0:
            continue
        rows.append(
            {
                "layer": layer_idx,
                "cuts": valid,
                "peak_at_cut": peak_at / valid,
                "peak_pm1": peak_pm1 / valid,
                "ratio": float(np.mean(ratios)) if ratios else 0.0,
                "local_rank": float(np.mean(local_ranks)) if local_ranks else 0.0,
            }
        )
    return rows


def aggregate_layer_summaries(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["layer"]].append(row)
    merged = []
    for layer, rows in sorted(grouped.items()):
        merged.append(
            {
                "layer": layer,
                "peak_at_cut": float(np.mean([row["peak_at_cut"] for row in rows])),
                "peak_pm1": float(np.mean([row["peak_pm1"] for row in rows])),
                "ratio": float(np.mean([row["ratio"] for row in rows])),
                "local_rank": float(np.mean([row["local_rank"] for row in rows])),
            }
        )
    return merged


def write_summary_report(
    save_path,
    sample_summaries,
    aggregated_layers,
    structural_scores,
    dataset_block_metrics,
    dataset_cut_rank,
):
    overall_scores = structural_scores["all"]
    overall_mean = float(np.mean(overall_scores)) if overall_scores else 0.0
    block_hist = Counter(item["block_size"] for item in dataset_block_metrics)

    lines = []
    lines.append("# Score Analysis Report")
    lines.append("")
    lines.append(f"- Number of samples: {len(sample_summaries)}")
    lines.append(f"- Global combined-score mean: {overall_mean:.6f}")
    lines.append(f"- Global cut local-rank mean: {np.mean(dataset_cut_rank):.4f}" if dataset_cut_rank else "- Global cut local-rank mean: N/A")
    lines.append(f"- Block size histogram: {dict(sorted(block_hist.items()))}")
    lines.append("")
    lines.append("## Sample Summary")
    lines.append("")
    for summary in sample_summaries:
        lines.append(
            f"- {summary['sample']}: blocks={summary['block_sizes']} | avg_block={summary['avg_block']:.3f} | "
            f"combined_mean={summary['combined_mean']:.6f} | cut_mean={summary['cut_mean']:.6f}"
        )
    lines.append("")
    lines.append("## Structural Token Lift")
    lines.append("")
    for label in ["newline", "punct", "keyword", "fence", "indent", "other"]:
        scores = structural_scores[label]
        if not scores:
            continue
        lines.append(f"- {label}: mean={np.mean(scores):.6f} | lift={np.mean(scores) / (overall_mean + 1e-12):.3f} | count={len(scores)}")
    lines.append("")
    lines.append("## Best Layers For Boundary Detection")
    lines.append("")
    best_layers = sorted(
        aggregated_layers,
        key=lambda row: (row["peak_pm1"], row["ratio"], row["local_rank"], row["peak_at_cut"]),
        reverse=True,
    )
    for row in best_layers[:12]:
        lines.append(
            f"- layer {row['layer']}: peak@cut={row['peak_at_cut']:.3f} | "
            f"peak@cut±1={row['peak_pm1']:.3f} | cut/neighbor={row['ratio']:.3f} | local_rank={row['local_rank']:.3f}"
        )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Detailed visualization and analysis for dynamic block score JSON outputs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/DELL/Desktop/code/diffusion/DynamicBlockSize/llada/output_analysis"),
        help="Directory containing sample_xxxx JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/DELL/Desktop/code/diffusion/DynamicBlockSize/llada/analysis_viz"),
        help="Directory to save analysis figures and reports.",
    )
    parser.add_argument("--cut-window", type=int, default=8, help="Half-window size for cut alignment plots.")
    args = parser.parse_args()

    samples = gather_sample_paths(args.input_dir)
    if not samples:
        raise FileNotFoundError(f"No sample_*_attn.json files found under {args.input_dir}")

    ensure_dir(args.output_dir)
    summary_dir = args.output_dir / "summary"
    ensure_dir(summary_dir)

    structural_scores = defaultdict(list)
    dataset_block_metrics = []
    dataset_cut_alignment = []
    dataset_cut_rank = []
    layer_summary_rows = []
    sample_summaries = []

    for sample_paths in samples:
        global_data = load_json(sample_paths["global_json"])
        sample = parse_global_sample(global_data)
        sample_name = sample_paths["name"]

        sample_dir = args.output_dir / sample_name
        ensure_dir(sample_dir)

        plot_global_scores(sample_name, sample, sample_dir / "global_scores.png")
        plot_cut_alignment(sample_name, sample["combined_scores"], sample["cut_positions"], sample_dir / "cut_alignment.png", window=args.cut_window)
        plot_cut_windows(sample_name, sample["tokens"], sample["combined_scores"], sample["cut_positions"], sample_dir / "cut_windows.png", window=args.cut_window)

        block_metrics = compute_block_metrics(sample)
        dataset_block_metrics.extend(block_metrics)
        plot_block_metrics(sample_name, block_metrics, sample_dir / "block_metrics.png")

        for idx, token in enumerate(sample["tokens"]):
            score = sample["combined_scores"][idx]
            structural_scores["all"].append(score)
            structural_scores[classify_token(token)].append(score)

        for cut in sample["cut_positions"]:
            if args.cut_window <= cut < len(sample["combined_scores"]) - args.cut_window:
                window = sample["combined_scores"][cut - args.cut_window:cut + args.cut_window + 1]
                dataset_cut_alignment.append(window)
            if 2 <= cut < len(sample["combined_scores"]) - 2:
                local = sample["combined_scores"][cut - 2:cut + 3]
                dataset_cut_rank.append(sum(1 for value in local if value <= sample["combined_scores"][cut]) / len(local))

        if sample_paths["layer_json"] is not None:
            layer_data = load_json(sample_paths["layer_json"])
            layer_bundle = parse_layer_sample(layer_data)
            plot_layer_heatmap(sample_name, layer_bundle, sample["cut_positions"], sample_dir / "layer_heatmap_combined.png")
            plot_layer_heatmap(sample_name, layer_bundle, sample["cut_positions"], sample_dir / "layer_heatmap_code_combined.png", use_code=True)
            layer_summary_rows.extend(compute_layer_summary(layer_bundle, sample["cut_positions"]))

        sample_summaries.append(
            {
                "sample": sample_name,
                "block_sizes": sample["block_sizes"],
                "avg_block": float(np.mean(sample["block_sizes"])) if sample["block_sizes"] else 0.0,
                "combined_mean": float(np.mean(sample["combined_scores"])) if sample["combined_scores"] else 0.0,
                "cut_mean": float(np.mean([sample["combined_scores"][cut] for cut in sample["cut_positions"] if cut < len(sample["combined_scores"])]) or [0.0]),
            }
        )

    aggregated_layers = aggregate_layer_summaries(layer_summary_rows)
    plot_structural_boxplot(structural_scores, summary_dir / "structural_token_boxplot.png")
    plot_structural_lift(structural_scores, summary_dir / "structural_token_lift.png")
    plot_block_scatter(dataset_block_metrics, summary_dir / "block_score_scatter.png")
    plot_dataset_cut_alignment(dataset_cut_alignment, summary_dir / "dataset_cut_alignment.png")
    plot_dataset_layer_summary(aggregated_layers, summary_dir / "layer_boundary_summary.png")
    write_summary_report(
        summary_dir / "report.md",
        sample_summaries,
        aggregated_layers,
        structural_scores,
        dataset_block_metrics,
        dataset_cut_rank,
    )

    print(f"Analysis saved to: {args.output_dir}")
    print(f"Summary report: {summary_dir / 'report.md'}")


if __name__ == "__main__":
    main()
