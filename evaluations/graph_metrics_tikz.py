import os
import re
import argparse
import warnings

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Label normalization and similarity
# ---------------------------------------------------------------------------

def normalize_label(label):
    """Normalize a label: lowercase, collapse whitespace, strip punctuation."""
    if not label:
        return ""
    label = str(label)
    normalized = re.sub(r'\s+', ' ', label.replace('\n', ' ').replace('\r', ' '))
    normalized = normalized.replace('(', '').replace(')', '')
    normalized = normalized.replace('[', '').replace(']', '')
    normalized = normalized.replace('"', '').replace("'", '')
    return normalized.strip().lower()


def calculate_label_similarity(label1, label2):
    """
    Calculate similarity between two labels using sequence matching,
    Jaccard token similarity, and proportional substring containment.
    """
    norm_label1 = normalize_label(label1)
    norm_label2 = normalize_label(label2)
    if not norm_label1 and not norm_label2:
        return 1.0
    if not norm_label1 or not norm_label2:
        return 0.0

    seq_sim = SequenceMatcher(None, norm_label1, norm_label2).ratio()

    tokens1 = set(norm_label1.split())
    tokens2 = set(norm_label2.split())
    if tokens1 or tokens2:
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        jaccard_sim = len(intersection) / len(union) if union else 0.0
    else:
        jaccard_sim = 0.0

    def substring_score(s1, s2):
        if s1 in s2:
            return len(s1) / max(len(s2), 1)
        if s2 in s1:
            return len(s2) / max(len(s1), 1)
        return 0.0

    substring_sim = substring_score(norm_label1, norm_label2)
    return max(seq_sim, jaccard_sim, substring_sim)


# ---------------------------------------------------------------------------
# Node matching (Hungarian algorithm)
# ---------------------------------------------------------------------------

def find_best_node_matches(gt_nodes, pred_nodes, similarity_threshold=0.3, sim_matrix=None):
    """
    Find optimal node matching using the Hungarian algorithm.

    Args:
        gt_nodes: dict mapping node ID -> label for ground truth.
        pred_nodes: dict mapping node ID -> label for predictions.
        similarity_threshold: Minimum similarity to accept a match.
        sim_matrix: Optional precomputed similarity matrix.

    Returns:
        matches: dict mapping gt_node -> (pred_node, similarity)
        unmatched_gt: list of unmatched GT node IDs
        unmatched_pred: list of unmatched predicted node IDs
    """
    gt_node_ids = list(gt_nodes.keys())
    pred_node_ids = list(pred_nodes.keys())
    n_gt = len(gt_node_ids)
    n_pred = len(pred_node_ids)
    if n_gt == 0 or n_pred == 0:
        return {}, list(gt_node_ids), list(pred_node_ids)

    max_dim = max(n_gt, n_pred)
    cost_matrix = np.ones((max_dim, max_dim))
    if sim_matrix is None:
        for i, gt_node in enumerate(gt_node_ids):
            for j, pred_node in enumerate(pred_node_ids):
                similarity = calculate_label_similarity(gt_nodes[gt_node], pred_nodes[pred_node])
                cost_matrix[i, j] = 1.0 - similarity
    else:
        for i in range(n_gt):
            for j in range(n_pred):
                cost_matrix[i, j] = 1.0 - sim_matrix[i, j]

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches = {}
    unmatched_gt = set(gt_node_ids)
    unmatched_pred = set(pred_node_ids)
    for i, j in zip(row_indices, col_indices):
        if i < n_gt and j < n_pred:
            similarity = 1.0 - cost_matrix[i, j]
            if similarity >= similarity_threshold:
                gt_node = gt_node_ids[i]
                pred_node = pred_node_ids[j]
                matches[gt_node] = (pred_node, similarity)
                unmatched_gt.discard(gt_node)
                unmatched_pred.discard(pred_node)
    return matches, list(unmatched_gt), list(unmatched_pred)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def calculate_node_metrics(gt_nodes, pred_nodes, matches, unmatched_gt, unmatched_pred):
    """Calculate similarity-weighted node precision, recall, F1, and accuracy."""
    n_gt = len(gt_nodes)
    n_pred = len(pred_nodes)
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 1.0
    if n_gt == 0:
        return 0.0, 1.0, 0.0, 0.0
    if n_pred == 0:
        return 1.0, 0.0, 0.0, 0.0
    total_similarity = sum(sim for _, sim in matches.values())
    precision = total_similarity / n_pred if n_pred > 0 else 0.0
    recall = total_similarity / n_gt if n_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_matches = sum(1 for _, sim in matches.values() if sim > 0.9)
    accuracy = exact_matches / max(n_gt, n_pred)
    return precision, recall, f1, accuracy


def calculate_edge_metrics(gt_edges, pred_edges, node_matches, directed=True):
    """Calculate edge precision, recall, F1, and Jaccard similarity."""
    if not node_matches:
        if len(gt_edges) == 0 and len(pred_edges) == 0:
            return 1.0, 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0, 0.0

    pred_to_gt_mapping = {pred_node: gt_node for gt_node, (pred_node, _) in node_matches.items()}
    mapped_pred_edges = set()
    for src, dst in pred_edges:
        mapped_src = pred_to_gt_mapping.get(src)
        mapped_dst = pred_to_gt_mapping.get(dst)
        if mapped_src is not None and mapped_dst is not None:
            e = (mapped_src, mapped_dst) if directed else tuple(sorted((mapped_src, mapped_dst)))
            mapped_pred_edges.add(e)

    gt_edges_set = set(gt_edges if directed else [tuple(sorted(e)) for e in gt_edges])
    true_positives = len(gt_edges_set.intersection(mapped_pred_edges))
    false_positives = len(mapped_pred_edges - gt_edges_set)
    false_negatives = len(gt_edges_set - mapped_pred_edges)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    union_size = len(gt_edges_set.union(mapped_pred_edges))
    jaccard = true_positives / union_size if union_size > 0 else 0.0
    return precision, recall, f1, jaccard


def calculate_node_pr_auc(gt_nodes, pred_nodes, num_thresholds=50):
    """Calculate Node Precision-Recall AUC across multiple similarity thresholds."""
    if len(gt_nodes) == 0 or len(pred_nodes) == 0:
        return 0.0
    gt_node_ids = list(gt_nodes.keys())
    pred_node_ids = list(pred_nodes.keys())
    sim_list = []
    sim_matrix = np.zeros((len(gt_node_ids), len(pred_node_ids)))
    for i, gt_node in enumerate(gt_node_ids):
        for j, pred_node in enumerate(pred_node_ids):
            s = calculate_label_similarity(gt_nodes[gt_node], pred_nodes[pred_node])
            sim_matrix[i, j] = s
            sim_list.append(s)
    if not sim_list:
        return 0.0
    thresholds = np.linspace(min(sim_list), max(sim_list), num_thresholds)
    precisions = []
    recalls = []
    for thr in thresholds:
        matches, unmatched_gt, unmatched_pred = find_best_node_matches(
            gt_nodes, pred_nodes, thr, sim_matrix=sim_matrix
        )
        p, r, _, _ = calculate_node_metrics(gt_nodes, pred_nodes, matches, unmatched_gt, unmatched_pred)
        precisions.append(p)
        recalls.append(r)
    if len(set(recalls)) > 1:
        sorted_pairs = sorted(zip(recalls, precisions))
        sr, sp = zip(*sorted_pairs)
        return auc(sr, sp)
    return np.mean(precisions)


def calculate_edge_pr_auc(gt_nodes, pred_nodes, gt_edges, pred_edges, num_thresholds=50, directed=True):
    """Calculate Edge Precision-Recall AUC across multiple node similarity thresholds."""
    if len(gt_edges) == 0 and len(pred_edges) == 0:
        return 1.0
    if len(gt_edges) == 0 or len(pred_edges) == 0:
        return 0.0
    gt_node_ids = list(gt_nodes.keys())
    pred_node_ids = list(pred_nodes.keys())
    sim_list = []
    sim_matrix = np.zeros((len(gt_node_ids), len(pred_node_ids)))
    for i, gt_node in enumerate(gt_node_ids):
        for j, pred_node in enumerate(pred_node_ids):
            s = calculate_label_similarity(gt_nodes[gt_node], pred_nodes[pred_node])
            sim_matrix[i, j] = s
            sim_list.append(s)
    if not sim_list:
        return 0.0
    thresholds = np.linspace(min(sim_list), max(sim_list), num_thresholds)
    precisions = []
    recalls = []
    for thr in thresholds:
        matches, _, _ = find_best_node_matches(gt_nodes, pred_nodes, thr, sim_matrix=sim_matrix)
        ep, er, _, _ = calculate_edge_metrics(gt_edges, pred_edges, matches, directed=directed)
        precisions.append(ep)
        recalls.append(er)
    if len(set(recalls)) > 1:
        sorted_pairs = sorted(zip(recalls, precisions))
        sr, sp = zip(*sorted_pairs)
        return auc(sr, sp)
    return np.mean(precisions)


# ---------------------------------------------------------------------------
# TikZ parsing
# ---------------------------------------------------------------------------

NODE_RE = re.compile(
    r'\\node\b(?:\s*\[.*?\])?\s*\((?P<id>[^)]+)\)\s*(?:\{(?P<label>.*?)\})?\s*;?',
    re.DOTALL
)
EDGE_RES = [
    re.compile(r'\((?P<src>[^)]+)\)\s*--\s*\((?P<dst>[^)]+)\)'),
    re.compile(r'\((?P<src>[^)]+)\)\s*->\s*\((?P<dst>[^)]+)\)'),
    re.compile(r'\((?P<src>[^)]+)\)\s*edge\b(?:\s*node\s*\{.*?\})?\s*\((?P<dst>[^)]+)\)'),
]


def parse_tikz_content(content):
    """
    Parse TikZ content and extract nodes and edges.

    Returns:
        nodes: dict mapping node ID -> label
        edges: list of (source, destination) tuples
    """
    nodes = {}
    for m in NODE_RE.finditer(content):
        nid = m.group('id').strip()
        label = m.group('label') or nid
        nodes[nid] = label.strip()

    edges = []
    for ere in EDGE_RES:
        for m in ere.finditer(content):
            s = m.group('src').strip()
            d = m.group('dst').strip()
            edges.append((s, d))

    # Ensure nodes referenced in edges are present
    paren_ids = re.findall(r'\(([^)]+)\)', content)
    for pid in paren_ids:
        pid = pid.strip()
        if pid not in nodes:
            nodes[pid] = pid

    # Deduplicate edges preserving order
    seen = set()
    dedup_edges = []
    for e in edges:
        if e not in seen:
            dedup_edges.append(e)
            seen.add(e)

    return nodes, dedup_edges


def calculate_graph_metrics_from_dicts(gt_nodes, gt_edges, pred_nodes, pred_edges,
                                       similarity_threshold=0.3, directed=True):
    """Calculate all graph metrics from pre-parsed node/edge dictionaries."""
    matches, unmatched_gt, unmatched_pred = find_best_node_matches(
        gt_nodes, pred_nodes, similarity_threshold
    )
    node_precision, node_recall, node_f1, node_accuracy = calculate_node_metrics(
        gt_nodes, pred_nodes, matches, unmatched_gt, unmatched_pred
    )
    edge_precision, edge_recall, edge_f1, edge_jaccard = calculate_edge_metrics(
        gt_edges, pred_edges, matches, directed=directed
    )
    node_pr_auc = calculate_node_pr_auc(gt_nodes, pred_nodes)
    edge_pr_auc = calculate_edge_pr_auc(
        gt_nodes, pred_nodes, gt_edges, pred_edges, directed=directed
    )
    return {
        'node_matches': len(matches),
        'node_precision': node_precision,
        'node_recall': node_recall,
        'node_f1': node_f1,
        'node_accuracy': node_accuracy,
        'node_pr_auc': node_pr_auc,
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'edge_f1': edge_f1,
        'edge_jaccard': edge_jaccard,
        'edge_pr_auc': edge_pr_auc,
        'unmatched_gt_nodes': len(unmatched_gt),
        'unmatched_pred_nodes': len(unmatched_pred),
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _empty_result(filename, error_msg):
    """Return a zero-valued result dict for a file that could not be processed."""
    return {
        'filename': filename,
        'node_matches': 0,
        'node_precision': 0.0,
        'node_recall': 0.0,
        'node_f1': 0.0,
        'node_accuracy': 0.0,
        'node_pr_auc': 0.0,
        'edge_precision': 0.0,
        'edge_recall': 0.0,
        'edge_f1': 0.0,
        'edge_jaccard': 0.0,
        'edge_pr_auc': 0.0,
        'unmatched_gt_nodes': 0,
        'unmatched_pred_nodes': 0,
        'error': error_msg,
    }


def process_files_from_tsv(gt_folder, pred_folder, test_tsv_file,
                            output_file=None, similarity_threshold=0.3, directed=True):
    """
    Process TikZ files listed in a TSV and calculate graph metrics.

    Args:
        gt_folder: Directory containing ground truth TikZ files.
        pred_folder: Directory containing predicted TikZ files.
        test_tsv_file: TSV file with a 'File Name' column.
        output_file: Optional CSV path to save results.
        similarity_threshold: Minimum similarity for node matching.
        directed: Whether to treat edges as directed.

    Returns:
        DataFrame with per-file and average metrics, or None on failure.
    """
    try:
        test_df = pd.read_csv(test_tsv_file, sep='\t')
        if 'File Name' not in test_df.columns:
            print("Error: 'File Name' column not found in test TSV file")
            return None
        files_to_process = test_df['File Name'].tolist()
        print(f"Found {len(files_to_process)} files to process from {test_tsv_file}")
    except Exception as e:
        print(f"Error reading test TSV file: {e}")
        return None

    results = []
    for filename in files_to_process:
        tikz_filename = filename.replace('.png', '.tikz').replace('.png', '.tex')
        gt_path = os.path.join(gt_folder, tikz_filename)
        pred_path = os.path.join(pred_folder, tikz_filename)

        print(f"Processing: {tikz_filename}")

        if not os.path.exists(gt_path):
            print(f"  Ground truth file not found: {gt_path}")
            results.append(_empty_result(filename, 'GT file not found'))
            continue

        if not os.path.exists(pred_path):
            print(f"  Prediction file not found: {pred_path}")
            results.append(_empty_result(filename, 'Pred file not found'))
            continue

        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_content = f.read()
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_content = f.read()
        except Exception as e:
            print(f"  Error reading files for {filename}: {e}")
            continue

        gt_nodes, gt_edges = parse_tikz_content(gt_content)
        pred_nodes, pred_edges = parse_tikz_content(pred_content)

        metrics = calculate_graph_metrics_from_dicts(
            gt_nodes, gt_edges, pred_nodes, pred_edges, similarity_threshold, directed
        )
        if metrics:
            results.append({'filename': filename, **metrics})
        else:
            print(f"  Failed to calculate metrics for {filename}")

    if results:
        results_df = pd.DataFrame(results)
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        avg_row = {'filename': 'AVERAGE'}
        for col in numeric_columns:
            avg_row[col] = results_df[col].mean()
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        if output_file:
            try:
                results_df.to_csv(output_file, index=False)
                print(f"\nResults saved to {output_file}")
            except Exception as e:
                print(f"Failed to save results to {output_file}: {e}")
        return results_df
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Calculate graph metrics for TikZ (.tikz/.tex) architecture diagram files.'
    )
    parser.add_argument('--gt_folder', type=str, required=True,
                        help='Path to ground truth TikZ files folder')
    parser.add_argument('--pred_folder', type=str, required=True,
                        help='Path to predicted TikZ files folder')
    parser.add_argument('--test_tsv_file', type=str,
                        help='Path to test TSV file with filenames (column: File Name)')
    parser.add_argument('--output', type=str,
                        help='Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Similarity threshold for node matching (default: 0.3)')
    parser.add_argument('--single_file', type=str,
                        help='Evaluate a single file instead of the TSV list')
    parser.add_argument('--directed', action='store_true',
                        help='Treat edges as directed (default: undirected)')

    args = parser.parse_args()
    directed = args.directed

    if args.single_file:
        gt_path = os.path.join(args.gt_folder, args.single_file)
        pred_path = os.path.join(args.pred_folder, args.single_file)
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Files not found: {gt_path} or {pred_path}")
            return
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_content = f.read()
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_content = f.read()
        gt_nodes, gt_edges = parse_tikz_content(gt_content)
        pred_nodes, pred_edges = parse_tikz_content(pred_content)
        metrics = calculate_graph_metrics_from_dicts(
            gt_nodes, gt_edges, pred_nodes, pred_edges, args.threshold, directed
        )
        if metrics:
            print(f"\nResults for {args.single_file}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("Failed to compute metrics for the single file.")
    else:
        if not args.test_tsv_file:
            print("Error: --test_tsv_file is required when --single_file is not provided.")
            return
        results_df = process_files_from_tsv(
            args.gt_folder,
            args.pred_folder,
            args.test_tsv_file,
            args.output,
            args.threshold,
            directed,
        )
        if results_df is not None and len(results_df) > 0:
            avg_row = results_df.iloc[-1]
            print("\nSummary Results:")
            print(f"  Node Precision: {avg_row['node_precision']:.4f}")
            print(f"  Node Recall:    {avg_row['node_recall']:.4f}")
            print(f"  Node F1:        {avg_row['node_f1']:.4f}")
            print(f"  Node PR-AUC:    {avg_row['node_pr_auc']:.4f}")
            print(f"  Edge Precision:  {avg_row['edge_precision']:.4f}")
            print(f"  Edge Recall:     {avg_row['edge_recall']:.4f}")
            print(f"  Edge F1:         {avg_row['edge_f1']:.4f}")
            print(f"  Edge Jaccard:    {avg_row['edge_jaccard']:.4f}")
            print(f"  Edge PR-AUC:     {avg_row['edge_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
