import os
import re
import argparse
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import networkx as nx
import Levenshtein
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from nltk.translate.chrf_score import sentence_chrf
from scipy.optimize import linear_sum_assignment
from codebleu import calc_codebleu

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# DOT file loading and preprocessing
# ---------------------------------------------------------------------------

def preprocess_xml_wrapped_dot(content):
    """
    Extract DOT content from XML-wrapped format (e.g., CDATA sections).
    Returns the raw content unchanged if no XML wrapper is detected.
    """
    try:
        if '<results>' in content and '<![CDATA[' in content:
            cdata_match = re.search(r'<!\[CDATA\[(.*?)\]\]>', content, re.DOTALL)
            if cdata_match:
                return cdata_match.group(1).strip()
            try:
                root = ET.fromstring(content)
                for elem in root.iter():
                    if elem.text and 'digraph' in elem.text:
                        return elem.text.strip()
            except ET.ParseError:
                pass
        return content
    except Exception:
        return content


def load_dot_file_content(file_path):
    """Read and preprocess a single .dot file, handling XML wrappers."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        return preprocess_xml_wrapped_dot(raw_content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def load_dot_files_from_folder(folder_path):
    """Load all .dot files from a folder into a dict mapping filename -> content."""
    contents = {}
    failed_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.dot'):
            file_path = os.path.join(folder_path, filename)
            content = load_dot_file_content(file_path)
            if content:
                contents[filename] = content
            else:
                failed_files.append(filename)
    if failed_files:
        print(f"Failed to read {len(failed_files)} files: "
              f"{failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    return contents


# ---------------------------------------------------------------------------
# Text-based metrics
# ---------------------------------------------------------------------------

def edit_distance(prediction, reference):
    """Calculate Levenshtein edit distance."""
    try:
        return Levenshtein.distance(prediction, reference)
    except Exception:
        return None


def rouge_l(prediction, reference):
    """Calculate ROUGE-L F-measure."""
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    except Exception:
        return None


def chrf(prediction, reference):
    """Calculate chrF score."""
    try:
        return sentence_chrf(reference, prediction)
    except Exception:
        return None


def pass_at_1(prediction, reference):
    """Check exact match (Pass@1)."""
    try:
        return 1 if prediction.strip() == reference.strip() else 0
    except Exception:
        return None


def ruby(prediction, reference):
    """Calculate RUBY score (Jaccard similarity over unique tokens)."""
    try:
        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())
        intersection = pred_tokens.intersection(ref_tokens)
        union = pred_tokens.union(ref_tokens)
        return len(intersection) / len(union) if union else 0.0
    except Exception:
        return None


def code_bleu_batch(predictions, references):
    """Calculate CodeBLEU for a batch of predictions and references."""
    result = calc_codebleu(
        references, predictions,
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None,
    )
    return result.get('codebleu', 0.0)


# ---------------------------------------------------------------------------
# Graph-based metrics
# ---------------------------------------------------------------------------

def load_dot_as_graph(file_path):
    """Load a .dot file as a NetworkX graph."""
    try:
        return nx.drawing.nx_agraph.read_dot(file_path)
    except Exception:
        return None


def get_node_labels(graph):
    """Extract node labels from a graph."""
    try:
        return {node: graph.nodes[node].get('label', node) for node in graph.nodes()}
    except Exception:
        return {}


def normalize_label(label):
    """Normalize a label for comparison: lowercase, collapse whitespace."""
    if not label:
        return ""
    normalized = re.sub(r'\s+', ' ', label.replace('\n', ' ').replace('\r', ' '))
    return normalized.strip().lower()


def string_similarity(str1, str2):
    """Calculate similarity using sequence matching and Jaccard token similarity."""
    norm_str1 = normalize_label(str1)
    norm_str2 = normalize_label(str2)

    basic_sim = SequenceMatcher(None, norm_str1, norm_str2).ratio()

    tokens1 = set(norm_str1.split())
    tokens2 = set(norm_str2.split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    token_sim = len(intersection) / len(union) if union else 0.0

    return max(basic_sim, token_sim)


def find_matching_nodes_hungarian(graph1, graph2, threshold=0.5):
    """
    Find optimal node matching between two graphs using the Hungarian algorithm.
    Returns dict mapping graph1 node -> (graph2 node, similarity score).
    """
    try:
        labels1 = get_node_labels(graph1)
        labels2 = get_node_labels(graph2)
        nodes1 = list(labels1.keys())
        nodes2 = list(labels2.keys())
        n1 = len(nodes1)
        n2 = len(nodes2)

        if n1 == 0 or n2 == 0:
            return {}

        cost_matrix = np.ones((n1, n2)) * -threshold
        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                score = string_similarity(labels1[node1], labels2[node2])
                if score >= threshold:
                    cost_matrix[i, j] = -score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < -threshold:
                matches[nodes1[i]] = (nodes2[j], -cost_matrix[i, j])

        return matches
    except Exception:
        return {}


def calculate_node_metrics(graph1, graph2, matches):
    """Calculate similarity-weighted node precision, recall, and F1."""
    try:
        labels1 = get_node_labels(graph1)
        labels2 = get_node_labels(graph2)
        if not matches or not labels1 or not labels2:
            return 0, 0, 0

        sum_similarities = sum(score for _, score in matches.values())
        node_precision = sum_similarities / len(labels2)
        node_recall = sum_similarities / len(labels1)
        f1 = (2 * (node_precision * node_recall) / (node_precision + node_recall)
              if (node_precision + node_recall) > 0 else 0)
        return node_precision, node_recall, f1
    except Exception:
        return 0, 0, 0


def create_adjacency_matrix(graph, node_order):
    """Create an adjacency matrix for the given node ordering."""
    try:
        n = len(node_order)
        matrix = np.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(node_order)}
        for edge in graph.edges():
            if edge[0] in node_to_idx and edge[1] in node_to_idx:
                i = node_to_idx[edge[0]]
                j = node_to_idx[edge[1]]
                matrix[i, j] = 1
        return matrix
    except Exception:
        return np.zeros((len(node_order), len(node_order)))


def calculate_edge_metrics(graph1, graph2, node_matches):
    """Calculate edge precision, recall, and Jaccard similarity."""
    try:
        if not node_matches:
            return 0.0, 0.0, 0.0

        matched_nodes1 = list(node_matches.keys())
        matched_nodes2 = [match[0] for match in node_matches.values()]

        adj1 = create_adjacency_matrix(graph1, matched_nodes1)
        adj2 = create_adjacency_matrix(graph2, matched_nodes2)

        tp = np.sum(np.logical_and(adj1 == 1, adj2 == 1))
        fp = np.sum(np.logical_and(adj1 == 0, adj2 == 1))
        fn = np.sum(np.logical_and(adj1 == 1, adj2 == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        intersection = np.sum(np.logical_and(adj1, adj2))
        union = np.sum(np.logical_or(adj1, adj2))
        jaccard = intersection / union if union > 0 else 0

        return precision, recall, jaccard
    except Exception:
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Combined evaluation pipeline
# ---------------------------------------------------------------------------

def calculate_combined_metrics(gt_folder, pred_folder, test_tsv_file, output_file=None,
                               include_graph_metrics=True, threshold=0.5, skip_on_error=True):
    """
    Calculate text and graph metrics for .dot files listed in a test TSV.

    Args:
        gt_folder: Directory with ground truth .dot files.
        pred_folder: Directory with predicted .dot files.
        test_tsv_file: TSV file with a 'File Name' column.
        output_file: Optional CSV path to save results.
        include_graph_metrics: Whether to compute graph-structural metrics.
        threshold: Similarity threshold for node matching.
        skip_on_error: Skip files with errors (True) or assign zero scores (False).

    Returns:
        DataFrame with per-file and average metrics, or an error string.
    """
    try:
        test_df = pd.read_csv(test_tsv_file, sep='\t')
        if 'File Name' not in test_df.columns:
            return "Error: 'File Name' column not found in test TSV file"
        files_to_process = test_df['File Name'].tolist()
        print(f"Found {len(files_to_process)} files to process from test TSV")
    except Exception as e:
        return f"Error reading test TSV file: {e}"

    gt_contents = load_dot_files_from_folder(gt_folder)
    pred_contents = load_dot_files_from_folder(pred_folder)
    print(f"Found {len(gt_contents)} ground truth files and {len(pred_contents)} predicted files")

    results = []
    all_references = []
    all_predictions = []
    pass_at_1_count = 0
    total_count = 0
    skipped_files = []
    missing_gt_files = []
    missing_pred_files = []
    error_summary = {
        'text_errors': 0, 'graph_errors': 0,
        'total_skipped': 0, 'missing_gt': 0, 'missing_pred': 0,
    }

    for filename in files_to_process:
        filename = filename.replace(".png", ".dot")
        gt_file_exists = filename in gt_contents
        pred_file_exists = filename in pred_contents

        if not gt_file_exists:
            missing_gt_files.append(filename)
            error_summary['missing_gt'] += 1
        if not pred_file_exists:
            missing_pred_files.append(filename)
            error_summary['missing_pred'] += 1

        if not gt_file_exists or not pred_file_exists:
            result = {
                "File Name": filename,
                'ROUGE-L': 0.0, 'Edit Distance': 0.0, 'chrF': 0.0,
                'Pass@1': 0, 'RUBY': 0.0, 'CodeBLEU': 0.0,
            }
            if include_graph_metrics:
                result.update({
                    "Node Matches": 0, "Node Precision": 0.0, "Node Recall": 0.0,
                    "Node F1": 0.0, "Edge Precision": 0.0, "Edge Recall": 0.0,
                    "Jaccard Similarity": 0.0,
                })
            results.append(result)
            all_references.append("")
            all_predictions.append("")
            total_count += 1
            continue

        reference = gt_contents[filename]
        prediction = pred_contents[filename]

        if not reference.strip() or not prediction.strip():
            skipped_files.append(f"{filename}: empty content")
            continue

        # --- Text metrics ---
        text_metrics = {}
        text_failed = False

        rouge_score = rouge_l(prediction, reference)
        if rouge_score is None:
            text_failed = True
        else:
            text_metrics['ROUGE-L'] = rouge_score

        edit_dist = edit_distance(prediction, reference)
        if edit_dist is None:
            text_failed = True
        else:
            text_metrics['Edit Distance'] = edit_dist

        chrf_score = chrf(prediction, reference)
        if chrf_score is None:
            text_failed = True
        else:
            text_metrics['chrF'] = chrf_score

        pass_at_1_score = pass_at_1(prediction, reference)
        if pass_at_1_score is None:
            text_failed = True
        else:
            text_metrics['Pass@1'] = pass_at_1_score

        ruby_score = ruby(prediction, reference)
        if ruby_score is None:
            text_failed = True
        else:
            text_metrics['RUBY'] = ruby_score

        if text_failed and skip_on_error:
            skipped_files.append(f"{filename}: text metric calculation failed")
            error_summary['text_errors'] += 1
            continue

        if text_failed and not skip_on_error:
            text_metrics.update({
                'ROUGE-L': 0.0, 'Edit Distance': 0.0, 'chrF': 0.0,
                'Pass@1': 0, 'RUBY': 0.0,
            })

        all_references.append(reference)
        all_predictions.append(prediction)
        total_count += 1

        if text_metrics.get('Pass@1', 0) == 1:
            pass_at_1_count += 1

        result = {"File Name": filename, **text_metrics}

        # --- Graph metrics ---
        if include_graph_metrics:
            gt_file_path = os.path.join(gt_folder, filename)
            pred_file_path = os.path.join(pred_folder, filename)

            gt_graph = load_dot_as_graph(gt_file_path)
            pred_graph = load_dot_as_graph(pred_file_path)
            graph_failed = gt_graph is None or pred_graph is None

            if graph_failed and skip_on_error:
                if not text_failed:
                    skipped_files.append(f"{filename}: graph parsing failed")
                    error_summary['graph_errors'] += 1
                    continue

            if not graph_failed:
                try:
                    node_matches = find_matching_nodes_hungarian(gt_graph, pred_graph, threshold)
                    node_precision, node_recall, node_f1 = calculate_node_metrics(
                        gt_graph, pred_graph, node_matches
                    )
                    edge_precision, edge_recall, jaccard = calculate_edge_metrics(
                        gt_graph, pred_graph, node_matches
                    )
                    result.update({
                        "Node Matches": len(node_matches),
                        "Node Precision": node_precision,
                        "Node Recall": node_recall,
                        "Node F1": node_f1,
                        "Edge Precision": edge_precision,
                        "Edge Recall": edge_recall,
                        "Jaccard Similarity": jaccard,
                    })
                except Exception as e:
                    print(f"Error calculating graph metrics for {filename}: {e}")
                    if skip_on_error:
                        skipped_files.append(f"{filename}: graph metric calculation failed")
                        error_summary['graph_errors'] += 1
                        continue
                    else:
                        result.update({
                            "Node Matches": 0, "Node Precision": 0, "Node Recall": 0,
                            "Node F1": 0, "Edge Precision": 0, "Edge Recall": 0,
                            "Jaccard Similarity": 0,
                        })
            else:
                if not skip_on_error:
                    result.update({
                        "Node Matches": 0, "Node Precision": 0, "Node Recall": 0,
                        "Node F1": 0, "Edge Precision": 0, "Edge Recall": 0,
                        "Jaccard Similarity": 0,
                    })

        results.append(result)

    # Report missing/skipped files
    if missing_gt_files:
        print(f"\nMissing ground truth files: {len(missing_gt_files)}")
        for f in missing_gt_files[:10]:
            print(f"  - {f}")
        if len(missing_gt_files) > 10:
            print(f"  ... and {len(missing_gt_files) - 10} more")

    if missing_pred_files:
        print(f"\nMissing prediction files: {len(missing_pred_files)}")
        for f in missing_pred_files[:10]:
            print(f"  - {f}")
        if len(missing_pred_files) > 10:
            print(f"  ... and {len(missing_pred_files) - 10} more")

    if skipped_files:
        error_summary['total_skipped'] = len(skipped_files)
        print(f"\nSkipped {len(skipped_files)} files due to errors:")
        for reason in skipped_files[:10]:
            print(f"  - {reason}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
        print(f"\nError Summary:")
        print(f"  Text metric errors:       {error_summary['text_errors']}")
        print(f"  Graph metric errors:       {error_summary['graph_errors']}")
        print(f"  Missing ground truth:      {error_summary['missing_gt']}")
        print(f"  Missing predictions:       {error_summary['missing_pred']}")
        print(f"  Total skipped:             {error_summary['total_skipped']}")
        print(f"  Successfully processed:    {len(results) - error_summary['missing_gt'] - error_summary['missing_pred']}")

    if not results:
        return "Error: No files could be processed successfully"

    # CodeBLEU (batch)
    code_bleu_score = 0.0
    if all_predictions and all_references:
        code_bleu_score = code_bleu_batch(all_predictions, all_references)

    for result in results:
        result["CodeBLEU"] = code_bleu_score

    results_df = pd.DataFrame(results)

    # Calculate averages
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    avg_metrics = {}
    for col in results_df.columns:
        if col not in numeric_columns:
            avg_metrics[col] = "AVERAGE"
    for col in numeric_columns:
        if col == "Pass@1":
            avg_metrics[col] = pass_at_1_count / total_count if total_count > 0 else 0
        else:
            avg_metrics[col] = results_df[col].mean()

    results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)

    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Calculate combined text and graph metrics for .dot files.'
    )
    parser.add_argument('--gt_folder', type=str, required=True,
                        help='Path to the folder containing ground truth .dot files')
    parser.add_argument('--pred_folder', type=str, required=True,
                        help='Path to the folder containing predicted .dot files')
    parser.add_argument('--test_tsv_file', type=str, required=True,
                        help='Path to the test TSV file with file names to process')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to save the results CSV file (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')
    parser.add_argument('--text_only', action='store_true',
                        help='Calculate only text-based metrics (skip graph metrics)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for node matching (default: 0.5)')
    parser.add_argument('--assign_zero_on_error', action='store_true', default=False,
                        help='Assign 0 scores for failed metrics instead of skipping files')

    args = parser.parse_args()

    include_graph_metrics = not args.text_only
    skip_on_error = not args.assign_zero_on_error

    results = calculate_combined_metrics(
        args.gt_folder,
        args.pred_folder,
        args.test_tsv_file,
        args.output,
        include_graph_metrics=include_graph_metrics,
        threshold=args.threshold,
        skip_on_error=skip_on_error,
    )

    if isinstance(results, pd.DataFrame):
        if args.verbose:
            print("\nDetailed metrics per file:")
            print(results.to_string(index=False))
        else:
            average_row = results.iloc[-1]
            print("\nAverage Text Metrics:")
            print(f"  ROUGE-L:       {average_row['ROUGE-L']:.4f}")
            print(f"  CodeBLEU:      {average_row['CodeBLEU']:.4f}")
            print(f"  Edit Distance: {average_row['Edit Distance']:.2f}")
            print(f"  chrF:          {average_row['chrF']:.4f}")
            print(f"  Pass@1:        {average_row['Pass@1']:.4f}")
            print(f"  RUBY:          {average_row['RUBY']:.4f}")

            if include_graph_metrics:
                print("\nAverage Graph Metrics:")
                print(f"  Node Matches:      {average_row['Node Matches']:.1f}")
                print(f"  Node F1:           {average_row['Node F1']:.4f}")
                print(f"  Edge Precision:    {average_row['Edge Precision']:.4f}")
                print(f"  Edge Recall:       {average_row['Edge Recall']:.4f}")
                print(f"  Jaccard Similarity: {average_row['Jaccard Similarity']:.4f}")

        if args.output:
            print(f"\nDetailed results saved to {args.output}")
    else:
        print(results)


if __name__ == "__main__":
    main()
