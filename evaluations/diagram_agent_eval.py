import argparse

import pandas as pd
import Levenshtein
from rouge_score import rouge_scorer
from nltk.translate.chrf_score import sentence_chrf
from codebleu import calc_codebleu


def edit_distance(prediction, reference):
    """Calculate Levenshtein edit distance between prediction and reference."""
    return Levenshtein.distance(prediction, reference)


def rouge_l(prediction, reference):
    """Calculate ROUGE-L F-measure between prediction and reference."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure


def chrf(prediction, reference):
    """Calculate chrF score between prediction and reference."""
    return sentence_chrf(reference, prediction)


def code_bleu(predictions, references):
    """Calculate CodeBLEU for a batch of predictions and references."""
    result = calc_codebleu(
        references, predictions,
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None
    )
    return result.get('codebleu', 0.0)


def pass_at_1(prediction, reference):
    """Check exact match between stripped prediction and reference."""
    return 1 if prediction.strip() == reference.strip() else 0


def ruby(prediction, reference):
    """Calculate RUBY score (Jaccard similarity over unique tokens)."""
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())
    intersection = pred_tokens.intersection(ref_tokens)
    union = pred_tokens.union(ref_tokens)
    return len(intersection) / len(union) if union else 0.0


def calculate_metrics(tsv_file, output_file=None):
    """
    Calculate all text-based metrics for each row in a TSV file.

    Args:
        tsv_file: Path to TSV file with 'Dot code' and 'predicted_code' columns.
        output_file: Optional path to save results as CSV.

    Returns:
        DataFrame with per-row and average metrics, or an error string.
    """
    try:
        df = pd.read_csv(tsv_file, sep='\t')

        required_columns = ["Dot code", "predicted_code"]
        for col in required_columns:
            if col not in df.columns:
                return f"Error: '{col}' column not found in the TSV file"

        results = []
        all_references = []
        all_predictions = []
        pass_at_1_count = 0
        total_count = 0

        for idx, row in df.iterrows():
            reference = str(row["Dot code"])
            prediction = str(row["predicted_code"])

            if pd.isna(reference) or pd.isna(prediction) or reference.strip() == "" or prediction.strip() == "":
                continue

            all_references.append(reference)
            all_predictions.append(prediction)
            total_count += 1

            rouge_score = rouge_l(prediction, reference)
            edit_dist = edit_distance(prediction, reference)
            chrf_score = chrf(prediction, reference)
            pass_at_1_score = pass_at_1(prediction, reference)
            ruby_score = ruby(prediction, reference)

            pass_at_1_count += pass_at_1_score

            results.append({
                "File Name": row.get("File Name", f"Row {idx}"),
                "ROUGE-L": rouge_score,
                "Edit Distance": edit_dist,
                "chrF": chrf_score,
                "Pass@1": pass_at_1_score,
                "RUBY": ruby_score,
            })

        code_bleu_score = code_bleu(all_predictions, all_references)

        for result in results:
            result["CodeBLEU"] = code_bleu_score

        results_df = pd.DataFrame(results)

        avg_metrics = {
            "File Name": "AVERAGE",
            "ROUGE-L": results_df["ROUGE-L"].mean(),
            "CodeBLEU": code_bleu_score,
            "Edit Distance": results_df["Edit Distance"].mean(),
            "chrF": results_df["chrF"].mean(),
            "Pass@1": pass_at_1_count / total_count if total_count > 0 else 0,
            "RUBY": results_df["RUBY"].mean(),
        }

        results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)

        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return results_df

    except Exception as e:
        return f"Error processing file: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate text-based metrics between ground truth and predicted DOT code.'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the TSV file containing the DOT codes')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to save the results CSV file (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')

    args = parser.parse_args()

    results = calculate_metrics(args.input_file, args.output)

    if isinstance(results, pd.DataFrame):
        if args.verbose:
            print("\nDetailed metrics per file:")
            print(results.to_string(index=False))
        else:
            average_row = results.iloc[-1]
            print("\nAverage Metrics:")
            print(f"  ROUGE-L:       {average_row['ROUGE-L']:.4f}")
            print(f"  CodeBLEU:      {average_row['CodeBLEU']:.4f}")
            print(f"  Edit Distance: {average_row['Edit Distance']:.2f}")
            print(f"  chrF:          {average_row['chrF']:.4f}")
            print(f"  Pass@1:        {average_row['Pass@1']:.4f}")
            print(f"  RUBY:          {average_row['RUBY']:.4f}")

        if args.output:
            print(f"\nDetailed results saved to {args.output}")
    else:
        print(results)
