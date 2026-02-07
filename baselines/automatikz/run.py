import argparse

import pandas as pd
from tqdm import tqdm

from automatikz.infer import TikzGenerator, load


def main():
    parser = argparse.ArgumentParser(
        description="Generate TikZ code from text descriptions using AutomaTikZ."
    )
    parser.add_argument("--input", required=True, help="Input TSV file path")
    parser.add_argument("--output", required=True, help="Output TSV file path")
    parser.add_argument(
        "--batch_save", type=int, default=50,
        help="Save intermediate results every N entries (default: 50)"
    )
    args = parser.parse_args()

    # Load the TikZ-CLiMA-13B model
    generate = TikzGenerator(*load("nllg/tikz-clima-13b"), stream=False)

    # Read input TSV
    df = pd.read_csv(args.input, sep="\t")
    df["predicted_tikz"] = ""

    # Process entries with periodic saving
    for idx in tqdm(range(len(df)), desc="Generating TikZ"):
        description = df.loc[idx, "Cleaned Description"]
        tikzdoc = generate(description)
        df.loc[idx, "predicted_tikz"] = tikzdoc.code

        if (idx + 1) % args.batch_save == 0:
            df.to_csv(args.output, sep="\t", index=False)

    # Final save
    df.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
