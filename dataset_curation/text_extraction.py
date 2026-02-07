import argparse
import json
import multiprocessing
import os
import re

import pypdf
import requests
from pypdf import PdfWriter, PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_figures(input_string):
    """Extract figure reference variations from a filename string.

    Args:
        input_string: Filename containing figure references (e.g., "Figure 3").

    Returns:
        List of figure reference variations (e.g., "Figure 3", "Fig 3", etc.).
    """
    pattern = r'\b(?:Figure|Fig|figure|fig|Figure\.|Fig\.|figure\.|fig\.)\s*([^\-]+?)\s*-\b'
    matches = re.findall(pattern, input_string)
    figure_variations = []
    for match in matches:
        figure_variations.extend([
            f"Figure {match}", f"Fig {match}", f"Figure. {match}",
            f"fig {match}", f"fig. {match}", f"figure. {match}",
        ])
    return list(set(figure_variations))


def extract_arxiv_id(input_string):
    """Extract an arXiv ID from a string.

    Args:
        input_string: String potentially containing an arXiv ID.

    Returns:
        The arXiv ID string, or None if not found.
    """
    match = re.search(r'\d{4}\.\d{0,6}v\d{0,6}', input_string)
    return match.group(0) if match else None


def download_arxiv_pdf(arxiv_id, save_path):
    """Download a PDF from arXiv by its ID.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.12345v1").
        save_path: Local path to save the downloaded PDF.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")


def remove_figures_and_captions(pdf_path, output_path):
    """Create a copy of a PDF with figure/caption lines filtered out.

    Removes lines containing figure references to avoid matching the
    caption text itself during similarity computation.

    Args:
        pdf_path: Path to the input PDF.
        output_path: Path to write the filtered PDF.
    """
    pdf_reader = PdfReader(pdf_path)
    pdf_writer = PdfWriter()
    figure_pattern = re.compile(r'(?i)\b(?:Figure|Fig)\s+\d+\b')
    caption_pattern = re.compile(r'(?i)^.*\b(?:Figure|Fig)\s+\d+\b.*$', re.MULTILINE)

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            lines = text.split('\n')
            filtered_lines = [
                line for line in lines
                if not figure_pattern.search(line) and not caption_pattern.search(line)
            ]
            pdf_writer.add_page(page)

    with open(output_path, "wb") as output_file:
        pdf_writer.write(output_file)


def find_all_indices(text, keyword):
    """Find all occurrence indices of a keyword in text (case-insensitive).

    Args:
        text: The text to search in.
        keyword: The keyword to find.

    Returns:
        List of starting indices where the keyword appears.
    """
    indices = []
    keyword_lower = keyword.lower()
    start_index = 0
    while True:
        index = text.lower().find(keyword_lower, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + len(keyword)
    return indices


def extract_surrounding_text(pdf_path, keyword, surrounding_chars=300):
    """Extract text surrounding keyword occurrences in a PDF.

    Args:
        pdf_path: Path to the PDF file.
        keyword: The keyword to search for.
        surrounding_chars: Number of characters to extract after the keyword.

    Returns:
        List of text excerpts surrounding each keyword occurrence.
    """
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        surrounding_texts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            wrong_index = text.lower().find(keyword.lower() + ":")
            keyword_indices = find_all_indices(text, keyword)
            for keyword_index in keyword_indices:
                if keyword_index != wrong_index:
                    start_index = max(0, keyword_index)
                    end_index = min(len(text), keyword_index + len(keyword) + 2 * surrounding_chars)
                    surrounding_text = text[start_index:end_index]
                    surrounding_texts.append(surrounding_text)
    return surrounding_texts


def calculate_cosine_similarity(text1, text2):
    """Compute TF-IDF cosine similarity between two text strings.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Cosine similarity score (float), or 0 if either text is empty.
    """
    if not text1 or not text2:
        return 0
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]


def process_ocr_result(ocr_data, json_ocr_text):
    """Combine OCR results from multiple sources into a single text string.

    Args:
        ocr_data: OCR data dict with an "ocr_result" key.
        json_ocr_text: Additional OCR text list from Florence model.

    Returns:
        Combined OCR text string.
    """
    if not ocr_data or "ocr_result" not in ocr_data:
        return ""
    ocr_texts = [entry["text"] for entry in ocr_data["ocr_result"]]
    ocr_texts = list(set(ocr_texts + json_ocr_text))
    return " ".join(ocr_texts)


def load_json_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r") as file:
        return json.load(file)


def process_file(f, json_data, json_ocr_path, pdf_cache_dir, output_json_dir):
    """Process a single image file: download paper, extract text, compute similarity.

    Args:
        f: Image filename.
        json_data: List of figure metadata dicts.
        json_ocr_path: Directory containing per-image OCR JSONs (or empty string).
        pdf_cache_dir: Directory to cache downloaded PDFs.
        output_json_dir: Directory to write output JSON files.
    """
    try:
        image_name = f[:-4]
        json_ocr = ""
        if json_ocr_path:
            ocr_file = os.path.join(json_ocr_path, f"{image_name}.json")
            if os.path.isfile(ocr_file):
                json_ocr = load_json_data(ocr_file)

        output_file = os.path.join(output_json_dir, f"{image_name}.json")
        if os.path.isfile(output_file):
            return

        arxiv_id = extract_arxiv_id(image_name)
        if not arxiv_id:
            print(f"No valid arXiv ID found for {image_name}")
            return

        try:
            pdf_path = os.path.join(pdf_cache_dir, f"{arxiv_id}.pdf")
            if not os.path.isfile(pdf_path):
                download_arxiv_pdf(arxiv_id, pdf_path)

            modified_pdf_path = f"modified_{arxiv_id}.pdf"
            remove_figures_and_captions(pdf_path, modified_pdf_path)

            keywords = extract_figures(image_name)
            results = {}
            k = 0
            json_ocr_text = []
            if json_ocr:
                json_ocr_text = json_ocr.get("<OCR_WITH_REGION>", {}).get("labels", [])

            surrounding_cluster_texts = []
            for keyword in keywords:
                surrounding_texts = extract_surrounding_text(
                    modified_pdf_path, keyword, surrounding_chars=3000
                )
                if surrounding_texts:
                    surrounding_cluster_texts.append(surrounding_texts)
            if not surrounding_cluster_texts:
                surrounding_cluster_texts = [[""]]

            for surrounding_texts in surrounding_cluster_texts:
                for extracted_text in surrounding_texts:
                    k += 1
                    json_item = next(
                        (item for item in json_data if item["figure_id"] == image_name),
                        None,
                    )
                    if json_item:
                        captions = json_item["captions"]
                        ocr_data = json_item.get("ocr_result", {})
                        ocr_text = process_ocr_result(ocr_data, json_ocr_text)
                        fig_caption = captions[0] if captions else ""
                        fig_caption_ocr = f"{fig_caption}{ocr_text}"
                        similarity = calculate_cosine_similarity(fig_caption_ocr, extracted_text)

                        results[f"{image_name}_{k}"] = {
                            "image name": image_name,
                            "#candidate": k,
                            "chosen_text": extracted_text,
                            "similarity": similarity,
                            "caption": fig_caption,
                            "OCR": ocr_text,
                        }

            # Clean up temporary files
            if os.path.isfile(modified_pdf_path):
                os.remove(modified_pdf_path)

            with open(output_file, "w") as outfile:
                json.dump(results, outfile, indent=2)
        except Exception as e:
            print(f"Error with {image_name}: {e}")
    except Exception as e:
        print(f"Outer error with {f}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from arXiv PDFs and match with figure data."
    )
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the directory of architecture diagram images")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the JSON file with figure metadata")
    parser.add_argument("--json_ocr_path", type=str, default="",
                        help="Path to the directory with per-image Florence OCR JSONs")
    parser.add_argument("--pdf_cache_dir", type=str, default="./pdf_cache",
                        help="Directory to cache downloaded arXiv PDFs")
    parser.add_argument("--output_json_dir", type=str, required=True,
                        help="Directory to save the output JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_json_dir, exist_ok=True)
    os.makedirs(args.pdf_cache_dir, exist_ok=True)

    json_data = load_json_data(args.json_path)
    files = os.listdir(args.path)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            process_file,
            [(f, json_data, args.json_ocr_path, args.pdf_cache_dir, args.output_json_dir)
             for f in files],
        )

    print("Finished!")


if __name__ == "__main__":
    main()
