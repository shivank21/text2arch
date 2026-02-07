import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def detect_arrows(image_path):
    """Detect arrows in an image using contour analysis.

    Uses Canny edge detection and convex hull analysis to identify
    arrow-shaped contours and extract their start/end points.

    Args:
        image_path: Path to the image file.

    Returns:
        List of dicts with "start" and "end" coordinate tuples.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrow_points = []

    def is_arrow(contour):
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        return len(approx) >= 5

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        if is_arrow(contour):
            hull = cv2.convexHull(contour)
            leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
            rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
            arrow_points.append({"start": leftmost, "end": rightmost})

    return arrow_points


def process_folder(input_folder_path, output_json_path):
    """Process all images in a folder with OCR and arrow detection.

    Loads the Florence-2 model, runs OCR with region detection on each image,
    detects arrows via contour analysis, and saves results incrementally.

    Args:
        input_folder_path: Directory containing images to process.
        output_json_path: Path to write the output JSON file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print("Loading model and processor...")

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large-ft",
        trust_remote_code=True,
    )

    results = {}
    prompt = "<OCR_WITH_REGION>"
    batch_size = 500

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    valid_files = [
        f for f in os.listdir(input_folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    total_files = len(valid_files)

    print(f"Found {total_files} valid images to process")
    if not os.path.exists(output_json_path):
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    for idx, filename in enumerate(valid_files, 1):
        try:
            image_path = os.path.join(input_folder_path, filename)
            image = Image.open(image_path)

            arrow_points = detect_arrows(image_path)

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(device, torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]

            parsed_answer = processor.post_process_generation(
                generated_text,
                task="<OCR_WITH_REGION>",
                image_size=(image.width, image.height),
            )

            results[filename] = {
                "ocr": parsed_answer,
                "arrow_points": arrow_points,
            }

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results[filename] = f"Error: {str(e)}"

        # Periodically flush results to disk
        if idx % batch_size == 0 or idx == total_files:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)

            existing_results.update(results)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=4, ensure_ascii=False)

            results.clear()

    print(f"Processing complete. Results saved to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR and arrow detection on a folder of images."
    )
    parser.add_argument(
        "--input_folder", "-i",
        type=str, required=True,
        help="Path to the folder containing images to process",
    )
    parser.add_argument(
        "--output_json", "-o",
        type=str, required=True,
        help="Path where the output JSON file should be saved",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        raise ValueError(f"Input folder does not exist: {args.input_folder}")

    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_folder(args.input_folder, args.output_json)


if __name__ == "__main__":
    main()
