import argparse
import json
import logging
import os

import cv2
import numpy as np


def map_quad_boxes_centers(bounding_box_class, quad_boxes, texts):
    """Map OCR quad boxes to a detection bounding box based on center containment.

    Args:
        bounding_box_class: Tuple of (bbox_coords, class_id).
        quad_boxes: List of OCR quad boxes (8 coordinates each).
        texts: List of OCR text strings corresponding to quad_boxes.

    Returns:
        Tuple of (matched_boxes, matched_texts) that fall within the bounding box.
    """
    x_min, y_min, x_max, y_max = bounding_box_class[0]
    result_boxes = []
    result_texts = []

    for quad_box, text in zip(quad_boxes, texts):
        x1, y1, x2, y2, x3, y3, x4, y4 = quad_box
        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4

        if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
            result_boxes.append(quad_box)
            result_texts.append(text)

    return result_boxes, result_texts


def find_closest_bbox(quad_box, class_boxes, class_names):
    """Find the closest detection bounding box to an OCR quad box.

    Args:
        quad_box: OCR quad box (8 coordinates).
        class_boxes: List of detection bounding boxes.
        class_names: List of class name strings.

    Returns:
        Tuple of (closest_bbox, closest_class_name).
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = quad_box
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4

    min_dist = float("inf")
    closest_bbox = None
    closest_class_name = None

    for bbox, class_name in zip(class_boxes, class_names):
        bx_min, by_min, bx_max, by_max = bbox
        bbox_center_x = (bx_min + bx_max) / 2
        bbox_center_y = (by_min + by_max) / 2

        dist = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)

        if dist < min_dist:
            min_dist = dist
            closest_bbox = bbox
            closest_class_name = class_name

    return closest_bbox, closest_class_name


def merge_json(image_path, json_path, ocr_json):
    """Merge flowchart detection JSON with OCR JSON for a single image.

    Args:
        image_path: Path to the source image.
        json_path: Path to the flowchart detection JSON.
        ocr_json: Path to the OCR results JSON.

    Returns:
        Merged JSON dict with Grouped_Texts and Grouped_Boxes fields,
        or None on error.
    """
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(json_path, "r") as f:
            data_json = json.load(f)

        class_boxes = data_json["detection"]["boxes"]
        classes = data_json["detection"]["classes"]

        with open(ocr_json, "r") as f:
            data_ocr = json.load(f)

        text_boxes = data_ocr["<OCR_WITH_REGION>"]["quad_boxes"]
        texts = data_ocr["<OCR_WITH_REGION>"]["labels"]

        done_text = {t: False for t in texts}

        class_names = ["Node", "Text", "Arrow", "Fig Label", "Node Label"]
        texts_nodes = []
        texts_nonnodes = []
        boxes_nodes = []
        boxes_nonnodes = []

        for bbox, bbox_class in zip(class_boxes, classes):
            if bbox_class == 0:
                res_boxes, res_texts = map_quad_boxes_centers(
                    (bbox, bbox_class), text_boxes, texts
                )
                for text in res_texts:
                    if not done_text[text]:
                        done_text[text] = True
                texts_nodes.append(res_texts)
                boxes_nodes.append(res_boxes)

        # Map remaining unprocessed texts to closest bounding box
        for quad_box, text in zip(text_boxes, texts):
            if not done_text[text]:
                closest_bbox, closest_class_name = find_closest_bbox(
                    quad_box, class_boxes, class_names
                )
                if closest_bbox is not None:
                    done_text[text] = True
                    texts_nonnodes.append(text)
                    boxes_nonnodes.append(quad_box)

        grouped_texts = [""] * len(classes)
        grouped_boxes = [""] * len(classes)
        i = 0
        j = 0
        k = -1
        for c in classes:
            if c == 0 and i < len(texts_nodes):
                k += 1
                grouped_texts[k] = texts_nodes[i]
                grouped_boxes[k] = boxes_nodes[i]
                i += 1
            elif c != 0 and j < len(texts_nonnodes):
                k += 1
                grouped_texts[k] = texts_nonnodes[j]
                grouped_boxes[k] = boxes_nonnodes[j]
                j += 1

        data_json["Grouped_Texts"] = grouped_texts
        data_json["Grouped_Boxes"] = grouped_boxes
        return data_json
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None


def process_images_from_folder(image_folder_path, json_folder_path, ocr_json_folder, output_json_folder):
    """Process all images in a folder, merging detection and OCR results.

    Args:
        image_folder_path: Directory containing source images.
        json_folder_path: Directory containing flowchart detection JSONs.
        ocr_json_folder: Directory containing OCR result JSONs.
        output_json_folder: Directory to write merged output JSONs.
    """
    try:
        os.makedirs(output_json_folder, exist_ok=True)
        image_files = [
            f for f in os.listdir(image_folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for image_file in image_files:
            image_name = os.path.splitext(image_file)[0]
            json_file = f"{image_name}.json"

            json_path = os.path.join(json_folder_path, json_file)
            ocr_json = os.path.join(ocr_json_folder, json_file)
            output_path = os.path.join(output_json_folder, json_file)

            if os.path.isfile(output_path):
                logging.warning(f"Output JSON {json_file} already exists. Skipping.")
                continue

            if not os.path.isfile(json_path):
                logging.warning(f"Detection JSON {json_file} not found. Skipping.")
                continue

            if not os.path.isfile(ocr_json):
                logging.warning(f"OCR JSON {json_file} not found. Skipping.")
                continue

            image_path = os.path.join(image_folder_path, image_file)
            output_json = merge_json(image_path, json_path, ocr_json)
            if output_json is not None:
                with open(output_path, "w") as f:
                    json.dump(output_json, f, indent=4)
                logging.info(f"Saved merged output to {output_path}")
    except Exception as e:
        logging.error(f"Error processing images from folder: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge flowchart detection and OCR results."
    )
    parser.add_argument("--image_folder_path", type=str, required=True,
                        help="Path to the folder containing images")
    parser.add_argument("--json_folder_path", type=str, required=True,
                        help="Path to the folder containing flowchart detection JSONs")
    parser.add_argument("--ocr_json_folder", type=str, required=True,
                        help="Path to the folder containing OCR result JSONs")
    parser.add_argument("--output_json_folder", type=str, required=True,
                        help="Path to save merged output JSONs")

    args = parser.parse_args()
    process_images_from_folder(
        args.image_folder_path,
        args.json_folder_path,
        args.ocr_json_folder,
        args.output_json_folder,
    )


if __name__ == "__main__":
    main()
