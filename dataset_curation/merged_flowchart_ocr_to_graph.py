import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from scipy.spatial import distance


def download_image(url, save_path):
    """Download an image from a URL and save it locally.

    Args:
        url: Image URL to download.
        save_path: Local path to save the image.

    Returns:
        The save_path on success, or None on failure.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def load_json(json_path):
    """Load and return JSON data from a file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON {json_path}: {e}")
        return None


def extract_grouped_boxes(data):
    """Extract grouped bounding boxes from merged JSON data."""
    return data.get("Grouped_Boxes", [])


def extract_bounding_boxes(data, label=None):
    """Extract bounding boxes for a specific class label.

    Args:
        data: Merged JSON data dict.
        label: Class label to filter by (0=Node, 2=Arrow, etc.).

    Returns:
        List of (bbox, index) tuples matching the label.
    """
    try:
        return [
            (box, i) for i, (box, cls)
            in enumerate(zip(data["detection"]["boxes"], data["detection"]["classes"]))
            if cls == label
        ]
    except KeyError:
        print("Bounding boxes or classes not found in JSON.")
        return []


def find_nearest_node(point, node_boxes, grouped_boxes):
    """Find the nearest node bounding box to a given point.

    Only considers nodes that have associated grouped text (non-empty).

    Args:
        point: (x, y) coordinates.
        node_boxes: List of (bbox, index) tuples for nodes.
        grouped_boxes: List of grouped box data from merged JSON.

    Returns:
        The (bbox, index) tuple of the nearest valid node, or None.
    """
    try:
        valid_nodes = [
            (box, i) for box, i in node_boxes
            if grouped_boxes[i] and grouped_boxes[i] != ""
        ]
        if not valid_nodes:
            return None
        return min(
            valid_nodes,
            key=lambda n: distance.euclidean(
                point, ((n[0][0] + n[0][2]) / 2, (n[0][1] + n[0][3]) / 2)
            ),
        )
    except Exception as e:
        print(f"Error finding nearest node: {e}")
        return None


def generate_dot_file(output_path, arrow_connections, node_labels):
    """Write a DOT graph file from arrow connections and node labels.

    Args:
        output_path: Path to write the DOT file.
        arrow_connections: List of (source_idx, target_idx) tuples.
        node_labels: Dict mapping node index to label string.
    """
    try:
        with open(output_path, "w") as f:
            f.write("digraph {\n")
            for i, label in node_labels.items():
                f.write(f'  {i} [label="{label}"]\n')
            for start, end in arrow_connections:
                f.write(f"  {start} -> {end}\n")
            f.write("}\n")
    except Exception as e:
        print(f"Error generating DOT file: {e}")


def load_tsv_to_dict(tsv_path):
    """Load a TSV file mapping image names to URLs.

    Args:
        tsv_path: Path to TSV file with (image_name, url) rows.

    Returns:
        Dict mapping image name to URL.
    """
    tsv_dict = {}
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    tsv_dict[row[0]] = row[1]
    except Exception as e:
        print(f"Error reading TSV file: {e}")
    return tsv_dict


def process_json_file(json_path, image_url, output_dot_folder):
    """Process a single merged JSON file and generate a DOT graph.

    Downloads the corresponding image, extracts node and arrow bounding boxes,
    determines arrow connectivity, and writes the DOT file.

    Args:
        json_path: Path to the merged JSON file.
        image_url: URL of the source image.
        output_dot_folder: Directory to save the DOT file.
    """
    json_name = os.path.basename(json_path).replace(".json", "")
    output_dot_path = os.path.join(output_dot_folder, f"{json_name}.dot")

    image_path = download_image(image_url, f"{json_name}.png")
    if not image_path:
        return

    data = load_json(json_path)
    if not data:
        return

    node_boxes = extract_bounding_boxes(data, 0)
    arrow_boxes = extract_bounding_boxes(data, 2)
    grouped_boxes = extract_grouped_boxes(data)

    arrow_connections = []
    for box, arrow_idx in arrow_boxes:
        x1, y1, x2, y2 = map(int, box)
        # Estimate arrow head and tail from bounding box extremes
        head_global = (x1, (y1 + y2) // 2)
        tail_global = (x2, (y1 + y2) // 2)

        nearest_head_node = find_nearest_node(head_global, node_boxes, grouped_boxes)
        nearest_tail_node = find_nearest_node(tail_global, node_boxes, grouped_boxes)

        if nearest_head_node and nearest_tail_node:
            arrow_connections.append((nearest_tail_node[1], nearest_head_node[1]))

    node_labels = {
        i: " ".join(data["Grouped_Texts"][i])
        for _, i in node_boxes
        if data["Grouped_Texts"][i]
    }

    generate_dot_file(output_dot_path, arrow_connections, node_labels)

    # Clean up temporary downloaded image
    try:
        os.remove(image_path)
    except OSError:
        pass


def process_all_jsons(json_folder, tsv_dict, output_dot_folder):
    """Process all merged JSON files in a folder.

    Args:
        json_folder: Directory containing merged JSON files.
        tsv_dict: Dict mapping image names to URLs.
        output_dot_folder: Directory to save DOT files.
    """
    os.makedirs(output_dot_folder, exist_ok=True)
    json_files = glob.glob(os.path.join(json_folder, "*.json"))

    for json_file in json_files:
        json_name = os.path.basename(json_file).replace(".json", "")
        if json_name in tsv_dict:
            image_url = tsv_dict[json_name]
            print(f"Processing {json_file}")
            process_json_file(json_file, image_url, output_dot_folder)
        else:
            print(f"No image URL found for {json_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert merged flowchart-OCR JSON files to DOT graph files."
    )
    parser.add_argument("--json_folder", required=True,
                        help="Path to the folder containing merged JSON files")
    parser.add_argument("--tsv_path", required=True,
                        help="Path to the TSV file mapping image names to URLs")
    parser.add_argument("--output_dot_folder", required=True,
                        help="Folder to save output DOT files")

    args = parser.parse_args()
    tsv_dict = load_tsv_to_dict(args.tsv_path)
    process_all_jsons(args.json_folder, tsv_dict, args.output_dot_folder)
