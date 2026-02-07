import argparse

import pandas as pd
import torch
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "DiagramAgent/Diagram_to_Code_Agent"
CHAT_TEMPLATE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PROCESSOR_SIZE = {"shortest_edge": 3136, "longest_edge": 12845056}


def load_model_and_processors(device: str):
    """Load the DiagramAgent model and processors."""
    print("Loading model and processors...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        size=PROCESSOR_SIZE,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    chat_processor = AutoProcessor.from_pretrained(
        CHAT_TEMPLATE_MODEL,
        size=PROCESSOR_SIZE,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return model, processor, chat_processor


def generate_dot_code(model, processor, chat_processor, image_url: str, device: str) -> str:
    """Generate DOT code for a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Generate a dot code for the following image"},
            ],
        }
    ]

    text = chat_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def process_images(input_file: str, output_file: str, hf_key: str):
    """Process all images in the input TSV and write results to output TSV."""
    login(hf_key)
    df = pd.read_csv(input_file, sep="\t")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, chat_processor = load_model_and_processors(device)

    # Write header
    columns = list(df.columns) + ["predicted_code"]
    with open(output_file, "w") as f:
        f.write("\t".join(columns) + "\n")

    # Process each image
    print(f"Processing {len(df)} images...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            output_text = generate_dot_code(
                model, processor, chat_processor, row["Image_URL"], device
            )
        except Exception as e:
            print(f"Error processing image {i + 1}/{len(df)}: {e}")
            output_text = "ERROR: " + str(e)

        # Append result to output file
        with open(output_file, "a") as f:
            row_values = [str(row[col]) for col in df.columns]
            row_values.append(output_text.replace("\n", "\\n").replace("\t", "\\t"))
            f.write("\t".join(row_values) + "\n")
            f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Generate DOT code from diagram images using DiagramAgent."
    )
    parser.add_argument("--input", type=str, required=True, help="Input TSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output TSV file path")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    args = parser.parse_args()

    process_images(args.input, args.output, args.hf_key)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
