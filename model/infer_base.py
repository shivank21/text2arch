import os
import argparse
import logging

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path):
    """Load a pretrained model and tokenizer for inference."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model = torch.compile(model)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise


def load_examples_from_file(examples_file_path=None):
    """Load few-shot examples from a TSV file, or return built-in defaults."""
    default_examples = [
        {
            "input": "A simple web application architecture with a frontend that connects to a backend API, which then connects to a database.",
            "output": '''digraph{
    rankdir=TB;
    node [shape=box];

    Frontend [label="Frontend"];
    Backend [label="Backend API"];
    Database [label="Database"];

    Frontend -> Backend [label="HTTP requests"];
    Backend -> Database [label="queries"];
}'''
        },
        {
            "input": "A microservices architecture where a load balancer distributes requests to multiple service instances, each connecting to its own database.",
            "output": '''digraph{
    rankdir=TB;
    node [shape=box];

    LoadBalancer [label="Load Balancer"];
    Service1 [label="Service 1"];
    Service2 [label="Service 2"];
    Service3 [label="Service 3"];
    DB1 [label="Database 1"];
    DB2 [label="Database 2"];
    DB3 [label="Database 3"];

    LoadBalancer -> Service1;
    LoadBalancer -> Service2;
    LoadBalancer -> Service3;
    Service1 -> DB1;
    Service2 -> DB2;
    Service3 -> DB3;
}'''
        },
        {
            "input": "A data processing pipeline where data flows from a source system through a processing engine and then to a destination system.",
            "output": '''digraph{
    rankdir=LR;
    node [shape=box];

    Source [label="Source System"];
    Processor [label="Processing Engine"];
    Destination [label="Destination System"];

    Source -> Processor [label="raw data"];
    Processor -> Destination [label="processed data"];
}'''
        }
    ]

    if examples_file_path and os.path.exists(examples_file_path):
        try:
            examples_df = pd.read_csv(examples_file_path, sep='\t')
            examples = []
            for _, row in examples_df.iterrows():
                examples.append({
                    "input": row['input'],
                    "output": row['output']
                })
            logger.info(f"Loaded {len(examples)} examples from {examples_file_path}")
            return examples
        except Exception as e:
            logger.warning(f"Failed to load examples from {examples_file_path}: {e}")
            logger.info("Using default examples instead")

    return default_examples


def prepare_prompt(description, tokenizer, examples_file_path=None):
    """Build a few-shot prompt for the base model using a chat template."""
    template_str = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'system' %}## System:{{- message['content'].strip() }}\n"
        "{%- elif message['role'] == 'user' %}## Instruction: {{- message['content'].strip() }}\n"
        "{%- elif message['role'] == 'assistant' %}## Response: {{- message['content'].strip() }}\n"
        "{%- endif %}"
        "{%- endfor %}"
    )
    tokenizer.chat_template = template_str

    examples = load_examples_from_file(examples_file_path)

    few_shot_examples = ""
    for i, example in enumerate(examples, 1):
        few_shot_examples += f"\n\nExample {i}:\n"
        few_shot_examples += f"Input: {example['input']}\n"
        few_shot_examples += f"Output:\n{example['output']}\n"

    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert in analyzing technical descriptions of system architecture, "
                "workflows, and process pipelines, and a code design specialist skilled in "
                "graph visualization using DOT language.\n\n"
                "Your task is to convert technical descriptions into DOT graph representations. "
                "Follow these guidelines:\n"
                "1. Use 'digraph G' as the graph declaration\n"
                "2. Set appropriate rankdir (TB for top-bottom, LR for left-right)\n"
                "3. Use box shapes for nodes unless specified otherwise\n"
                "4. Create meaningful node labels\n"
                "5. Add edge labels where appropriate to describe relationships\n"
                "6. Keep the graph structure clear and readable\n\n"
                "Here are some examples of how to convert descriptions to DOT graphs:"
                + few_shot_examples
            )
        },
        {
            "role": "user",
            "content": f"Now convert the following description to DOT graph format:\n\nInput: {description}\n\nOutput:"
        },
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt


def generate_response(model, tokenizer, prompt, max_length=1024):
    """Generate DOT code from a prompt using the model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("## Response:")[-1].strip()
        return response

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return "ERROR: Failed to generate response"


def main(args):
    """Run inference on test data and save results."""
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    test_df = pd.read_csv(args.test_file, sep='\t', dtype=str)

    results = []
    for idx, row in test_df.iterrows():
        try:
            description = row['Cleaned Description']
            prompt = prepare_prompt(description, tokenizer, args.examples_file)
            response = generate_response(model, tokenizer, prompt)

            results.append({
                'Description': description,
                'Generated DOT': response,
                'Ground Truth DOT': row['Dot code']
            })
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            results.append({
                'Description': row.get('Cleaned Description', 'ERROR'),
                'Generated DOT': f"ERROR: {str(e)}",
                'Ground Truth DOT': row.get('Dot code', 'N/A')
            })

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_df)} samples")
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.output_file, sep='\t', index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, sep='\t', index=False)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few-shot inference with a base model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.tsv")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results TSV")
    parser.add_argument("--examples_file", type=str, default=None, help="Path to few-shot examples TSV")

    args = parser.parse_args()
    main(args)
