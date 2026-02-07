import os
import argparse
import logging

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path):
    """Load a pretrained instruction-tuned model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = torch.compile(model)
        model.to("cuda")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise


def load_examples_from_file(examples_file_path=None):
    """Load few-shot examples from a TSV file, or return built-in defaults.

    The TSV file may use pipe ('|') characters as newline separators within
    the output column for compact storage.
    """
    default_examples = [
        {
            "input": "A simple web application architecture with a frontend that connects to a backend API, which then connects to a database.",
            "output": '''digraph {
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
            "output": '''digraph {
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
            "output": '''digraph {
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
                output_code = row['output']
                if '|' in output_code:
                    output_code = output_code.replace('|', '\n').strip()
                    if output_code.startswith('"') and output_code.endswith('"'):
                        output_code = output_code[1:-1]
                examples.append({
                    "input": row['input'],
                    "output": output_code
                })
            logger.info(f"Loaded {len(examples)} examples from {examples_file_path}")
            return examples
        except Exception as e:
            logger.warning(f"Failed to load examples from {examples_file_path}: {e}")
            logger.info("Using default examples instead")

    return default_examples


def prepare_prompt(description, tokenizer, examples_file_path=None):
    """Build a few-shot prompt using the model's native chat template."""
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
                "1. Use 'digraph {' as the graph declaration\n"
                "2. Set appropriate rankdir (TB for top-bottom, LR for left-right) if needed\n"
                "3. Use appropriate node shapes (box is default)\n"
                "4. Create meaningful node labels\n"
                "5. Add edge labels where appropriate to describe relationships\n"
                "6. Keep the graph structure clear and readable\n"
                "7. IMPORTANT: Respond with ONLY the DOT code, no explanations or additional text\n\n"
                "Here are examples of how to convert descriptions to DOT graphs:"
                + few_shot_examples
            )
        },
        {
            "role": "user",
            "content": (
                "Convert the following description into DOT language code. "
                f"Respond with ONLY the DOT code and nothing else:\n\n{description}"
            )
        },
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt


def extract_dot_code(response):
    """Extract DOT graph code from a model response.

    Parses the response to find a complete digraph block by tracking
    brace matching. Falls back to the full response if no digraph
    structure is found.
    """
    lines = response.split('\n')
    dot_lines = []
    in_digraph = False
    brace_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('digraph'):
            in_digraph = True
            dot_lines.append(stripped)
            brace_count += stripped.count('{') - stripped.count('}')
        elif in_digraph:
            dot_lines.append(stripped)
            brace_count += stripped.count('{') - stripped.count('}')
            if brace_count <= 0:
                break

    if dot_lines:
        return '\n'.join(dot_lines)

    logger.warning("Could not extract DOT structure from response, returning full text")
    return response


def generate_response(model, tokenizer, prompt, max_length=4096):
    """Generate DOT code from a prompt and extract the digraph block."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return extract_dot_code(response).strip()

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return f"GENERATION_ERROR: {str(e)}"


def save_results_to_tsv(results, output_file):
    """Save inference results to a TSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Results saved to {output_file}")


def main(args):
    """Run inference on test data and save results."""
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    test_df = pd.read_csv(args.test_file, sep='\t', dtype=str)
    logger.info(f"Loaded {len(test_df)} test samples")

    results = []
    for idx, row in test_df.iterrows():
        description = row['Cleaned Description']
        prompt = prepare_prompt(description, tokenizer, args.examples_file)
        response = generate_response(model, tokenizer, prompt)

        results.append({
            'Cleaned Description': description,
            'Generated DOT': response,
            'Ground Truth DOT': row['Dot code'],
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_df)} samples")
            save_results_to_tsv(results, args.output_file)

    save_results_to_tsv(results, args.output_file)
    logger.info(f"Final results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few-shot inference with an instruction-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the instruction-tuned model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.tsv")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results TSV")
    parser.add_argument("--examples_file", type=str, default=None, help="Path to few-shot examples TSV")

    args = parser.parse_args()
    main(args)
