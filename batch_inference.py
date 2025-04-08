import argparse
import json
import os
import time
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run batched inference using VLLM")
    parser.add_argument("--output-file", type=str, default="results.json", help="Path to output JSON file for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="Number of GPUs to use for tensor parallelism")
    return parser.parse_args()


def load_prompts(input_file: str) -> List[Dict[str, Any]]:
    """Load prompts from input file."""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to output file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()

    model_directory = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    output_file = args.output_file
    tensor_parallel_size = args.tensor_parallel_size
    batch_size = args.batch_size
    max_tokens = args.max_tokens

    sample_prompts = [
    "Please provide a comprehensive analysis of the impact of artificial intelligence on healthcare over the next decade. Include potential benefits, challenges, ethical considerations, and how it might transform patient care, diagnostics, and medical research.",
    
    "Write a detailed essay exploring the history, current state, and future prospects of renewable energy technologies. Discuss solar, wind, hydroelectric, and emerging technologies, along with their economic viability and environmental impact.",
    
    "Explain in depth how large language models work, including their architecture, training process, capabilities, limitations, and ethical concerns. Use examples to illustrate key concepts and discuss potential future developments in this field.",
    
    "Provide a thorough comparison of different economic systems (capitalism, socialism, mixed economies) with historical examples of their implementation, successes, failures, and lessons learned. Include analysis of how these systems might evolve in response to global challenges.",
    
    "Write a detailed guide on building a scalable web application from scratch, covering architecture decisions, technology stack selection, development best practices, security considerations, deployment strategies, and maintenance approaches.",
    
    "Analyze the philosophical concept of consciousness from multiple perspectives, including neuroscience, philosophy of mind, artificial intelligence, and various cultural and religious traditions. Discuss the hard problem of consciousness and theories attempting to explain it.",
    
    "Present a comprehensive overview of climate change science, including evidence, causes, projected impacts, mitigation strategies, adaptation approaches, and the challenges of international cooperation. Include discussion of recent research and policy developments.",
    
    "Explore the evolution of storytelling across human history, from oral traditions to digital media. Analyze how narrative structures, themes, and techniques have changed or remained constant, and how storytelling functions in different cultures and contexts.",
    
    "Provide an in-depth analysis of global food security challenges, including agricultural systems, distribution networks, economic factors, climate impacts, and potential solutions. Discuss both technological and policy approaches to ensuring sustainable food systems.",
    
    "Write a detailed examination of how quantum computing works, its current state of development, potential applications, technical challenges, and how it might transform fields like cryptography, materials science, drug discovery, and artificial intelligence."
    ]

    print(f"Loading model: {model_directory}")
    llm = LLM(
        model=model_directory,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=430000,
        kv_cache_dtype="fp8",
    )
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    
    results = []
    total_prompts = len(sample_prompts)
    
    print(f"Running inference on {total_prompts} prompts with batch size {batch_size}")
    start_time = time.time()
    ids = [i for i in range(total_prompts)]
    for i in tqdm(range(0, total_prompts, batch_size)):
        batch = sample_prompts[i:i + batch_size]
        batch_prompts = batch
        batch_ids = ids[i:i + batch_size]
        
        outputs = llm.generate(batch_prompts, sampling_params)
        
        for output, prompt_id, original_item in zip(outputs, batch_ids, batch):
            generated_text = output.outputs[0].text
            
            result = {
                "id": prompt_id,
                "prompt": output.prompt,
                "generated_text": generated_text,
                "original_data": original_item,
            }
            results.append(result)
    
    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time / total_prompts:.4f} seconds")
    
    print(f"Saving results to: {output_file}")
    save_results(results, output_file)
    print("Done!")


if __name__ == "__main__":
    main()
