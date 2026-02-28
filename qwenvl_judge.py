import argparse
import base64
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import torch
import warnings
from packaging import version
import transformers
try:
    current_version = transformers.__version__
    if version.parse(current_version) != version.parse("4.57.0"):
        warnings.warn(f"transformers version {current_version} detected, but 4.57.0 is recommended", UserWarning)
except Exception as e:
    warnings.warn(f"Could not check transformers version: {e}", UserWarning)

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from judge import encode_image, create_evaluation_prompts, get_all_numbers, IMAGE_EXTENSIONS


class QwenVLModel:
    def __init__(self):
        model_name = "Qwen/Qwen3-VL-8B-Instruct"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def inference(self, messages) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text



def find_reference_images(directory, number):
    ref_images = []
    i = 0
    while True:
        found = False
        for ext in IMAGE_EXTENSIONS:
            file_path = directory / f"{number}_{i}{ext}"
            if file_path.exists():
                ref_images.append(file_path)
                found = True
                break
        if not found:
            break
        i += 1

    return sorted(ref_images)


def find_prompt_file(directory, number):
    prompt_path = directory / f"{number}_prompt.txt"
    return prompt_path if prompt_path.exists() else None


def find_generated_image(directory, number):
    for ext in IMAGE_EXTENSIONS:
        file_path = directory / f"{number}_generated{ext}"
        if file_path.exists():
            return file_path
    return None


def judge_image_with_qwenvl(
    qwenvl,
    directory,
    generated_image_path,
    ref_image_paths,
    prompt_path,
    number,
):
    try:
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()
        else:
            instruction = "No instruction file found"

        generated_image_base64 = encode_image(generated_image_path)

        prompt_a, prompt_b, prompt_c = create_evaluation_prompts(instruction)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_a}]}]

        for ref_path in ref_image_paths:
            ref_base64 = encode_image(ref_path)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ref_base64}"},
                }
            )

        messages[0]["content"].append({"type": "text", "text": prompt_b})

        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{generated_image_base64}"
                },
            }
        )

        messages[0]["content"].append({"type": "text", "text": prompt_c})

        result_text = qwenvl.inference(messages)

        return result_text

    except Exception as e:
        return f"Error during evaluation: {str(e)}"


def process_image(qwenvl, directory, number, task_name, model="qwenvl"):
    try:
        output_path = directory / f"{number}_{model}_judge.txt"

        generated_image_path = find_generated_image(directory, number)
        if not generated_image_path:
            raise FileNotFoundError("Generated image not found")

        ref_image_paths = find_reference_images(directory, number)
        if not ref_image_paths:
            raise FileNotFoundError("Reference images not found")

        prompt_path = find_prompt_file(directory, number)
        if not prompt_path:
            raise FileNotFoundError("Prompt file not found")

        result = judge_image_with_qwenvl(
            qwenvl, directory, generated_image_path, ref_image_paths, prompt_path, number
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)

        return {"success": True, "task": task_name, "number": number}

    except Exception as e:
        return {"success": False, "task": task_name, "number": number, "error": str(e)}


def collect_all_tasks(model):
    if not BASE_DIR.exists():
        return []

    all_tasks = []
    for task_dir in sorted(BASE_DIR.iterdir()):
        if task_dir.is_dir() and not task_dir.name.startswith("."):
            for number in get_all_numbers(task_dir):
                judge_file = task_dir / f"{number}_{model}_judge.txt"
                if not judge_file.exists():
                    all_tasks.append(
                        {
                            "directory": task_dir,
                            "number": number,
                            "task_name": task_dir.name,
                        }
                    )
    return all_tasks


def extract_scores(content):
    patterns = {
        "Instruction Alignment": r"Instruction Alignment:\s*(\d+)",
        "Reference Consistency": r"Reference Consistency:\s*(\d+)",
        "Background-Subject Match": r"Background-Subject Match:\s*(\d+)",
        "Physical Realism": r"Physical Realism:\s*(\d+)",
        "Visual Quality": r"Visual Quality:\s*(\d+)",
    }
    scores = {}
    for metric, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            scores[metric] = int(match.group(1))
    return scores if len(scores) == 5 else None


def extract_results(base_dir, model, output_dir):
    results = defaultdict(
        lambda: {
            "Instruction Alignment": [],
            "Reference Consistency": [],
            "Background-Subject Match": [],
            "Physical Realism": [],
            "Visual Quality": [],
        }
    )

    for filepath in base_dir.rglob(f"*_{model}_judge.txt"):
        task = filepath.parent.name
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            scores = extract_scores(content)
            if scores:
                for metric, value in scores.items():
                    results[task][metric].append(value)
        except Exception:
            continue

    summary = []
    for task in sorted(results.keys()):
        metrics = results[task]
        if not metrics["Instruction Alignment"]:
            continue
        row = {
            "Task": task,
            "Samples": len(metrics["Instruction Alignment"]),
            "Instruction_Alignment": sum(metrics["Instruction Alignment"])
            / len(metrics["Instruction Alignment"]),
            "Reference_Consistency": sum(metrics["Reference Consistency"])
            / len(metrics["Reference Consistency"]),
            "Background_Subject_Match": sum(metrics["Background-Subject Match"])
            / len(metrics["Background-Subject Match"]),
            "Physical_Realism": sum(metrics["Physical Realism"])
            / len(metrics["Physical Realism"]),
            "Visual_Quality": sum(metrics["Visual Quality"])
            / len(metrics["Visual Quality"]),
        }
        row["Average"] = (
            row["Instruction_Alignment"]
            + row["Reference_Consistency"]
            + row["Background_Subject_Match"]
            + row["Physical_Realism"]
            + row["Visual_Quality"]
        ) / 5
        summary.append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv = output_path / f"results_{model}.csv"
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(
            "Task,Samples,Instruction_Alignment,Reference_Consistency,Background_Subject_Match,Physical_Realism,Visual_Quality,Average\n"
        )
        for row in summary:
            f.write(
                f"{row['Task']},{row['Samples']},"
                f"{row['Instruction_Alignment']:.3f},{row['Reference_Consistency']:.3f},"
                f"{row['Background_Subject_Match']:.3f},{row['Physical_Realism']:.3f},"
                f"{row['Visual_Quality']:.3f},{row['Average']:.3f}\n"
            )

    output_json = output_path / f"results_{model}.json"
    json_results = {}
    for row in summary:
        json_results[row["Task"]] = {
            "samples": row["Samples"],
            "Instruction_Alignment": round(row["Instruction_Alignment"], 3),
            "Reference_Consistency": round(row["Reference_Consistency"], 3),
            "Background_Subject_Match": round(row["Background_Subject_Match"], 3),
            "Physical_Realism": round(row["Physical_Realism"], 3),
            "Visual_Quality": round(row["Visual_Quality"], 3),
            "Average": round(row["Average"], 3),
        }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(summary)} tasks, saved to {output_csv} and {output_json}")


def main(output_dir=None):
    model = "qwenvl"

    all_tasks = collect_all_tasks(model)
    if not all_tasks:
        print("No tasks to judge. All tasks already have judge files.")
        if output_dir:
            print(f"\nExtracting results...")
            extract_results(BASE_DIR, model, output_dir)
        return

    print(f"Found {len(all_tasks)} tasks to judge.")
    start_time = time.time()
    results = []

    qwenvl = QwenVLModel()

    for task in tqdm(all_tasks):
        result = process_image(qwenvl, task["directory"], task["number"], task["task_name"], model)
        results.append(result)

    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\nCompleted: {successful}/{len(all_tasks)} successful, {failed} failed")
    print(
        f"Time: {elapsed_time:.2f}s (avg: {elapsed_time / len(all_tasks):.2f}s per image)"
    )

    if output_dir:
        print(f"\nExtracting results...")
        extract_results(BASE_DIR, model, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge by Qwen3-VL for MultiBananaBenchmark"
    )
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Base directory of the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for CSV and JSON files",
    )
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    main(output_dir=args.output_dir)

