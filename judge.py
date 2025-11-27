import argparse
import base64
import glob
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_evaluation_prompts(instruction):
    prompt_a = f"""You are a strict data rater specializing in grading multi-reference drien image generation. 
You will be given reference images, task instruction, and the generation results. 
Reference Images:"""

    prompt_b = f"""Editing Instruction: {instruction}
Final Output:"""

    prompt_c = """
Your task is to evaluate the effectiveness of replacement editing from five independent perspectives, each on a 10-point scale.
Note that the average score should be considered 4 points.

1. Text-Instruction Alignment

Evaluate whether the generated image accurately follows the given text instruction.
Check whether the specified objects appear in the correct positions, whether the instructed subjects are depicted properly, and whether no unintended elements are introduced.
For example, if the instruction says "change the language," but the actual written content itself is altered incorrectly, or if unnecessary objects are added, the score should be reduced. 
If the instruction requires including a reference subject but the generated image fails to include that referenced content, the score must be 1.
Even if the instruction is followed correctly, the score must not exceed 6 points if the generated image still exhibits any composited or unnatural appearance.

2. Reference Consistency

Evaluate how consistent the generated image is with the provided reference images.
Compare the output to each reference and assess how faithfully the structure and attributes are reproduced.
Fine details, such as hair ornaments, patterns on clothing, and other small features, must match the references, otherwise the score must not exceed 4 points.
If even a single object fails to follow the details of the reference images, the score must not exceed 6 points.

3. Background-Subject Match

Evaluate whether the subject blends naturally with the background.
Check whether the subject appears to be floating, unnaturally pasted on, or visually inconsistent with its surroundings.
Images that look like multiple pictures simply pasted together should receive a score of 1.
If there is even the slightest inconsistency in style, tone, lighting, or overall visual impression compared to the reference images, the score must also not exceed 4 points.

4. Physical Realism

Evaluate whether the generated image maintains physical plausibility.
Penalize cases where the image violates basic physical laws—for example, a person floating in mid-air, standing on water, or having the lower body missing despite no obstruction.
If there is even a slight impression that the image looks composited or artificially pasted together, the score must not exceed 4 points.
Likewise, if it is unclear whether the subject is actually making proper contact with the ground, the score must also not exceed 6 points.

5. Visual Quality

Evaluate the overall perceptual quality of the image.
Assess whether the image is visually appealing and aesthetically coherent.
If the composition appears unnatural or the image does not look aesthetically pleasing to a human observer, the score must not exceed 4 points.

Each of the five scores must be evaluated independently. Do not force any score to be tied to or capped by another score.

First explain the reasoning, then present the final assessment. Start the reasoning with 'Reasoning: '.
After explaining the reasoning, present the final assessment with format below.

Format:
Instruction Alignment: <A number from 1 to 10>.
Reference Consistency: <A number from 1 to 10>.
Background-Subject Match: <A number from 1 to 10>.
Physical Realism: <A number from 1 to 10>.
Visual Quality: <A number from 1 to 10>.
"""

    return prompt_a, prompt_b, prompt_c


def find_reference_images(directory, number):
    ref_images = []
    i = 1
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


def judge_image_with_gemini(
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

        ref_images = [Image.open(p) for p in ref_image_paths]

        generated_image = Image.open(generated_image_path)

        prompt_a, prompt_b, prompt_c = create_evaluation_prompts(instruction)

        contents = [prompt_a] + ref_images + [prompt_b, generated_image, prompt_c]

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        result_text = ""
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                result_text += part.text

        return result_text

    except Exception as e:
        return f"Error during evaluation: {str(e)}"


def judge_image_with_gpt(
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

        response = openai_client.chat.completions.create(
            model="gpt-5", messages=messages, max_completion_tokens=8192
        )

        result_text = response.choices[0].message.content

        return result_text

    except Exception as e:
        return f"Error during evaluation: {str(e)}"


def get_all_numbers(directory):
    numbers = set()
    for ext in IMAGE_EXTENSIONS:
        for file_path in directory.glob(f"*_generated{ext}"):
            number = file_path.stem.replace("_generated", "")
            numbers.add(number)
    return sorted(numbers)


def process_image(directory, number, task_name, model="gemini"):
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

        if model == "gemini":
            if not gemini_client:
                raise ValueError("GEMINI_API_KEY is not set")
            result = judge_image_with_gemini(
                directory, generated_image_path, ref_image_paths, prompt_path, number
            )
        elif model == "gpt":
            if not openai_client:
                raise ValueError("OPENAI_API_KEY is not set")
            result = judge_image_with_gpt(
                directory, generated_image_path, ref_image_paths, prompt_path, number
            )
        else:
            raise ValueError(f"Unknown model: {model}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)

        time.sleep(0.5)
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


def main(batch_size=32, model="gemini", output_dir=None):
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

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(
                process_image,
                task["directory"],
                task["number"],
                task["task_name"],
                model,
            ): task
            for task in all_tasks
        }

        with tqdm(total=len(all_tasks), desc=f"{model.upper()} Evaluation") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

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
        description="LLM-as-a-Judge for MultiBananaBenchmark"
    )
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Base directory of the dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gemini", "gpt"],
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of parallel workers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for CSV and JSON files",
    )
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    main(batch_size=args.batch_size, model=args.model, output_dir=args.output_dir)
