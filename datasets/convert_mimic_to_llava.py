import os
import json
import ast
import argparse
import random
from typing import List
from collections import Counter
import statistics


prompts = [
     "This is a chest X-ray image. Please generate a detailed radiology findings report describing any observed abnormalities or normal structures.",
    # "Analyze the following chest X-ray and produce a professional findings report summarizing key observations.",
    # "Generate a concise radiology-style findings report for this chest X-ray image, noting both normal and abnormal features.",
    # "Please review this chest X-ray and provide a structured findings report as a radiologist would.",
    # "From this chest X-ray, draft a clinical findings report describing the visible anatomical details and any abnormalities.",
    # "Examine this chest X-ray image and write a formal findings section similar to what appears in a radiology report.",
    # "Create a descriptive findings report for this chest X-ray, identifying any notable pathologies or confirming normal appearance.",
    # "Based on the provided chest X-ray, generate a findings paragraph outlining radiological observations.",
    # "Interpret this chest X-ray and summarize your findings in a standard medical imaging report format.",
    # "Please describe the radiologic findings visible in this chest X-ray image in a professional, report-style summary.",
    # "Review this chest X-ray image and compose a findings report that highlights relevant diagnostic details."
]


def to_list(x) -> List[str]:
    """Normalize various image path field formats to a list of strings.
    Supports: list/tuple; plain string; stringified list like "['a.jpg','b.jpg']".
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(p).strip() for p in x if str(p).strip()]
    s = str(x).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, (list, tuple)):
                return [str(p).strip() for p in lst if str(p).strip()]
        except Exception:
            pass
    return [s]


def build_sample(image_paths: List[str], report_text: str) -> dict:
    prompt = random.choice(prompts)
    if len(image_paths) == 0:
        return None
    if len(image_paths) == 1:
        return {
            "image": image_paths[0],
            "conversations": [
                {"from": "human", "value": "<image>\n" + prompt},
                {"from": "gpt", "value": str(report_text).strip()},
            ],
        }
    return {
        "images": image_paths,
        "conversations": [
            {"from": "human", "value": "\n".join(["<image>"] * len(image_paths)) + "\n" + prompt},
            {"from": "gpt", "value": str(report_text).strip()},
        ],
    }


def process_split(records: List[dict], out_path: str) -> None:
    total = 0
    drop_no_report = 0
    drop_no_image = 0
    output = []
    num_images_list = []

    for r in records:
        total += 1
        report_text = r.get("report") or r.get("report_text")
        if report_text is None or str(report_text).strip() == "":
            drop_no_report += 1
            continue

        image_field = r.get("image_path")
        if image_field is None:
            image_field = r.get("image_paths")
        if image_field is None:
            image_field = r.get("images")

        paths = to_list(image_field)
        if len(paths) == 0:
            drop_no_image += 1
            continue

        sample = build_sample(paths, report_text)
        if sample:
            # Preserve original identifiers from the input record if present
            for keep_key in ("id", "study_id", "subject_id"):
                if keep_key in r:
                    sample[keep_key] = r[keep_key]
            output.append(sample)
            num_images_list.append(len(paths))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(
        f"Saved {len(output)} / {total} -> {out_path} | "
        f"dropped(no_report)={drop_no_report}, dropped(no_image)={drop_no_image}"
    )

    if len(num_images_list) > 0:
        cnt = Counter(num_images_list)
        num_saved = len(num_images_list)
        min_n = min(num_images_list)
        max_n = max(num_images_list)
        mean_n = sum(num_images_list) / num_saved
        median_n = statistics.median(num_images_list)
        multi_ratio = sum(1 for n in num_images_list if n > 1) / num_saved

        print(f"Image-per-sample stats for {out_path}:")
        print(f"  count={num_saved}, min={min_n}, max={max_n}, mean={mean_n:.3f}, median={median_n}")
        print(f"  multi-image ratio(>1)={multi_ratio:.3%}")
        for k in sorted(cnt.keys()):
            print(f"    images={k}: {cnt[k]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/jhcnas5/chenzhixuan/data/mimic_annotations.json")
    parser.add_argument("--out_root", type=str, default="/jhcnas5/chenzhixuan/checkpoints/VIRAL/mimic_findings")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If top-level is a dict with splits
    if isinstance(data, dict):
        # normalize common split keys
        split_map = {
            "train": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "test": "test",
        }
        found = False
        for k, v in list(data.items()):
            if isinstance(v, list) and k.lower() in split_map:
                found = True
        if found:
            for k, v in data.items():
                if not isinstance(v, list):
                    continue
                key_norm = k.lower()
                if key_norm not in split_map:
                    continue
                tag = split_map[key_norm]
                out_path = f"{args.out_root}_{tag}.json"
                process_split(v, out_path)
            return

    # Otherwise assume it's a flat list
    if isinstance(data, list):
        process_split(data, f"{args.out_root}.json")
        return

    raise ValueError("Unsupported input JSON structure. Expect dict with splits or list of records.")


if __name__ == "__main__":
    main()
