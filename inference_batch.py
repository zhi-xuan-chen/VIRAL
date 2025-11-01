import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import json
import csv
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/jhcnas5/chenzhixuan/checkpoints/VIRAL/outputs/llava-v1.5-7b-instruct-llava-mimic_new/checkpoint-9000")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--image-folder", type=str, default="/jhcnas4/kyle/Xray/DATA/MIMIC-CXR/files", help="图像文件夹路径")
    parser.add_argument("--json-path", type=str, default="/jhcnas5/chenzhixuan/checkpoints/VIRAL/mimic_findings_test.json", help="包含批量样本的 JSON 文件路径（列表，每项含 image_path 与可选 prompt）")
    parser.add_argument("--output-csv", type=str, default="/home/chenzhixuan/Workspace/VIRAL/results/mimic_llava_outputs.csv", help="批量推理结果输出的 CSV 路径")
    parser.add_argument("--prompt", type=str,
                        default="This is a chest X-ray image. Please generate a detailed radiology findings report describing any observed abnormalities or normal structures.",
                        help="当样本缺失 prompt 时的默认提示")
    parser.add_argument("--batch-size", type=int, default=8, help="推理的批大小")
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    is_7b = "7b" in args.model_path.lower()
    model_base = "lmsys/vicuna-7b-v1.5" if is_7b else "lmsys/vicuna-13b-v1.5"
    model_name = "llava-v1.5-7b-lora" if is_7b else "llava-v1.5-13b-lora"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=model_base,
        model_name=model_name
    )
    model.to(device=device, dtype=dtype).eval()

    for attr in ("vra_loss", "residual", "residual_target_layers"):
        if hasattr(model, attr):
            if attr == "residual_target_layers":
                setattr(model, attr, [16])  # 필요 없다면 주석 처리 가능
            else:
                setattr(model, attr, False)

    # 仅支持 JSON 批量推理并输出 CSV（移除单张推理分支）
    if args.json_path is None:
        raise ValueError("请通过 --json-path 指定批量 JSON 文件")

    with open(args.json_path, "r") as f:
        data_list = json.load(f)
    if not isinstance(data_list, list):
        raise ValueError("JSON 格式需为列表：[{...}, {...}, ...]")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    written = 0
    skipped = 0

    def to_prompt(item, default_prompt):
        convs = item.get("conversations")
        if isinstance(convs, list):
            for m in convs:
                if m.get("from") == "human" and isinstance(m.get("value"), str):
                    val = m.get("value")
                    if isinstance(val, str):
                        val = val.replace("<image>", "").lstrip("\n").strip()
                    return val if val else default_prompt
        return (
            item.get("prompt")
            or item.get("instruction")
            or item.get("question")
            or default_prompt
        )

    def to_image_path(item):
        p = (
            item.get("image_path")
            or item.get("image")
            or item.get("img_path")
            or item.get("path")
        )
        if not p:
            return None
        # 若为相对路径，则拼接到根目录 args.image_folder
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(args.image_folder, p))
        return p

    def to_gold_report(item):
        convs = item.get("conversations")
        if isinstance(convs, list):
            for m in convs:
                if m.get("from") == "gpt" and isinstance(m.get("value"), str):
                    return m.get("value")
        return item.get("report")

    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image_path",
            "id",
            "study_id",
            "subject_id",
            "prompt",
            "response",
            "gold_report"
        ])  # 表头

        class JSONDataset(Dataset):
            def __init__(self, items, default_prompt):
                self.items = items
                self.default_prompt = default_prompt

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                item = self.items[index]
                image_path = to_image_path(item)
                if not image_path or not os.path.exists(image_path):
                    return None  # 交由 collate 过滤
                prompt = to_prompt(item, self.default_prompt)
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    return None
                return {
                    "image_path": image_path,
                    "prompt": prompt,
                    "image": image,
                    "report": to_gold_report(item),
                    "id": item.get("id"),
                    "study_id": item.get("study_id"),
                    "subject_id": item.get("subject_id")
                }

        def collate_fn(batch):
            valid = [b for b in batch if b is not None]
            nonlocal skipped
            skipped += (len(batch) - len(valid))
            if len(valid) == 0:
                return None

            batch_images = [b["image"] for b in valid]
            batch_items = [(b["image_path"], b["prompt"]) for b in valid]

            batch_convs = []
            batch_prompts = []
            for _, prompt in batch_items:
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
                conv.append_message(conv.roles[1], None)
                batch_convs.append(conv)
                batch_prompts.append(conv.get_prompt())

            pixel_values = process_images(batch_images, image_processor, model.config)
            if isinstance(pixel_values, list):
                pixel_values = torch.stack(pixel_values, dim=0)

            ids_list = [
                tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                for p in batch_prompts
            ]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            max_len = max(t.size(0) for t in ids_list)
            input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long, device="cpu")
            attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long, device="cpu")
            for i, t in enumerate(ids_list):
                cur_len = t.size(0)
                input_ids[i, :cur_len] = t.to("cpu")
                attention_mask[i, :cur_len] = 1

            meta = {
                "items": batch_items,
                "convs": batch_convs,
                "reports": [b.get("report") for b in valid],
                "ids": [b.get("id") for b in valid],
                "study_ids": [b.get("study_id") for b in valid],
                "subject_ids": [b.get("subject_id") for b in valid]
            }
            return input_ids, attention_mask, pixel_values, meta

        dataset = JSONDataset(data_list, args.prompt)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda")
        )

        total_samples = len(dataset)
        processed_so_far = 0
        print(f"开始推理：共 {total_samples} 条样本，batch_size={args.batch_size}")

        for batch in tqdm(loader):
            if batch is None:
                continue
            input_ids, attention_mask, pixel_values, meta = batch

            # 将张量移动到目标设备
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device=device, dtype=dtype)

            with torch.inference_mode():
                outputs = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )

            sequences = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
            for (image_path, prompt), conv, seq, gold, _id, sid, subid in zip(
                meta["items"], meta["convs"], sequences, meta["reports"], meta["ids"], meta["study_ids"], meta["subject_ids"]
            ):
                output_text = tokenizer.decode(seq, skip_special_tokens=True)
                if conv.sep_style == SeparatorStyle.TWO:
                    response = output_text.split(conv.sep2)[-1].strip()
                elif conv.sep_style == SeparatorStyle.LLAMA_2:
                    response = output_text.split(conv.roles[1] + ":")[-1].strip()
                else:
                    response = output_text.split(conv.roles[1] + ":")[-1].strip()
                
                writer.writerow([image_path, _id, sid, subid, prompt, response, gold])
                written += 1

            processed_so_far += len(meta["items"])
            pct = (processed_so_far / total_samples * 100.0) if total_samples > 0 else 100.0
            print(f"进度：{processed_so_far}/{total_samples} ({pct:.1f}%)，已写入 {written}，跳过 {skipped}")

    print(f"完成批量推理（按 batch_size={args.batch_size}）：写入 {written} 条，跳过 {skipped} 条。CSV 输出：{args.output_csv}")


if __name__ == "__main__":
    main()