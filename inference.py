import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/jhcnas5/chenzhixuan/checkpoints/VIRAL/outputs/llava-v1.5-7b-instruct-viral-mimic/checkpoint-9000")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--image-path", type=str, default="/jhcnas4/kyle/Xray/DATA/MIMIC-CXR/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg")
    parser.add_argument("--prompt", type=str,
                        default="This is a chest X-ray image. Please generate a detailed radiology findings report describing any observed abnormalities or normal structures.")
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

    image = Image.open(args.image_path).convert("RGB")
    if hasattr(image_processor, "preprocess"):
        pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    else:
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs=input_ids,
            images=pixel_values,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    # 兼容返回 Tensor（老默认）或包含 sequences 的对象（return_dict_in_generate=True）
    sequences = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
    output_text = tokenizer.decode(sequences[0], skip_special_tokens=True)

    if conv.sep_style == SeparatorStyle.TWO:
        response = output_text.split(conv.sep2)[-1].strip()
    elif conv.sep_style == SeparatorStyle.LLAMA_2:
        response = output_text.split(conv.roles[1] + ":")[-1].strip()
    else:
        response = output_text.split(conv.roles[1] + ":")[-1].strip()

    print("Input:", args.prompt)
    print("\nResponse:", response)


if __name__ == "__main__":
    main()