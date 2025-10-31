#!/usr/bin/env python3
"""
SLAKE-Specific MIA Conversation Generation

This script generates model responses for Membership Inference Attacks (MIA)
using YOUR specific LLaVA-Med checkpoint and model loading code.

Key adaptations for your setup:
1. Uses your exact model loading from downstream_inference_fixed.py
2. Handles FSDP checkpoints with low_cpu_mem_usage=True
3. Handles vision tower meta tensor reloading
4. Uses "simple" conversation mode (not "mistral_instruct")
5. Compatible with your SLAKE dataset format
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import sys

# Import from your LLaVA-Med codebase
from transformers import AutoTokenizer, AutoConfig
from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    """Patch config if needed (from your code)"""
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


class KeywordsStoppingCriteria(StoppingCriteria):
    """Stopping criteria from your code"""
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def load_model(model_name):
    """
    Load model using YOUR exact loading code from downstream_inference_fixed.py
    This ensures compatibility with your FSDP checkpoints.
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}\n")

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_config(model_name)

    print("Loading LLaVA model...")
    # Use YOUR exact loading code
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_cache=True,
        low_cpu_mem_usage=True,  # Critical for FSDP checkpoints
        device_map="auto"
    )

    print("Loading image processor...")
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower,
        torch_dtype=torch.float16
    )

    # Handle vision tower (YOUR code for meta tensor fix)
    print("Checking vision tower...")
    vision_tower = model.model.vision_tower[0]
    if hasattr(vision_tower, 'is_meta') or not next(vision_tower.parameters()).is_cuda:
        print("  Reloading vision tower from pretrained...")
        vision_tower = CLIPVisionModel.from_pretrained(
            model.config.mm_vision_tower,
            torch_dtype=torch.float16
        ).cuda()
        model.model.vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)

    # Setup tokens (YOUR code)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    model.eval()
    torch.set_grad_enabled(False)

    print(f"✅ Model loaded successfully!\n")

    return model, tokenizer, image_processor, image_token_len, mm_use_im_start_end


def main(args):
    # Load model
    model, tokenizer, image_processor, image_token_len, mm_use_im_start_end = load_model(
        os.path.expanduser(args.model_path)
    )

    # Load input data
    print(f"Loading data from: {args.input_json_path}")
    with open(args.input_json_path, 'r') as f:
        data = json.load(f)
    print(f"Total samples: {len(data)}\n")

    # Check for existing results (resume capability)
    if os.path.exists(args.output_json_path):
        with open(args.output_json_path, 'r') as f:
            results = json.load(f)
        completed_ids = {item['image_id'] for item in results}
        print(f"Found {len(completed_ids)} completed samples, resuming...")
    else:
        os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
        results = []
        completed_ids = set()

    # Process each sample
    errors = []
    for idx, item in enumerate(tqdm(data, desc="Generating conversations")):
        item_id = item.get('id', idx)

        if item_id in completed_ids:
            continue

        try:
            # Load image
            image_path = os.path.join(args.image_folder, item['image'])
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()

            # Prepare result structure
            conversation_result = {"image_id": item_id}

            # Generate for each temperature
            for temperature in args.temperatures:
                conversation_result[f"conversations_{temperature}"] = []

                # Process conversations
                try:
                    conversations = item.get("conversations", item.get("conversatons", []))
                except:
                    print(f"Warning: No conversations found for item {item_id}")
                    continue

                for conv_item in conversations:
                    if conv_item["from"] == "human":
                        # Extract question
                        question_text = conv_item["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()

                        conversation_result[f"conversations_{temperature}"].append({
                            "from": "human",
                            "value": question_text
                        })

                        # Generate responses (with repetitions for image-only attack)
                        for repeat_idx in range(args.repeat):
                            # Prepare prompt (YOUR code style)
                            qs = question_text
                            if mm_use_im_start_end:
                                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
                            else:
                                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

                            # Use YOUR conversation template
                            conv = conv_templates[args.conv_mode].copy()
                            conv.append_message(conv.roles[0], qs)
                            prompt = conv.get_prompt()

                            # Tokenize
                            inputs = tokenizer([prompt])
                            input_ids = torch.as_tensor(inputs.input_ids).cuda()

                            # Stopping criteria
                            keywords = ['###']
                            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                            # Generate (with temperature control for MIA)
                            with torch.inference_mode():
                                if temperature == 0 or not args.use_sampling:
                                    # Deterministic (temperature 0)
                                    output_ids = model.generate(
                                        input_ids,
                                        images=images,
                                        do_sample=False,
                                        temperature=None,
                                        max_new_tokens=args.max_new_tokens,
                                        stopping_criteria=[stopping_criteria]
                                    )
                                else:
                                    # Sampling with temperature
                                    output_ids = model.generate(
                                        input_ids,
                                        images=images,
                                        do_sample=True,
                                        temperature=temperature,
                                        max_new_tokens=args.max_new_tokens,
                                        stopping_criteria=[stopping_criteria]
                                    )

                            # Decode
                            input_token_len = input_ids.shape[1]
                            outputs = tokenizer.batch_decode(
                                output_ids[:, input_token_len:],
                                skip_special_tokens=True
                            )[0].strip()

                            # Post-process (YOUR code)
                            try:
                                index = outputs.index(conv.sep)
                                outputs = outputs[:index].strip()
                            except ValueError:
                                pass  # No separator found

                            # Save response
                            repeat_label = f"vlm_{repeat_idx + 1}" if args.repeat > 1 else "vlm"
                            conversation_result[f"conversations_{temperature}"].append({
                                "from": repeat_label,
                                "value": outputs
                            })

                    elif conv_item["from"] == "gpt":
                        # Save ground truth
                        conversation_result[f"conversations_{temperature}"].append({
                            "from": "ground truth",
                            "value": conv_item["value"]
                        })

            results.append(conversation_result)

            # Save checkpoint every 50 samples
            if (idx + 1) % 50 == 0:
                with open(args.output_json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Saved checkpoint at {idx + 1} samples")

        except Exception as e:
            error_msg = f"Error processing {item_id}: {str(e)}"
            print(f"\n❌ {error_msg}")
            errors.append({'image_id': item_id, 'error': error_msg})
            import traceback
            traceback.print_exc()

    # Final save
    with open(args.output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Conversation generation complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_json_path}")
    print(f"Total samples: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"{'='*80}\n")

    if errors:
        error_file = args.output_json_path.replace('.json', '_errors.json')
        with open(error_file, 'w') as f:
            json.dump({"errors": errors}, f, indent=2)
        print(f"⚠️  Errors saved to: {error_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIA conversations for SLAKE dataset")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to your fine-tuned checkpoint (e.g., /scratch1/runhuixu/outputs/test/checkpoint-110)")
    parser.add_argument("--input-json-path", type=str, required=True,
                        help="Path to input data (member or non-member JSON)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to SLAKE images folder")
    parser.add_argument("--output-json-path", type=str, required=True,
                        help="Path to save generated conversations")
    parser.add_argument("--conv-mode", type=str, default="simple",
                        help="Conversation template (use 'simple' for your model)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.1],
                        help="List of temperatures (e.g., 0.1 or 0.1 1.5)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of repetitions (>1 for image-only attack)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum response length")
    parser.add_argument("--use-sampling", action="store_true",
                        help="Enable sampling for non-zero temperatures")

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    if not os.path.exists(args.input_json_path):
        raise ValueError(f"Input JSON does not exist: {args.input_json_path}")
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder does not exist: {args.image_folder}")

    main(args)
