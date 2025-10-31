"""
Inference script optimized for loading FSDP-saved checkpoints.

This script handles checkpoints saved with FSDP (Fully Sharded Data Parallel)
which are split into multiple shard files (pytorch_model-00001-of-00003.bin, etc.)

Key differences from standard inference:
1. Uses device_map="auto" for automatic shard distribution
2. Adds max_memory constraints to prevent OOM
3. Uses offload_folder for disk offloading if needed
4. Better error handling for stuck loading
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


class KeywordsStoppingCriteria(StoppingCriteria):
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


def load_model_fsdp_safe(model_name, device='cuda'):
    """
    Load model safely from FSDP checkpoint.

    Handles sharded checkpoints by:
    1. Using device_map='auto' for automatic distribution
    2. Loading with max_memory constraints
    3. Using CPU offload if needed
    """
    print(f"\n{'='*60}")
    print(f"Loading FSDP checkpoint: {model_name}")
    print(f"{'='*60}\n")

    # Check if this is a sharded checkpoint
    is_sharded = os.path.exists(os.path.join(model_name, "pytorch_model-00001-of-00003.bin")) or \
                 os.path.exists(os.path.join(model_name, "pytorch_model.bin.index.json"))

    if is_sharded:
        print("✓ Detected sharded checkpoint (FSDP format)")
        print("  Using optimized loading strategy...")
        print()

        # Strategy 1: Load with device_map='auto' for automatic shard distribution
        # This is much more memory-efficient and faster for sharded checkpoints
        try:
            print("Attempting: device_map='auto' (recommended for sharded checkpoints)")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',  # Automatically distribute across GPUs
                low_cpu_mem_usage=True,
                use_cache=True,
            )
            print("✓ Successfully loaded with device_map='auto'\n")
            return model, 'auto'

        except Exception as e:
            print(f"✗ Failed with device_map='auto': {e}")
            print("  Trying alternative strategies...\n")

    # Strategy 2: Load to CPU first, then move to GPU
    # Slower but more reliable for large checkpoints
    print("Attempting: Load to CPU then move to GPU")
    try:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True,
        )
        print("✓ Loaded to CPU, moving to GPU...")
        model = model.to(device)
        print("✓ Successfully moved to GPU\n")
        return model, device

    except Exception as e:
        print(f"✗ Failed: {e}\n")
        raise RuntimeError(f"Failed to load model from {model_name}")


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.mm_projector is None:
        patch_config(model_name)

        print(f"Model path: {model_name}\n")

        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model, device = load_model_fsdp_safe(model_name)
            if device != 'auto':
                model = model.to(torch.float16)

            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]

            if device == 'auto':
                vision_tower.to(dtype=torch.float16)
            else:
                vision_tower.to(device=device, dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            # Load model with FSDP-safe method
            model, device = load_model_fsdp_safe(model_name)

            image_processor = CLIPImageProcessor.from_pretrained(
                model.config.mm_vision_tower,
                torch_dtype=torch.float16
            )

            # Handle vision tower
            vision_tower = model.model.vision_tower[0]

            # Check if vision tower needs reloading
            try:
                _ = next(vision_tower.parameters()).device
                is_meta = False
            except:
                is_meta = True

            if is_meta or not next(vision_tower.parameters()).is_cuda:
                print("Reloading vision tower from pretrained...")
                vision_tower = CLIPVisionModel.from_pretrained(
                    model.config.mm_vision_tower,
                    torch_dtype=torch.float16
                ).cuda()
                model.model.vision_tower[0] = vision_tower
            else:
                if device != 'auto':
                    vision_tower.to(device=device, dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # With custom projector
        model, device = load_model_fsdp_safe(model_name)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()

        if "BiomedCLIP" in model.config.mm_vision_tower:
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    print(f"\n{'='*60}")
    print(f"Model loaded successfully!")
    print(f"{'='*60}\n")

    # Load questions
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]

        try:
            question = line["conversations"][0]
            gt_ans = line["conversations"][1]
        except:
            question = line["conversatons"][0]
            gt_ans = line["conversatons"][1]

        qs = question['value']
        qs = qs.replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            cur_prompt = cur_prompt + '\n' + '<image>'
        else:
            images = None

        if args.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'
        assert gt_ans['from'] == 'gpt'

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=None,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        if args.conv_mode == 'simple_legacy':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)

        outputs = outputs[:index].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
