import argparse
import torch
import deepspeed
import transformers
from PIL import Image
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="lisa_model_path_or_hf_name")
    parser.add_argument("--text", type=str, default="Put the yellow cone on top of the tennis ball.")
    parser.add_argument("--image_path", type=str, default="/home/seongyeol/LISA/0000814.png")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.version,
                                              cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    print("load tokenizer")
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Processor (text + image)
    processor = transformers.AutoProcessor.from_pretrained(args.version)

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": 256,
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": "/home/seongyeol/LISA/sam_vit_h_4b8939.pth",
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }

    # Model dtype
    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.float16
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16

    # Load model
    model = LISAForCausalLM.from_pretrained(args.version, torch_dtype=torch_dtype, **model_args)
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        "llava_v1"
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = 16
        lora_dropout = 0.05
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    # DeepSpeed inference
    model_engine = deepspeed.init_inference(
        model=model,
        mp_size=1,
        dtype=torch_dtype,
        replace_with_kernel_inject=False,
    )

    print("data preprocessing")
    # Prepare inputs
    image = Image.open(args.image_path).convert("RGB")
    inputs = processor(images=image, text=args.text, return_tensors="pt").to(args.local_rank)

    with torch.no_grad():
        outputs = model_engine.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            do_sample=False,
        )

    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
