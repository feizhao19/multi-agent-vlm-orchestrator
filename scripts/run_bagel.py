from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _add_bagel_repo_to_path() -> None:
    repo_path = os.environ.get("BAGEL_REPO_PATH")
    if not repo_path:
        raise RuntimeError("Set BAGEL_REPO_PATH to a local clone of the official BAGEL repo.")
    sys.path.insert(0, repo_path)


def _load_inferencer(model_id: str) -> Any:
    _add_bagel_repo_to_path()
    try:
        import torch
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
        from data.data_utils import add_special_tokens
        from modeling.bagel import (
            BagelConfig,
            Bagel,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.siglip import SiglipImageProcessor
        from modeling.autoencoder import load_ae
        from inferencer import InterleaveInferencer
    except ImportError as exc:
        raise RuntimeError(
            "BAGEL dependencies are not importable. Activate the BAGEL environment and "
            "make sure BAGEL_REPO_PATH points to the official repo clone."
        ) from exc

    model_path = Path(os.environ.get("BAGEL_MODEL_PATH", model_id)).expanduser()
    llm_config = Qwen2Config.from_json_file(model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with torch.device("meta"):
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path / "tokenizer")
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = vae_config.vae_transform
    vit_transform = SiglipImageProcessor(
        size=384,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
    )

    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(model_path / "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
    )
    model = model.eval()

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


def _text_to_image(args: argparse.Namespace) -> None:
    inferencer = _load_inferencer(args.model_id)
    image = inferencer(
        text=args.prompt,
        think=False,
        understanding_output=False,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )["image"]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _vision_to_text(args: argparse.Namespace) -> None:
    inferencer = _load_inferencer(args.model_id)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for BAGEL vision-to-text.") from exc

    image = Image.open(args.image_path).convert("RGB")
    output = inferencer(
        image=image,
        text=args.prompt,
        understanding_output=True,
        max_think_token_n=args.max_new_tokens,
        do_sample=False,
    )
    print(str(output.get("text", "")).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Local wrapper for official BAGEL inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    t2i = subparsers.add_parser("text-to-image")
    t2i.add_argument("--model-id", required=True)
    t2i.add_argument("--prompt", required=True)
    t2i.add_argument("--output-path", required=True)
    t2i.add_argument("--cfg-text-scale", type=float, default=4.0)
    t2i.add_argument("--num-timesteps", type=int, default=50)
    t2i.set_defaults(func=_text_to_image)

    v2t = subparsers.add_parser("vision-to-text")
    v2t.add_argument("--model-id", required=True)
    v2t.add_argument("--image-path", required=True)
    v2t.add_argument("--prompt", required=True)
    v2t.add_argument("--max-new-tokens", type=int, default=512)
    v2t.set_defaults(func=_vision_to_text)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
