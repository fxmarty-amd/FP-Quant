import gc
import argparse
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quantizer import Quantizer
from .quant_ops import pack_fp4_to_uint8, prepare_scales_for_saving

from typing import List

from ..utils.model_utils import get_attention_layer, get_mlp_layer
from ..transforms.transforms import build_transform, get_transform_matrix

from tqdm import tqdm
from quark.torch.algorithm.utils.module import get_nested_attr_from_module
from quark.torch.algorithm.rotation.rotation_utils import  transform_rms_norm_and_linear
from quark.torch.utils.accelerate_helper import untie_parameters

SCALING_LAYERS_ABSTRACT = {
    "first_layer": [
        {
            "prev_modules": ["model.embed_tokens"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]
        }
    ],
    "middle_layers": [
        {
            "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]
        }
    ],
    "last_layer": [
        {
            "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
            "norm_module": "model.norm",
            "next_modules": ["lm_head"]
        }
    ]
}

class ModuleWrapped(nn.Module):
    def __init__(self, module: nn.Module, transform, position: str):
        super().__init__()
        self.module = module
        self.transform = transform
        self.position = position

        assert self.position in ["before", "after"]
    
    def forward(self, x):
        if self.position == "before":
            x = self.transform(x)
        
        x = self.module(x)

        if self.position == "after":
            x = self.transform(x)
        
        return x

def get_prev_out_channels_dims(prev_modules: List[nn.Module]) -> List[int]:
    prev_out_channels_dims = []
    for module in prev_modules:
        if isinstance(module, nn.Embedding):
            prev_out_channels_dims.append(1)
        elif isinstance(module, nn.Linear):
            prev_out_channels_dims.append(0)
        else:
            raise ValueError("prev_modules is wrong")
    return prev_out_channels_dims


def get_scaling_layers(scaling_layers_abstract, model):
    scaling_layers = []
    for i in range(len(model.model.layers)):
        scaling_layers_cur = []

        if i == 0:
            for layers_pattern in scaling_layers_abstract["first_layer"]:
                scaling_layers_cur.append({
                    "prev_modules":
                    [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["prev_modules"]],
                    "norm_module":
                    layers_pattern["norm_module"].replace("layer_id", str(i)),
                    "next_modules":
                    [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["next_modules"]]
                })
        else:
            for layers_pattern in scaling_layers_abstract["middle_layers"]:
                scaling_layers_cur.append({
                    "prev_modules": [
                        layer_name.replace("pre_layer_id", str(i - 1)).replace("layer_id", str(i))
                        for layer_name in layers_pattern["prev_modules"]
                    ],
                    "norm_module":
                    layers_pattern["norm_module"].replace("layer_id", str(i)),
                    "next_modules":
                    [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["next_modules"]]
                })
            if i == len(model.model.layers) - 1:
                for layers_pattern in scaling_layers_abstract["last_layer"]:
                    scaling_layers_cur.append({
                        "prev_modules":
                        [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["prev_modules"]],
                        "norm_module":
                        layers_pattern["norm_module"].replace("layer_id", str(i)),
                        "next_modules":
                        [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["next_modules"]]
                    })
        scaling_layers.extend(scaling_layers_cur)
    return scaling_layers


def rtn_quantization(
    model: AutoModelForCausalLM, 
    args: argparse.Namespace, 
    device: torch.device
) -> Optional[dict[str, torch.Tensor]]:
    print("RTN quantization...")
    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    # State dict with quantized weights, scales and hadamards
    quantized_state_dict = {}
    # Get transformer blocks
    blocks = model.model.layers
    # Define common transform kwargs
    transform_kwargs = dict(group_size=args.hadamard_group_size)
    print("transform_kwargs", transform_kwargs)

    assert model.config.model_type == "llama"

    # Init quantizers
    weight_quantizer = None
    if args.w_bits < 16 and not args.no_quant:
        weight_quantizer = Quantizer(
            bits=args.w_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.w_granularity,
            observer=args.w_observer, 
            group_size=args.w_group_size,
            scale_precision=args.scale_precision,
            scale_factor=args.mxfp_scale_factor,
        )

    act_quantizer = None
    if args.a_bits < 16 and not args.no_quant:
        act_quantizer = Quantizer(
            bits=args.a_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.a_granularity,
            observer=args.a_observer, 
            group_size=args.a_group_size,
            scale_precision=args.scale_precision,
            scale_factor=args.mxfp_scale_factor,
        )

    # embed_tokens and lm_head parameters are shared.
    model = untie_parameters(model)

    # R1: shared accross all layers.
    r1_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
    
    if args.transform_class == "hadamard" and args.fuse_rotations:
        scaling_layers = get_scaling_layers(SCALING_LAYERS_ABSTRACT, model)

        # Before applying R1: edit LayerNorm layers.
        for layers_pattern in tqdm(scaling_layers, desc="RMSNorm update"):
            norm_module = get_nested_attr_from_module(model, layers_pattern["norm_module"])
            next_modules = [
                get_nested_attr_from_module(model, layer_name) for layer_name in layers_pattern["next_modules"]
            ]

            transform_rms_norm_and_linear(norm_module, next_modules)

        # Add R1 to embed_tokens, R1^(-1) to lm_head.
        # R1 will be added in linear layers using the `qkv_in_transform` and `gate_up_in_transform` logic.
        model.model.embed_tokens = ModuleWrapped(model.model.embed_tokens, r1_transform, position="after")

        model.lm_head = ModuleWrapped(model.lm_head, r1_transform, position="before")
        # lm_head can have bias.
        assert not hasattr(model.model.embed_tokens, "bias")

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")

        # R2
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)

        # R4
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config,
            layer_idx=block_idx,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
            qkv_in_transform=r1_transform,
            o_in_transform=o_in_transform,
            gate_up_in_transform=r1_transform,
            fuse_rotations=args.fuse_rotations,
        )
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
            gate_up_in_transform=r1_transform,
            down_in_transform=down_in_transform,
            qkv_in_transform=r1_transform,
            fuse_rotations=args.fuse_rotations,
        )

        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)

        block.self_attn = quantized_attn
        block.mlp = quantized_mlp

        # Move to original device and dtype
        block = block.to(device=device, dtype=orig_dtype)   

        # 3. Fix model parametrization
        if args.real_quant:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    with torch.no_grad():
                        # NOTE for real_quant all transforms are identical
                        weight = r1_transform(layer.weight, inv_t=True)
                        scales, zeros = layer.weight_quantizer.get_quantization_params(weight)
                        qweight = layer.weight_quantizer.quantize(weight, scales, zeros)

                    quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                        "qweight": pack_fp4_to_uint8(qweight),
                        "scales": prepare_scales_for_saving(scales, args.scale_precision, args.mxfp_scale_factor),
                        "forward_hadamard_matrix": get_transform_matrix(args.transform_class, args.w_group_size, device, orig_dtype),
                        "backward_hadamard_matrix": get_transform_matrix(args.transform_class, args.w_group_size, device, orig_dtype)
                    }

        quantized_attn.fix_parametrization()
        quantized_mlp.fix_parametrization()
    
    gc.collect()
    torch.cuda.empty_cache()

    return quantized_state_dict
