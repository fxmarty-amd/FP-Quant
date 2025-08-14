import gc
import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quantizer import Quantizer
from .quant_ops import pack_fp4_to_uint8, prepare_scales_for_saving

from ..utils.model_utils import get_attention_layer, get_mlp_layer
from ..transforms.transforms import build_transform, get_transform_matrix


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

    # Init quantizers
    weight_quantizer = None
    if args.w_bits < 16:
        weight_quantizer = Quantizer(
            bits=args.w_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.w_granularity,
            observer=args.w_observer, 
            group_size=args.w_group_size,
            scale_precision=args.scale_precision,
            scale_factor=args.mxfp_scale_factor
        )
    act_quantizer = None
    if args.a_bits < 16:
        act_quantizer = Quantizer(
            bits=args.a_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.a_granularity,
            observer=args.a_observer, 
            group_size=args.a_group_size,
            scale_precision=args.scale_precision,
            scale_factor=args.mxfp_scale_factor
        )

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        print(f"args.transform_class: {args.transform_class}")
        # 1. Init transforms
        qkv_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config,
            layer_idx=block_idx,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
            qkv_in_transform=qkv_in_transform,
            o_in_transform=o_in_transform,
            gate_up_in_transform=gate_up_in_transform
        )
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
            gate_up_in_transform=gate_up_in_transform,
            down_in_transform=down_in_transform,
            qkv_in_transform=qkv_in_transform
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
                        weight = qkv_in_transform(layer.weight, inv_t=True)
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
