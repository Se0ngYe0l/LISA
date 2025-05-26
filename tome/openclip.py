import torch
from typing import Dict, Any, Tuple, Optional
from torch import nn

from .utils import isinstance_str, init_generator, parse_r
from .merge import merge_wavg, merge_source, bipartite_soft_matching, bipartite_soft_matching_random2d

class ToMeResidualAttentionBlock(nn.Module):
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                **kwargs):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        x_attn, metric = self.self_attn(self.layer_norm1(x), attn_size)
        x = x + x_attn

        r = self._tome_info["r"].pop(0)
        if r > 0:
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self.mlp(self.layer_norm2(x))
        #print("ToMe block output shape:", x.shape)
        #print(x.shape)
        return x


class ToMeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

    def forward(
        self, x: torch.Tensor,
        size: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.q_proj.weight.dtype)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, N, C = x.shape
        assert C == self.embed_dim, f"Input channel mismatch: got {C}, expected {self.embed_dim}"

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn += size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        #print(k.mean(1).shape)
        return x, k.mean(1)

def convert_attention_block(src: nn.Module, dst: ToMeAttention) -> Tuple[ToMeAttention, torch.device]:
    src_state_dict = src.state_dict()
    dst_state_dict = dst.state_dict()

    dst_state_dict["q_proj.weight"] = src_state_dict["q_proj.weight"]
    dst_state_dict["q_proj.bias"] = src_state_dict["q_proj.bias"]
    dst_state_dict["k_proj.weight"] = src_state_dict["k_proj.weight"]
    dst_state_dict["k_proj.bias"] = src_state_dict["k_proj.bias"]
    dst_state_dict["v_proj.weight"] = src_state_dict["v_proj.weight"]
    dst_state_dict["v_proj.bias"] = src_state_dict["v_proj.bias"]
    dst_state_dict["out_proj.weight"] = src_state_dict["out_proj.weight"]
    dst_state_dict["out_proj.bias"] = src_state_dict["out_proj.bias"]

    dst.load_state_dict(dst_state_dict)
    src_device = dst_state_dict["q_proj.weight"].device
    return dst.to(src_device), src_device


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.encoder.layers), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def patch_openclip(model, trace_source: bool = False, prop_attn: bool = True):
    vision_model = model
    ToMeVisionTransformer = make_tome_class(vision_model.__class__)

    vision_model.__class__ = ToMeVisionTransformer
    model.r = 0
    vision_model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": hasattr(vision_model, "dist_token") and vision_model.dist_token is not None,
    }

    for i, resblock in enumerate(vision_model.encoder.layers):
        resblock.__class__ = ToMeResidualAttentionBlock
        resblock._tome_info = vision_model._tome_info

        attn = ToMeAttention(
            resblock.self_attn.q_proj.in_features,
            resblock.self_attn.num_heads,
            qkv_bias=True
        )
        _, device = convert_attention_block(resblock.self_attn, attn)
        attn = attn.to(dtype=resblock.self_attn.q_proj.weight.dtype, device=device)
        resblock.self_attn = attn
