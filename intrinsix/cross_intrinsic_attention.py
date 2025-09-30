import einops
import torch
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb

from torch.nn.modules.module import _global_forward_hooks, _global_forward_hooks_always_called


class CrossIntrinsicAttnProcessor2_0(nn.Module):
    """
    Processor for implementing scaled dot-product attention between multiple images within each batch.
    Inspired by the FluxAttnProcessor2_0 of diffusers.
    """

    def __init__(self, *args, dropout=None, **kwargs):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CrossIntrinsicAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.dropout = dropout if dropout is not None else 0

    def process_attention(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        num_components = batch_size

        # print(f"Hidden_states before: {hidden_states[0], hidden_states[1]}")

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Enable cross-attention along the batch dimension
        # ===============================================
        use_crossattention = torch.rand(1) > self.dropout
        if use_crossattention:
            key = einops.rearrange(key, "b h f d -> 1 h (b f) d").repeat(batch_size, 1, 1, 1)
            value = einops.rearrange(value, "b h f d -> 1 h (b f) d").repeat(batch_size, 1, 1, 1)
            if encoder_hidden_states is not None:
                key_rotary_embedding = tuple((torch.cat([emb[:512]] + [emb[512:]] * num_components, dim=0) for emb in image_rotary_emb))
            else:
                key_rotary_embedding = tuple((emb.repeat(num_components, 1) for emb in image_rotary_emb))
        else:
            key_rotary_embedding = image_rotary_emb

        # ===============================================

        # The attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, key_rotary_embedding)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        called_always_called_hooks = set()
        result = self.process_attention(attn=attn,
                                       hidden_states=hidden_states,
                                       encoder_hidden_states=encoder_hidden_states,
                                       attention_mask=attention_mask,
                                       image_rotary_emb=image_rotary_emb)
        args = []
        kwargs = {"attn": attn,
                 "hidden_states": hidden_states,
                 "encoder_hidden_states": encoder_hidden_states,
                 "attention_mask": attention_mask,
                 "image_rotary_emb": image_rotary_emb}
        
        if _global_forward_hooks or self._forward_hooks:
            for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
            ):
                # mark that always called hook is run
                if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                    called_always_called_hooks.add(hook_id)

                if hook_id in self._forward_hooks_with_kwargs:
                    hook_result = hook(self, args, kwargs, result)
                else:
                    hook_result = hook(self, args, result)

                if hook_result is not None:
                    result = hook_result
        return result