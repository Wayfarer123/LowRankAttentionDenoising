import torch
from torch import nn
from typing import Optional, Tuple


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def DINT_Attention_forward(
    module: nn.Module,
    query: torch.Tensor, 
    key: torch.Tensor, 
    query2: torch.Tensor, 
    key2: torch.Tensoor, 
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float=0.0,
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    key_states2 = repeat_kv(key2, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[0]]

        attn_weights = nn.functional.softmax(torch.matmul(query, key_states.transpose(2, 3)) * scaling + causal_mask, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = attn_weights.mean(dim=-2, keepdims=True) * module.lambd + attn_weights
        attn_weights = attn_weights - module.lambd * nn.functional.softmax(torch.matmul(query2, key_states2.transpose(2,3)) * scaling + causal_mask, dim=-1, dtype=torch.float32).to(query2.dtype)

    else:
        attn_weights = nn.functional.sfotmax(torch.matmul(query, key_states.transpose(2, 3)) * scaling, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = attn_weights.mean(dim=-2, keepdims=True) * module.lambd + attn_weights
        attn_weights = attn_weights - module.lambd * nn.functional.softmax(torch.matmul(query2, key_states2.transpose(2,3)) * scaling, dim=-1, dtype=torch.float32).to(query2.dtype)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaDINT_EnhancedAttention(nn.Module):
    def __init__(
        self,
        LlamaAttentionLayer=None,
        **kwargs
    ):
        if LlamaAttentionLayer is None:
            raise ValueError("cannot create LlamaDINT_EnhancedAttention, 'LlamaAttentionLayer' not passed")
        super().__init__()
        self._fromLlamaAttention(LlamaAttentionLayer, **kwargs)

    def _fromLlamaAttention(
        self, 
        LlamaAttentionLayer,
        adapter_head_dim,
        lambd_init=0.0,
        **kwargs
    ):
        self.config = LlamaAttentionLayer.config
        self.layer_idx = LlamaAttentionLayer.layer_idx
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = LlamaAttentionLayer.q_proj
        self.k_proj = LlamaAttentionLayer.k_proj
        self.v_proj = LlamaAttentionLayer.v_proj
        self.o_proj = LlamaAttentionLayer.o_proj

        self.adapterHead_dim = adapter_head_dim

        self.adapter_q_proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.num_attention_Heads * adapter_head_dim),
            nn.Linear(self.config.num_attention_heads * adapter_head_dim, self.config.num_attention_heads * self.head_dim)
        )
        
        self.adapter_k_proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.key_value_heads * adapter_head_dim),
            nn.Linear(self.config.key_value_heads * adapter_head_dim, self.config.key_value_heads * self.head_dim)
        )

        self.lambd = nn.Parameter(torch.tensor([lambd_init], requeires_grad=True))

        if kwargs.get("use_cache", False):
            self.use_cache = True
        else:
            self.use_cache = False

    def forward(
        self, 
        hidden_states: torch.tensor, 
        position_embeddings: Tuple[torch.tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
        cache_position: Optional[torch.Longtensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1,2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)

        query2 = self.adapter_q_proj(hidden_states).view(hidden_shape).transpose(1,2)
        key2 = self.adapter_k_proj(hidden_states).view(hidden_shape).transpose(1,2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states2, key_states2 = apply_rotary_pos_emb(query_states2, key_states2, cos, sin)

        if self.use_cache:
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, key_states2, value_states = past_key_value.update(key_states, key_states2, value_states, self.layer_idx, cache_kwargs)
        
        attn_output, attn_weights = DINT_Attention_forward(
            self, 
            query_states,
            key_states,
            query_states2,
            key_states2,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1).contigous()
        attn_output = self.o_proj(attn_output) 

        return attn_output, attn_weights
