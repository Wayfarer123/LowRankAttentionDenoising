from .modules import LlamaDINT_EnhancedAttention
import torch
from types import MethodType
from transformers.cache_utils import DynamicCache


def convert_to_DINT_attention(
    model,
    **kwargs
):
    for i in range(len(model.model.layers)):
        if not hasattr(model.model.layers[i].self_attn, "adapter_q_proj"):
            device = next(model.model.layers[i].self_attn.parameters()).device
            dtype = next(model.model.layers[i].self_attn.parameters()).dtype
            model.model.layers[i].self_attn = LlamaDINT_EnhancedAttention(model.model.layers[i].self_attn, **kwargs).to(device, dtype)

    def new___getitem__(self, layer_idx: int):

        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.key_cache2[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def new___iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.key_cache2[layer_idx], self.value_cache[layer_idx])
        
    def new_update(
        self,
        key_states: torch.Tensor,
        key_states2: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs = None
    ):
        # Update the number of seen tokens
        if layer_idx ==0:
            self_seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.key_cache2.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.key_cache2.append(key_states2)
                self.value_cache.append(value_states)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ): # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.key_cache2[layer_idx] = key_states2
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states2], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.key_cache2[layer_idx], self.value_cache[layer_idx]
    
    def new_to_legacy_cache(self):
        legacy_cache = ()

        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.key_cache2[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache
    
    def new_from_legacy_cache(self, cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, key_states2, value_states = past_key_values[layer_idx]
                cache.update(key_states, key_states2, value_states, layer_idx)
        return cache

    def new_crop(self, max_length: int):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        
        if self.get_seq_length() <= max_length:
            return
        
        self._seen_tokens = max_length
        
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., : max_length, :]
                self.key_cache2[idx] = self.key_cache2[idx][..., : max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., : max_length, :]

    def new_batch_split(self, full_batch_size: int, split_size: int):
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.key_cache2 = [tensor[i : i + split_size] for tensor in self.key_cache2]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out
    
    def new_from_batch_splits(self, cls, splits):
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            key_cache2 = [current.key_cache2[idx] for current in splits if current.key_cache2[idx] != []]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx] != []]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_keys2 = torch.cat(key_cache2, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_keys2, layer_values, idx)
        return cache
    
    def new_batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_intrerleave(repeats, dim=0)
            self.key_cache2[layer_idx] = self.key_cache2[layer_idx].repeat_intrerleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_intrerleave(repeats, dim=0)
    
    def new_batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.key_cache2[layer_idx] = self.key_cache2[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
    
    def _reset(self):
        self.key_cache = []
        self.key_cache2 = []
        self.value_cache = []
        return self
    

    if kwargs.get("use_cache", False):
        new_dynamic_cache = DynamicCache()
        new_dynamic_cache.key_cache2 = []

        new_dynamic_cache.__getitem__ = MethodType(new___getitem__, new_dynamic_cache)
        new_dynamic_cache.__iter__ = MethodType(new___iter__, new_dynamic_cache)
        new_dynamic_cache.update = MethodType(new_update, new_dynamic_cache)
        new_dynamic_cache.to_legacy_cache = MethodType(new_to_legacy_cache, new_dynamic_cache)
        new_dynamic_cache.from_legacy_cache = MethodType(new_from_legacy_cache, new_dynamic_cache)
        new_dynamic_cache.crop = MethodType(new_crop, new_dynamic_cache)
        new_dynamic_cache.batch_split = MethodType(new_batch_split, new_dynamic_cache)
        new_dynamic_cache.from_batch_splits = MethodType(new_from_batch_splits, new_dynamic_cache)
        new_dynamic_cache.batch_repeat_interleave = MethodType(new_batch_repeat_interleave, new_dynamic_cache)
        new_dynamic_cache.batch_select_indices = MethodType(new_batch_select_indices, new_dynamic_cache)

        new_dynamic_cache.reset = MethodType(_reset, new_dynamic_cache)

        original_forward = model.forward 

        def new_forward_with_overriding_dynamic_cache(self, *args, **kwargs):
            if kwargs.get("past_key_values") is None:
                kwargs["past_key_values"] = new_dynamic_cache._reset()
            elif len(kwargs["past_key_values"]) == 0:
                kwargs["past_key_values"] = new_dynamic_cache._reset()
            return original_forward(self, *args, **kwargs)
        
        model.forward = MethodType(new_forward_with_overriding_dynamic_cache, model)

    return model


def make_DINT_adapter_trainable(
    model
):
    for layer in model.model.layers:
        for param in layer.self_attn.adapter_q_proj.parameters():
            param.requires_grad = True
        for param in layer.self_attn.adapter_k_proj.parameters():
            param.requires_grad = True
        layer.self_attn.lambd.requires_grad = True

    return model