import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

from bitnet import BitLinearNew
import torch
from torch import nn
from transformers import DynamicCache
from transformers.activations import ACT2FN
from transformers.utils import logging

from model.anemone_config import AnemoneConfig


# try except block so it'll work with trust_remote_code. Later we can have `if is_mamba_ssm_available():`
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

# try except block so it'll work with trust_remote_code. Later we can have `if is_causal_conv1d_available():`
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)


logger = logging.get_logger(__name__)

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Jamba
class AnemoneRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).
    It stores the Key and Value states as a list of tensors, one for each layer.
    The expected shape for each tensor for attention layers is `[batch_size, num_heads, seq_len, head_dim]`.
    For the mamba layers, the `key_cache` represents the convolution state and has a shape of `[batch_size, d_inner, 1, d_conv]`,
    and the `value_cache` represents the ssm state and has a shape of `[batch_size, d_inner, 1, d_state]`. Mamba cache
    shape[2] is a dummy "seqlen" dimension to match the number of attention cache dimensions. For mamba, the cache
    doesn't grow with seqlen so this dimension is always 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_layer_idx = None  # used to know which layer has data on seqlen in the cache shape

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `HybridMambaAttentionDynamicCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if self.attention_layer_idx is None and self._is_attn_layer(key_states, value_states):
            self.attention_layer_idx = layer_idx
        if self.attention_layer_idx is not None and layer_idx == self.attention_layer_idx:
            if hasattr(self, "_seen_tokens"):
                self._seen_tokens += key_states.shape[-2]
            else:
                self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                # attention layer - append the new states to the existing cache on the seqlen dimension
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            else:
                # mamba layer - replace the cache with the new states
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is not None:
            if len(self.key_cache) <= layer_idx:
                return 0
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                return self.key_cache[layer_idx].shape[-2]
            else:
                warnings.warn(
                    f"Asked to get the sequence length from cache of layer {layer_idx} which is not an attention layer. "
                    f"Ignoring that and using an attention layer cache"
                )
        if self.attention_layer_idx is None or len(self.key_cache) <= self.attention_layer_idx:
            return 0
        return self.key_cache[self.attention_layer_idx].shape[-2]

    @staticmethod
    def _is_attn_layer(key_states: torch.Tensor, value_states: torch.Tensor):
        return key_states.shape[-1] == value_states.shape[-1]


@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class AnemoneMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: AnemoneConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.apply_inner_layernorms = config.mamba_inner_layernorms

        self.use_fast_kernels = config.use_mamba_kernels

        # projection of the input hidden states
        self.in_proj = BitLinearNew(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = BitLinearNew(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = BitLinearNew(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = BitLinearNew(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        if self.apply_inner_layernorms:
            self.dt_layernorm = AnemoneRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
            self.B_layernorm = AnemoneRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
            self.C_layernorm = AnemoneRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: MambaCacheParams = None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if (
                self.training and cache_params is None and not self.apply_inner_layernorms
        ):  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            time_step, B, C = self._apply_layernorms(time_step, B, C)

            # Here we need to apply dt_proj without the bias, as the bias is added in the selective scan kernel.
            # This is a hack to apply dt_proj while still using the forward pass of `torch.nn.Linear`, which is needed
            # in order to make quantization work. Quantization code replaces `torch.nn.Linear` layers with quantized
            # linear layers, and requires to call the forward pass directly.
            # The original code here was: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
            if hasattr(self.dt_proj, "base_layer"):
                # In case of LoRA, we need to access the base layer to get the weight
                time_proj_bias = self.dt_proj.base_layer.bias
                self.dt_proj.base_layer.bias = None
            else:
                time_proj_bias = self.dt_proj.bias
                self.dt_proj.bias = None
            discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
            if hasattr(self.dt_proj, "base_layer"):
                self.dt_proj.base_layer.bias = time_proj_bias
            else:
                self.dt_proj.bias = time_proj_bias

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params: MambaCacheParams = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            if self.training:
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backwards pass
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        time_step, B, C = self._apply_layernorms(time_step, B, C)
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, intermediade_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def mixer_forward(self, hidden_states, cache_params: MambaCacheParams = None):
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device"
                )
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        if past_key_value is not None:
            cache_params = MambaCacheParams(
                seqlen_offset=0 if hidden_states.shape[1] > 1 else past_key_value.seen_tokens,
            )
            if len(past_key_value.key_cache) > self.layer_idx:
                # we already have cache for this layer, use it
                # remove the dummy seqlen dim (dim=2)
                cache_params.conv_states[self.layer_idx] = past_key_value.key_cache[self.layer_idx].squeeze(2)
                cache_params.ssm_states[self.layer_idx] = past_key_value.value_cache[self.layer_idx].squeeze(2)
            else:
                # we don't have cache for this layer, initialize it with zeros
                batch_size = hidden_states.shape[0]
                cache_params.conv_states[self.layer_idx] = torch.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                cache_params.ssm_states[self.layer_idx] = torch.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
        else:
            cache_params = None

        res = self.mixer_forward(hidden_states, cache_params)

        if past_key_value is not None:
            past_key_value.update(
                # add dummy seqlen dim (dim=2) to match the number of dimensions of the attention cache
                cache_params.conv_states[self.layer_idx].unsqueeze(2),
                cache_params.ssm_states[self.layer_idx].unsqueeze(2),
                self.layer_idx,
            )

        return res, past_key_value
