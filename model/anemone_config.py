from transformers import PretrainedConfig

import math

class AnemoneConfig(PretrainedConfig):

    def __init__(self,
                 vocab_size=65536,
                 tie_word_embeddings=False,
                 hidden_size=4096,
                 intermediate_size=14336,
                 num_hidden_layers=32,
                 num_attention_heads=32,
                 num_key_value_heads=8,
                 hidden_act="silu",
                 initializer_range=0.02,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 calc_logits_for_entire_prompt=False,
                 output_router_logits=False,
                 mlp_router_aux_loss_coef=0.001,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 sliding_window=None,
                 n_ctx=262144,
                 attention_dropout=0.0,
                 num_experts_per_tok=2,
                 num_experts=16,
                 expert_layer_period=2,
                 expert_layer_offset=1,
                 attn_layer_period=8,
                 attn_layer_offset=4,
                 use_mamba_kernels=True,
                 mamba_d_state=16,
                 mamba_d_conv=4,
                 mamba_expand=2,
                 mamba_dt_rank="auto",
                 mamba_conv_bias=True,
                 mamba_proj_bias=False,
                 mamba_inner_layernorms=True,
                 attn_num_experts=16,
                 attn_top_k=4,
                 attn_router_aux_loss_coef=0.001,
                 mod_aux_loss_coef=0.001,
                 mod_routing=True,
                 mod_aux_routing=False,
                 capacity=128,
                 skip_blocks=2,
                 expert_num_heads=8,
                 **kwargs, ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.n_ctx = n_ctx
        self.attention_dropout = attention_dropout

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.calc_logits_for_entire_prompt = calc_logits_for_entire_prompt
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = mlp_router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms

        # Adding config for moah
        self.attn_num_experts = attn_num_experts
        self.attn_top_k = attn_top_k
        self.attn_router_aux_loss_coef = attn_router_aux_loss_coef

        # Adding config for mod
        self.mod_aux_loss_coef = mod_aux_loss_coef
        self.mod_routing = mod_routing
        self.mod_aux_routing = mod_aux_routing
        self.capacity = capacity
        self.skip_blocks = skip_blocks
        self.expert_num_heads = expert_num_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )