from transformers import PretrainedConfig


class LOCOSTConfig_Mamba(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32128,
        d_model=768,
        d_state=256,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_decoder_layers=None,
        num_heads=12,
        num_ssm_heads=None,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        encoder_attention_type="local",
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        relative_attention_num_buckets=32,   # or 128 for long contexts
        relative_attention_max_distance=128, # typical default for T5
        max_length=4096,
        global_block_size=16,
        local_radius=127,
        # Mamba-specific
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        **kwargs,
    ):
        # Standard transformer-ish params
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else num_layers
        )
        self.num_heads = num_heads
        self.num_ssm_heads = num_ssm_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.encoder_attention_type = encoder_attention_type
        self.use_cache = use_cache
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.max_length = max_length
        self.global_block_size = global_block_size
        self.local_radius = local_radius

        # Mamba-specific params
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.bias = bias
        self.use_fast_path = use_fast_path

        # Activation handling
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if (len(act_info) > 1 and act_info[0] != "gated") or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function. "
                "Use `gated-{ACT_FN}` or `{ACT_FN}`, e.g. 'gated-gelu' or 'relu'."
            )

        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )