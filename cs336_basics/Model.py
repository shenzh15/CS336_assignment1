import torch
import einops

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()

        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')

        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
        self.weight = torch.nn.Parameter(weight)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        self.weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.weight, a=-3, b=3)
        self.weight = torch.nn.Parameter(self.weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        self.weight = torch.ones((d_model), device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(self.weight)
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        x = x.to(torch.float32)
        result = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight
        return result.to(input_type)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')

        # Calculate d_ff automatically if not provided
        if d_ff is None:
            # Set d_ff to approximately 8/3 * d_model, rounded to nearest multiple of 64
            d_ff_raw = int((8 / 3) * d_model)
            d_ff = ((d_ff_raw + 63) // 64) * 64  # Round up to nearest multiple of 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.d_model = d_model
        self.d_ff = d_ff

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')

        assert d_k % 2 == 0, "d_k must be even for rotary positional embedding"
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k) )
        positions = torch.arange(max_seq_len, device=device).float()
        freqs_pos = torch.outer(positions, freqs)
        self.register_buffer('cos_cache', torch.cos(freqs_pos))
        self.register_buffer('sin_cache', torch.sin(freqs_pos))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)

        cos = self.cos_cache[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cache[token_positions]  # (..., seq_len, d_k//2)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_even * cos - x_odd * sin # broadcast happens here
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos

        return x_rotated

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    scale = torch.sqrt(
        torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))
    scores = einops.einsum(q, k, "... i d_k, ... j d_k -> ... i j") / scale
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attention = einops.einsum(softmax(scores, dim=-1), v,
                              "... i j, ... j d_v -> ... i d_v")
    return attention

class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.hd_k = self.head_dim*num_heads # d_q == d_k == d_v
        self.qkv = Linear(d_model, 3 * self.hd_k, device=device, dtype=dtype)
        self.out = Linear(self.hd_k, d_model, device=device, dtype=dtype) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.hd_k, dim=-1)
        q = einops.rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = einops.rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = einops.rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool))
        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = einops.rearrange(attention, "... h seq_len d_k -> ... seq_len (h d_k)")
        return self.out(attention)

class multihead_self_attention_with_rope(torch.nn.Module):
    shared_rope = None
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.hd_k = self.head_dim*num_heads # d_q == d_k == d_v
        self.qkv = Linear(d_model, 3 * self.hd_k, device=device, dtype=dtype)
        self.out = Linear(self.hd_k, d_model, device=device, dtype=dtype) 
        cls = type(self)
        if cls.shared_rope is None:
            cls.shared_rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
        self.rope = cls.shared_rope

    def forward(self, x: torch.Tensor, token_positions : torch.Tensor | None = None) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.hd_k, dim=-1)
        q = einops.rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = einops.rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = einops.rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool))
        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = einops.rearrange(attention, "... h seq_len d_k -> ... seq_len (h d_k)")
        return self.out(attention)

class transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class transformer_lm(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,      # The number of unique items in the output vocabulary to be predicted.
        context_length: int,  # The maximum number of tokens to process at once.
        d_model: int,         # The dimension of the model.
        num_layers: int,      # The number of transformer blocks.
        num_heads: int,       # The number of attention heads.
        d_ff: int,            # The dimension of the feedforward network.
        rope_theta: float,    # The theta value for RoPE.
        device = None,        # The device to use for the model.
        dtype = None          # The dtype to use for the model.
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)       # (batch_size, context_length, d_model)
        x = self.lm_head(x)  # (batch_size, context_length, vocab_size)
        return x
        