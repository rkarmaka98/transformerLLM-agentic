import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        """
        Initializes the MultiHeadAttention module.

        Args:
            hidden_dim (int): The dimension of the input and output features.
            num_heads (int): The number of attention heads.
        """
        # calling constructor (__init__) of the parent class of MultiHeadAttention (nn.Module)
        # initialize the _parameters(hold nn.Parameter objects assigned later),
        #  _buffers(store persistent non-learnable tensors), and _modules(nn.Linear or nn.Conv2d) attributes & cuda(), to(), .train(), .eval() methods
        super(MultiHeadAttention, self).__init__()
        # size of the embedding vector (feature dimension) that each token will be represented by)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def check_sdpa_inputs(self, x):
        assert (
            x.size(1) == self.num_heads
        ), "Input must have the same number of heads as the model"
        assert (
            x.size(3) == self.head_dim
        ), "Input must have the same head dimension as the model"

    def scaled_dot_product_attention(
        self, query, key, value, attention_mask=None, padding_mask=None
    ):
        """
        Computes the scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
            key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
            value (Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim).
            mask (Tensor): Optional mask tensor.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        self.check_sdpa_inputs(query)
        self.check_sdpa_inputs(key)
        self.check_sdpa_inputs(value)

        d_k = query.size(-1)
        tgt_len = query.size(-2)
        src_len = key.size(-2)

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Attention mask handling
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                assert attention_mask.size() == (
                    tgt_len,
                    src_len,
                ), "Attention mask must match the dimensions of query and key"
                attention_mask = attention_mask.unsqueeze(0)
                logits = logits + attention_mask
            else:
                raise ValueError("Attention mask size {attention_mask.size()})")

        # Padding mask handling
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits + padding_mask

        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, value)

        return output, attention

    def split_into_heads(self, x, num_heads):
        """
        Splits the input tensor into multiple heads.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            num_heads (int): Number of attention heads.
        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        return x.permute(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

    def combine_heads(self, x):
        """
        Combines the multiple heads back into a single tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        Returns:
            Tensor: Reshaped tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

    def forward(self, q, k, v, attention_mask=None, key_padding_mask=None):
        """
        Forward pass of the MultiHeadAttention module.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, hidden_dim).
            k (Tensor): Key tensor of shape (batch_size, seq_len, hidden_dim).
            v (Tensor): Value tensor of shape (batch_size, seq_len, hidden_dim).
            attention_mask (Tensor): Optional attention mask.
            key_padding_mask (Tensor): Optional key padding mask.

        Returns:
            Tensor: Output tensor after applying multi-head attention.
        """
        q = self.query_linear(q)
        k = self.key_linear(k)
        v = self.value_linear(v)

        q = self.split_into_heads(q, self.num_heads)
        k = self.split_into_heads(k, self.num_heads)
        v = self.split_into_heads(v, self.num_heads)

        attn_values, attn_weights = self.scaled_dot_product_attention(
            q, k, v, attention_mask, key_padding_mask
        )
        grouped = self.combine_heads(attn_values)
        output = self.Wo(grouped)

        self.attention_weights = attn_weights

        return output
