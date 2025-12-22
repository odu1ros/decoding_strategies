import numpy as np
import polars as pl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional, Mapping, Set, NamedTuple, TypedDict
from typing_extensions import Self

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# BUILD TRANSFORMER

class RMSNorm(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._eps = eps
        self._scale = nn.Parameter(torch.ones(embedding_size))
        self._shift = nn.Parameter(torch.zeros(embedding_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self._eps)
        norm_x = norm_x * self._scale
        if self._shift is not None:
            norm_x = norm_x + self._shift
        return norm_x.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self._fc1 = nn.Linear(embedding_size, hidden_size, bias=False)
        self._fc2 = nn.Linear(embedding_size, hidden_size, bias=False)
        self._fc3 = nn.Linear(hidden_size, embedding_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self._fc1(x)
        x_fc2 = self._fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self._fc3(x)

class MoEFeedForward(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_hidden_size: int,
    ) -> None:
        super().__init__()
        self._num_experts_per_tok = num_experts_per_token
        self._num_experts = num_experts
        self._embedding_size = embedding_size
        self._gate = nn.Linear(embedding_size, num_experts, bias=False)
        
        self._fc1 = nn.ModuleList(
            [
                nn.Linear(embedding_size, moe_hidden_size, bias=False)
                for _ in range(num_experts)
            ]
        )
        self._fc2 = nn.ModuleList(
            [
                nn.Linear(embedding_size, moe_hidden_size, bias=False)
                for _ in range(num_experts)
            ]
        )
        self._fc3 = nn.ModuleList(
            [
                nn.Linear(moe_hidden_size, embedding_size, bias=False)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self._gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self._num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(batch * seq_len, self._embedding_size, device=x.device)

        topk_indices_flat = topk_indices.reshape(-1, self._num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self._num_experts_per_tok)

        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = torch.nn.functional.silu(self._fc1[expert_id](expert_input)) * self._fc2[expert_id](expert_input)
            expert_out = self._fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch, seq_len, self._embedding_size)

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        base: int = 1_000,
    ) -> None:
        super().__init__()
        self._theta = 1 / (torch.pow(torch.tensor(base), (torch.arange(0, embedding_size, 2).float() / embedding_size)))
        self._theta = self._theta.repeat_interleave(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(0, x.size(-2), device=x.device)
        position_matrix = torch.outer(position_ids, self._theta.to(x.device))
        cos = torch.cos(position_matrix)
        sin = torch.sin(position_matrix)
        x_odd = x[..., ::2]
        x_even = x[..., 1::2]

        _x = torch.empty_like(x, device=x.device)
        _x[..., 0::2] = -x_even
        _x[..., 1::2] = x_odd

        _x = _x * sin[:x.size(-2), :]
        x = x * cos[:x.size(-2), :]
        return x + _x

class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        head_embedding_size: int,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._embedding_size = embedding_size
        self._head_embedding_size = head_embedding_size
        self._Q = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._K = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._V = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._W_proj = nn.Linear(self._num_heads * self._head_embedding_size, self._embedding_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self._Q.forward(query).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        k = self._K.forward(key).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        v = self._V.forward(value).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)

        a = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        if mask is not None:
            mask = mask.unsqueeze(1)
            a = a.masked_fill(mask == 0, -torch.inf)
        
        alpha = F.softmax(a, -1)

        z = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(batch_size, -1, self._num_heads * self._head_embedding_size)
        z = self._W_proj(z)
        return z

class RoPEMultiHeadedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        head_embedding_size: int,
        positional_embedding: RotaryPositionEmbedding,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._embedding_size = embedding_size
        self._head_embedding_size = head_embedding_size
        self._positional_embedding = positional_embedding
        self._Q = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._K = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._V = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._W_proj = nn.Linear(self._num_heads * self._head_embedding_size, self._embedding_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self._Q.forward(query).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        k = self._K.forward(key).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        v = self._V.forward(value).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)

        q_rope = self._positional_embedding.forward(q)
        k_rope = self._positional_embedding.forward(k)

        attention_numerator = torch.exp(
            torch.matmul(q_rope, k_rope.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        )
        attention_denominator = torch.exp(
            torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        )
        attention_denominator = torch.sum(attention_denominator, dim=-1, keepdim=True)
        a = attention_numerator / attention_denominator
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            a = a.masked_fill(mask == 0, -torch.inf)
        
        alpha = F.softmax(a, -1)

        z = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(batch_size, -1, self._num_heads * self._head_embedding_size)
        return self._W_proj(z)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_groups: int,
        embedding_size: int,
        head_embedding_size: int,
        positional_embedding: RotaryPositionEmbedding,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._num_kv_groups = num_kv_groups
        self._embedding_size = embedding_size
        self._group_size = num_heads // num_kv_groups
        self._head_embedding_size = head_embedding_size
        self._positional_embedding = positional_embedding
        self._Q = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._K = nn.Linear(self._embedding_size, self._num_kv_groups * self._head_embedding_size)
        self._V = nn.Linear(self._embedding_size, self._num_kv_groups * self._head_embedding_size)
        self._W_proj = nn.Linear(self._num_heads * self._head_embedding_size, self._embedding_size)
        
        self._q_norm = RMSNorm(self._head_embedding_size, eps=1e-6)
        self._k_norm = RMSNorm(self._head_embedding_size, eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self._Q.forward(query).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        k = self._K.forward(key).view(batch_size, -1, self._num_kv_groups, self._head_embedding_size).transpose(1, 2)
        v = self._V.forward(value).view(batch_size, -1, self._num_kv_groups, self._head_embedding_size).transpose(1, 2)

        q = self._q_norm.forward(q)
        v = self._q_norm.forward(v)

        q_rope = self._positional_embedding.forward(q)
        k_rope = self._positional_embedding.forward(k)

        k_rope = k_rope.repeat_interleave(self._group_size, dim=1)
        v = v.repeat_interleave(self._group_size, dim=1)

        a = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        if mask is not None:
            mask = mask.unsqueeze(1)
            a = a.masked_fill(mask == 0, -torch.inf)
        
        alpha = F.softmax(a, -1)

        z = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(batch_size, -1, self._num_heads * self._head_embedding_size)
        return self._W_proj(z)

class DecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        num_kv_groups: int,
        num_experts: int,
        num_experts_per_token: int,
        head_embedding_size: int,
        fcnn_hidden_size: int,
        positional_embedding: RotaryPositionEmbedding,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._mha = GroupedQueryAttention(
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            head_embedding_size=head_embedding_size,
            positional_embedding=positional_embedding,
        )
        self._fcnn = MoEFeedForward(
            embedding_size=embedding_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_hidden_size=fcnn_hidden_size,
        )
        self._rms_norm1 = RMSNorm(embedding_size)
        self._rms_norm2 = RMSNorm(embedding_size)
        self._dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self._rms_norm1(x)
        z = self._mha(x, x, x, mask)

        x = x + self._dropout(z)

        z = self._rms_norm2(x)
        z = self._fcnn(z)
        return x + self._dropout(z)

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        embedding_size: int,
        num_heads: int,
        num_kv_groups: int,
        num_experts: int,
        num_experts_per_token: int,
        head_embedding_size: int,
        fcnn_hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self._positional_embedding = RotaryPositionEmbedding(
            embedding_size=head_embedding_size,
        )
        self._layers = nn.ModuleList(
            DecoderLayer(
                embedding_size=embedding_size,
                num_heads=num_heads,
                num_kv_groups=num_kv_groups,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                head_embedding_size=head_embedding_size,
                fcnn_hidden_size=fcnn_hidden_size,
                positional_embedding=self._positional_embedding,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self._rms_norm = RMSNorm(embedding_size)

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self._embeddings(x)
        for layer in self._layers:
            z = layer(z, mask)
        return self._rms_norm(z)

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        embedding_size: int,
        num_heads: int,
        num_kv_groups: int,
        num_experts: int,
        num_experts_per_token: int,
        head_embedding_size: int,
        fcnn_hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self._decoder = Decoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            head_embedding_size=head_embedding_size,
            fcnn_hidden_size=fcnn_hidden_size,
            dropout=dropout,
        )
        self._logits = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(
        self,
        x: torch.LongTensor,
    ) -> torch.Tensor:
        mask = ~torch.triu(torch.ones((1, x.size(-1), x.size(-1)), device=x.device), 1).to(torch.bool)
        mask = mask & (x != 0).unsqueeze(1) 
        z = self._decoder(x, mask)
        return self._logits(z)

class CharTokenizer:
    def __init__(self):
        self._start_token = "<s>"
        self._end_token = "</s>"
        self._unknown_token = "<UNK>"
        self._padding_token = "<PAD>"
        self._cls_token = "<CLS>"
        self._sep_token = "<SEP>"
        self._padding_id = 0
        self._cls_id = 1
        self._sep_id = 2
        self._start_token_id = 3
        self._end_token_id = 4
        self._unknown_token_id = 5
        self._init_vocab()

    @property
    def vocab(self) -> Mapping[int, str]:
        return self._vocab

    @property
    def reverse_vocab(self) -> Mapping[int, str]:
        return {token: id for id, token in self._vocab.items()}

    @property
    def start_token_id(self) -> int:
        return self._start_token_id

    @property
    def end_token_id(self) -> int:
        return self._end_token_id

    def _init_vocab(self) -> None:
        self._vocab = {
            self._padding_id: self._padding_token,
            self._cls_id: self._cls_token,
            self._sep_id: self._sep_token,
            self._start_token_id: self._start_token,
            self._end_token_id: self._end_token,
            self._unknown_token_id: self._unknown_token,
        }

    def fit(self, corpus: List[str]) -> Self:
        self._init_vocab()
        flat_corpus = "\n".join(corpus)
        for char in set(flat_corpus):
            if char in self._vocab.values():
                continue
            self._vocab[len(self._vocab)] = char
        return self

    def tokenize_text(self, text: str | List[str]) -> List[str] | List[List[str]]:
        if isinstance(text, str):
            return self._tokenize_text(text)
        assert isinstance(text, list), "`text` should be str or List[str]"
        return [self._tokenize_text(chunk) for chunk in text]

    def tokenize_ids(self, text: str | List[str]) -> List[int] | List[List[int]]:
        if isinstance(text, str):
            return self._tokenize_ids(text)
        assert isinstance(text, list), "`text` should be str or List[str]"
        return [self._tokenize_ids(chunk) for chunk in text]

    def decode(self, tokens: List[int]) -> str:
        content = []
        for token in tokens:
            if token in [self._padding_id, self._cls_id, self._sep_id, self._start_token_id, self._end_token_id, self._unknown_token_id]:
                continue
            content.append(
                self._vocab.get(token, self._unknown_token)
            )
        return "".join(content)

    def _tokenize_text(self, text: str) -> List[str]:
        tokens = [self._start_token]
        reverse_vocab = self.reverse_vocab
        for char in list(text):
            if char in reverse_vocab:
               tokens.append(char)
            else:
                tokens.append(self._unknown_token)
        tokens.append(self._end_token)
        return tokens

    def _tokenize_ids(self, text: str) -> List[int]:
        tokens = self._tokenize_text(text)
        reversed_vocab = self.reverse_vocab
        tokens_ids = [reversed_vocab[token] for token in tokens]
        return tokens_ids

class SimpleTextDataset(Dataset):
    def __init__(
        self,
        corpus: List[str],
        fitted_tokenizer: CharTokenizer,
        max_seq_length: int = 100,
    ):
        self._data: List[List[int]] = []
        for sentence in corpus:
            x = fitted_tokenizer.tokenize_ids(sentence[:max_seq_length - 2])
            self._data.append(x)
            
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return torch.LongTensor(self._data[idx])

def collate(data: List[torch.Tensor]):
    x = [torch.LongTensor(seq) for seq in data]
    return nn.utils.rnn.pad_sequence(x, batch_first=True)

def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to megabytes
    total_memory_mb = total_memory_bytes / (1024**2)

    return total_memory_mb