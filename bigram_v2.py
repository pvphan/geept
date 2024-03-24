"""
Following along with Andrej Karpathy's video: "Let's build GPT: from scratch, in code, spelled out."
(link: https://www.youtube.com/watch?v=kCc8FmEb1nY)
"""
import time
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeePT:
    def __init__(self, vocabulary: str, device: str):
        # A list of the valid tokens. The position of the token
        #   is that tokens encoded value.
        self._vocabulary = vocabulary
        self._device = device

    def encode(self, chars: str) -> torch.Tensor:
        """
        Applies a simple encoding of characters to integers.
        """
        encoded = torch.tensor(
            [self._vocabulary.index(char) for char in chars],
            device=self._device,
        )
        return encoded

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Applies a simple decoding from a list of integers to characters.
        """
        return "".join([self._vocabulary[t] for t in tokens])


class Dataset:
    def __init__(
        self,
        gpt: GeePT,
        text: str,
        train_ratio: float,
        device: str,
        batch_size: int,
        block_size: int,
    ) -> None:
        data = gpt.encode(text)
        N = int(train_ratio * len(data))
        self.train_data = data[:N]
        self.val_data = data[N:]
        self._device = device
        self._batch_size = batch_size
        self._block_size = block_size

    def getBatch(
        self,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a small batch of inputs x and targets y.

        split -- "train" or "val".
        batch_size -- the number of independent sequences to process in parallel.
        block_size -- the maximum context length.
        """
        if split not in {"train", "val"}:
            raise ValueError("Split must be 'train' or 'val'")

        data = self.train_data if split == "train" else self.val_data
        random_idxs = torch.randint(len(data) - self._block_size, (self._batch_size,))
        x = torch.stack([data[i : i + self._block_size] for i in random_idxs])
        y = torch.stack([data[i + 1 : i + self._block_size + 1] for i in random_idxs])
        x = x.to(self._device)
        y = y.to(self._device)
        return x, y


def createVocabulary(text: str) -> str:
    """
    Condense a string into the sorted set of characters
    that make up that string.
    """
    return "".join(sorted(list(set(text))))


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, num_embed_dims: int, depth_factor: int, dropout_ratio: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed_dims, depth_factor * num_embed_dims),
            nn.ReLU(),
            nn.Linear(depth_factor * num_embed_dims, num_embed_dims),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation.
    """

    def __init__(
        self,
        block_size: int,
        num_embed_dims: int,
        num_heads: int,
        depth_factor: int,
        dropout_ratio: float,
    ):
        super().__init__()
        head_size = num_embed_dims // num_heads
        self._self_attention = MultiHeadAttention(
            num_embed_dims, block_size, head_size, num_heads, dropout_ratio
        )
        self._feed_forward = FeedForward(num_embed_dims, depth_factor, dropout_ratio)
        self._layer_norm1 = nn.LayerNorm(num_embed_dims)
        self._layer_norm2 = nn.LayerNorm(num_embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._self_attention(self._layer_norm1(x))
        x = x + self._feed_forward(self._layer_norm2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_embed_dims: int,
        num_heads: int,
        num_layers: int,
        depth_factor: int,
        dropout_ratio: float,
        device: str,
    ):
        head_size = num_embed_dims // num_heads
        super().__init__()
        print(f"{vocab_size=}")
        print(f"{block_size=}")
        print(f"{num_embed_dims=}")
        print(f"{num_heads=}")
        print(f"{head_size=}")
        print(f"{device=}")
        self._num_embed_dims = num_embed_dims
        self._vocab_size = vocab_size
        self._block_size = block_size
        self._token_embedding_table = nn.Embedding(vocab_size, num_embed_dims)
        self._position_embedding_table = nn.Embedding(block_size, num_embed_dims)
        self._lm_head = nn.Linear(num_embed_dims, vocab_size)
        self._blocks = nn.Sequential(
            *[
                TransformerBlock(
                    block_size=block_size,
                    num_embed_dims=num_embed_dims,
                    num_heads=num_heads,
                    depth_factor=depth_factor,
                    dropout_ratio=dropout_ratio)
                for _ in range(num_layers)
            ]
        )
        self._device = device
        self._layer_norm = nn.LayerNorm(num_embed_dims)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        B, T = idx.shape
        C = self._num_embed_dims

        token_embeddings = self._token_embedding_table(idx)  # (B, T, C)
        assert token_embeddings.shape == (B, T, C)

        position_embeddings = self._position_embedding_table(
                torch.arange(T, device=self._device)
        )
        assert position_embeddings.shape == (T, C)

        x = token_embeddings + position_embeddings
        assert x.shape == (B, T, C)

        x = self._blocks(x)
        assert x.shape == (B, T, C)

        x = self._layer_norm(x)
        assert x.shape == (B, T, C)

        logits = self._lm_head(x)
        assert logits.shape == (B, T, self._vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cropped = idx[:, -self._block_size:]
            # get the predictions
            logits, _ = self(idx_cropped)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def generateSequence(
    gpt: GeePT, model: nn.Module, sequence_length: int, device: str
) -> str:
    initial_token = torch.zeros((1, 1), device=device, dtype=torch.long)
    generated_sequence = gpt.decode(
        model.generate(initial_token, max_new_tokens=sequence_length)[0].tolist()
    )
    return generated_sequence


@torch.no_grad()
def estimateLoss(
    model: nn.Module,
    dataset: Dataset,
    eval_iters: int,
) -> Dict[str, torch.Tensor]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.getBatch(split=split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    should_use_cuda = True and torch.cuda.is_available()
    device = "cuda" if should_use_cuda else "cpu"
    with open("input.txt", "r") as f:
        text = f.read()
    vocabulary = createVocabulary(text)
    gpt = GeePT(vocabulary, device)

    train_ratio = 0.9
    batch_size = 64
    block_size = 256
    dataset = Dataset(
        gpt=gpt,
        text=text,
        train_ratio=train_ratio,
        device=device,
        batch_size=batch_size,
        block_size=block_size,
    )

    vocab_size = len(vocabulary)
    num_embed_dims = 384
    num_heads = 6
    num_layers = 6
    depth_factor = 4
    dropout_ratio = 0.2
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        num_embed_dims=num_embed_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        depth_factor=depth_factor,
        dropout_ratio=dropout_ratio,
        device=device,
    )
    model.to(device)
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    sequence_length = 100

    ts_training = time.time()
    num_epochs = 5_000
    eval_interval = 100
    eval_iters = 200
    for iter in range(num_epochs):
        if iter % eval_interval == 0:
            losses = estimateLoss(model, dataset, eval_iters)
            print(
                f"Step {iter} ({int(time.time() - ts_training)}s): train loss {losses['train']:0.4f}, "
                f"val loss {losses['val']:0.4f}"
            )
            generated_sequence = generateSequence(gpt, model, sequence_length, device)
            print(f"Generated sequence, epoch={iter}:")
            print(f"{bcolors.OKCYAN}{generated_sequence}{bcolors.RESET}")

        # sample a batch of data
        xb, yb = dataset.getBatch(split="train")

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    td_training = time.time() - ts_training
    print(f"Training took {td_training:0.3f} sec, loss={loss.item():0.5f}.")
    generated_sequence = generateSequence(gpt, model, sequence_length, device)
    print("Generated sequence, post-training:")
    print(bcolors.OKCYAN + generated_sequence)


# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def contextAveraging():
    B, T, C = 4, 8, 2  # batch, time, channels
    x = torch.randn(B, T, C)

    # Version 1
    xbow_loop = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, : t + 1]  # (t, C)
            xbow_loop[b, t] = torch.mean(xprev, 0)

    # Version 2: broadcasting
    w = torch.tril(torch.ones(T, T))
    w /= w.sum(1, keepdim=True)
    xbow_vec = w @ x  # (B, T, T) @ (B, T, C) --> (B, T, C)
    assert torch.allclose(xbow_loop, xbow_vec, atol=1e-7)

    # Version 3: use Softmax
    tril = torch.tril(torch.ones(T, T))
    w = torch.zeros((T, T))
    w = w.masked_fill(tril == 0, float("-inf"))
    w = F.softmax(w, dim=-1)
    xbow_softmax = w @ x
    assert torch.allclose(xbow_loop, xbow_softmax, atol=1e-7)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_embed_dims: int,
        block_size: int,
        head_size: int,
        num_heads: int,
        dropout_ratio: float,
    ):
        """
        A multi-head attention block.
        """
        super().__init__()
        self._heads = nn.ModuleList([
            SingleHeadAttention(num_embed_dims, block_size, head_size)
            for _ in range(num_heads)
        ])
        self._proj = nn.Linear(num_embed_dims, num_embed_dims)
        self._dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self._heads], dim=-1)
        out = self._dropout(self._proj(out))
        return out


class SingleHeadAttention(nn.Module):
    def __init__(self, num_embed_dims: int, block_size: int, head_size: int):
        """
        A single self-attention head.
        """
        super().__init__()
        C = num_embed_dims
        T = block_size
        H = head_size
        self._head_size = head_size
        self._key = nn.Linear(C, H, bias=False)
        self._query = nn.Linear(C, H, bias=False)
        self._value = nn.Linear(C, H, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(T, T)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self._head_size
        B, T, C = x.shape

        k = self._key(x)
        assert k.shape == (B, T, H)

        q = self._query(x)
        assert q.shape == (B, T, H)

        # compute affinities, with scaling
        w = q @ k.transpose(-2, -1) * C**-0.5
        assert w.shape == (B, T, T)

        # make sure the future doesn't communicate with the past
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        assert w.shape == (B, T, T)
        w = F.softmax(w, dim=-1)
        assert w.shape == (B, T, T)

        v = self._value(x)
        assert v.shape == (B, T, H)

        out = w @ v # (B, T, T) @ (B, T, H) --> (B, T, H)
        assert out.shape == (B, T, H)
        return out


def selfAttention():
    # Version 4: self-attention (single head)
    B, T, C = 4, 8, 32  # batch, time, channels
    H = 16 # head_size
    x = torch.randn(B, T, C)
    tril = torch.tril(torch.ones(T, T))
    query = nn.Linear(C, H, bias=False)
    key = nn.Linear(C, H, bias=False)
    value = nn.Linear(C, H, bias=False)

    # query: "the information I'm interested in"
    q = query(x) # (B, T, 16)
    # key: "the information I have"
    k = key(x) # (B, T, 16)
    # value: "if you find me interesting, here's the information I'll communicate
    #   to you". this is the thing that gets aggregated
    v = value(x) # (B, T, 16)

    # compute "affinities" as w
    w = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) --> (B, T, T) "affinities"
    w = w.masked_fill(tril == 0, float("-inf"))
    w = F.softmax(w, dim=-1)

    # apply the weights to the vector we aggregate, not the raw weights
    out = w @ v # (B, T, T) @ (B, T, 16) --> (B, T, 16)
    # think of x as private information to a particular token
    assert out.shape == (B, T, H)


if __name__ == "__main__":
    torch.manual_seed(1337)
    train()
    # selfAttention()
