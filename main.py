"""
Following along with Andrej Karpathy's video: "Let's build GPT: from scratch, in code, spelled out."
(link: https://www.youtube.com/watch?v=kCc8FmEb1nY)
"""
import time
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeePT:
    def __init__(self, vocabulary: str, device):
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
        )
        if self._device:
            encoded = encoded.to(self._device)
        return encoded

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Applies a simple decoding from a list of integers to characters.
        """
        return "".join([self._vocabulary[t] for t in tokens])


class Dataset:
    def __init__(self, gpt: GeePT, text: str, train_ratio: float):
        data = gpt.encode(text)
        N = int(train_ratio * len(data))
        self.train_data = data[:N]
        self.val_data = data[N:]

    def getBatch(
        self, split: str, batch_size: int, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a small batch of inputs x and targets y.

        split -- "train" or "val".
        batch_size -- the number of independent sequences to process in parallel.
        block_size -- the maximum context length.
        """
        if split not in {"train", "test"}:
            raise ValueError("Split must be 'train' or 'test'")

        data = self.train_data if split == "train" else self.val_data
        random_idxs = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in random_idxs])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in random_idxs])
        return x, y


def createVocabulary(text: str) -> str:
    """
    Condense a string into the sorted set of characters
    that make up that string.
    """
    return "".join(sorted(list(set(text))))


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        logits = self._token_embedding_table(idx)  # (B, T, C)
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
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def main():
    torch.manual_seed(1337)

    should_use_cuda = True and torch.cuda.is_available()
    device = torch.device("cuda" if should_use_cuda else "cpu")
    with open("input.txt", "r") as f:
        text = f.read()
    vocabulary = createVocabulary(text)
    gpt = GeePT(vocabulary, device)
    # print(f"{vocabulary=}")
    # print(f"{gpt.encode('hii there')=}")
    # print(f"{gpt.decode(gpt.encode('hii there'))=}")

    train_ratio = 0.9
    dataset = Dataset(gpt, text, train_ratio)

    model = BigramLanguageModel(len(vocabulary))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sequence_length = 100
    generated_sequence = generateSequence(gpt, model, sequence_length)
    print("Generated sequence, pre-training:")
    print(80 * "<")
    print(generated_sequence)
    print(80 * ">")

    ts_training = time.time()
    num_epochs = 10_000
    for step in range(num_epochs):
        # sample a batch of data
        xb, yb = dataset.getBatch(split="train", batch_size=32, block_size=8)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    td_training = time.time() - ts_training
    print(f"Training took {td_training:0.3f} sec, loss={loss.item():0.5f}.")
    generated_sequence = generateSequence(gpt, model, sequence_length)

    print("Generated sequence, post-training:")
    print(80 * "<")
    print(generated_sequence)
    print(80 * ">")


def generateSequence(gpt: GeePT, model: nn.Module, sequence_length: int) -> str:
    initial_token = torch.zeros((1, 1), dtype=torch.long)
    if next(model.parameters()).is_cuda:
        initial_token = initial_token.to("cuda")
    generated_sequence = gpt.decode(
        model.generate(initial_token, max_new_tokens=sequence_length)[0].tolist()
    )
    return generated_sequence


def initializeDevice(model: nn.Module):
    shouldUseCuda = torch.cuda.is_available()
    device = torch.device("cuda" if shouldUseCuda else "cpu")
    return device


if __name__ == "__main__":
    main()
