"""
Following along with Andrej Karpathy's video: "Let's build GPT: from scratch, in code, spelled out."
(link: https://www.youtube.com/watch?v=kCc8FmEb1nY)
"""
from typing import List
# import torch


class GeePT:
    def __init__(self, vocabulary: str):
        # A list of the valid tokens. The position of the token
        #   is that tokens encoded value.
        self._vocabulary = vocabulary

    def encode(self, chars: str) -> List[int]:
        """
        Applies a simple encoding of characters to integers.
        """
        return [self._vocabulary.index(char) for char in chars]


    def decode(self, tokens: List[int]) -> str:
        """
        Applies a simple decoding from a list of integers to characters.
        """
        return "".join([self._vocabulary[t] for t in tokens])


def createVocabulary(text: str) -> str:
    """
    Condense a string into the sorted set of characters
    that make up that string.
    """
    return "".join(sorted(list(set(text))))


def main():
    with open("input.txt", "r") as f:
        text = f.read()
    vocabulary = createVocabulary(text)
    gpt = GeePT(vocabulary)
    print(f"{vocabulary=}")
    print(f"{gpt.encode('hii there')=}")
    print(f"{gpt.decode(gpt.encode('hii there'))=}")


if __name__ == "__main__":
    main()
