from abc import ABC
from dataclasses import dataclass
from typing import Iterable, Iterator

import regex as re
import pickle

# pre-tokenization pattern from GPT-2
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
PRE_TOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT_STRING = "<|endoftext|>"


class Tokenizer(ABC):
    """Abstruct interface for a tokenizer."""
    def encode(self, string:str) -> list[int]:
        raise NotImplementedError

    def decode(self, indicies: list[int]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]                 # index -> bytes
    merges: list[tuple[bytes, bytes]]       # bytes1, bytes2
    special_tokens: list[str] | None = None # list of special token strings


    def __post_init__(self):
            # 1. Check if special_tokens exists
            if self.special_tokens is not None:
                
                # 2. Sort the tokens (usually by length descending is best for tokenization regex)
                # If you just want alphabetical, remove the key argument.
                sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
                
                # 3. Use object.__setattr__ to bypass the frozen constraint
                object.__setattr__(self, 'special_tokens', sorted_tokens)

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.stoi = {token: idx for idx, token in self.params.vocab.items()}
        if self.params.special_tokens is not None:
            escaped = [re.escape(t) for t in self.params.special_tokens]
            # keep the delimiters
            self.pattern = "(" + "|".join(escaped) + ")"

    @staticmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """
        Class method that constructs and return a Tokenizer 
        from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens.
        
        :param cls: Description
        :param vocab_filepath: Serialized vocabulary path.
        :param merges_filepath: List of merges
        :param special_tokens: List of speical tokens.
        """
        with open(vocab_filepath, 'r') as vocab_file:
            vocab = pickle.load(vocab_file)
        
        with open(merges_filepath, 'r') as merges_file:
            merges = pickle.load(merges_file)

        params = BPETokenizerParams(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens
        )
        return cls(params)

    def _merge_tokens(self, input_string: str) -> list[int]:
        token_bytes = input_string.encode("utf-8")
        # Convert bytes object to a tuple of single-byte objects
        # e.g. b'hi' -> (b'h', b'i')
        tokens = tuple(bytes([b]) for b in token_bytes)

        # print(f"\n === encode, pre-tokenized tokens: {tokens}")
        # merges
        for pair in self.params.merges:
            byte1, byte2 = pair
            if byte1 in tokens and byte2 in tokens:
                # Reconstruct the token sequence with the merge
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == byte1 and tokens[i+1] == byte2:
                        new_tokens.append(byte1 + byte2)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = tuple(new_tokens)
            else:
                continue
        # after merge, get the encoded indices
        encoded_indices = []
        for token in tokens:
            encoded_indices.append(self.stoi[token])
        return encoded_indices
    
    def _encode_chunk(self, chunk: str) -> list[int]:
        # pre-tokenize
        encoded_indices = []
        matches = re.finditer(PRE_TOKEN_PAT, chunk)
        for m in matches:
            m_str =  m.group()
            assert m_str != EOT_STRING
            encoded_indices += self._merge_tokens(m_str)
        return encoded_indices

    def encode(self, input_string: str) -> list[int]:
        if self.params.special_tokens is None:
            return self._encode_chunk(input_string)

        # Handle Special Tokens
        # split chunk so special tokens become isolated boundaries
        # the delimiters are preserved using the pattern
        chunks = re.split(self.pattern, input_string)
        # print("\n pattern", self.pattern)
        # 2. Regex Pre-tokenization
        encoded_indices = []
        for chunk in chunks:
            # print("\n chunk", chunk)
            # if it's one of the special token
            if chunk in self.params.special_tokens:
                token_bytes = chunk.encode("utf-8")
                encoded_indices.append(self.stoi[token_bytes])
            else:
                encoded_indices += self._encode_chunk(chunk)

        return encoded_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Docstring for encode_iterable
        
        :param self: Description
        :param iterable: Description
        :type iterable: Iterable[str]
        :return: Description
        :rtype: Iterator[int]
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string
