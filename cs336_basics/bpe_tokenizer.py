from abc import ABC
import multiprocessing
import pickle
import regex as re
from collections import defaultdict
from dataclasses import dataclass
import time

import os
from typing import BinaryIO


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


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.stoi = {token: idx for idx, token in self.params.vocab.items()}

    def encode(self, input_string: str) -> list[int]:
        # 1. Handle Special Tokens
        escaped = [re.escape(t) for t in self.params.special_tokens]
        pattern = "|".join(escaped)
        # Split chunk so special tokens become isolated boundaries
        doc_list = re.split(pattern, input_string)

        # 2. Regex Pre-tokenization
        for doc in doc_list:
            # pre-tokenize
            matches = re.finditer(PRE_TOKEN_PAT, doc)
            for m in matches:
                m_str =  m.group()
                assert m_str != EOT_STRING
                token_bytes = m_str.encode("utf-8")
                # Convert bytes object to a tuple of single-byte objects
                # e.g. b'hi' -> (b'h', b'i')
                tokens = tuple(bytes([b]) for b in token_bytes)

                print(f"\n === encode, pre-tokenized tokens: {tokens}")
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
                print(f"\n === encode, merged tokens: {tokens}")


    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _initialize_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    """Creates the initial vocab with special tokens and base 256 bytes."""
    vocab: dict[int, bytes] = {}
    idx = 0
    
    # 1. Add Special Tokens
    for token in special_tokens:
        # Note: We treat special tokens as raw bytes of their UTF-8 encoding here
        # based on your original logic.
        vocab[idx] = token.encode("utf-8") 
        idx += 1
        
    # 2. Add Base 256 Bytes
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
        
    return vocab, idx



def _find_best_pair(token_freq: dict[tuple[bytes, ...], int]) -> tuple[bytes, bytes]:
    """Finds the most frequent adjacent pair of bytes."""
    counts = defaultdict(int)
    for tokens, freq in token_freq.items():
        # Iterate through adjacent pairs in the token sequence
        for pair in zip(tokens, tokens[1:]):
            counts[pair] += freq

    if not counts:
        return None

    # Tie-breaker: Max count, then lexicographically first
    return max(counts, key=lambda k: (counts[k], k))

def _merge_tokens(token_freq: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    """
    Updates the token frequency dictionary by merging the specified pair.
    """
    byte1, byte2 = pair
    new_token = byte1 + byte2
    new_token_freq = defaultdict(int)
    # print("\n ==== _merge_tokens, original size: ", len(token_freq))
    for tokens, freq in token_freq.items():
        # Simple check, not perfect but fast
        if byte1 in tokens and byte2 in tokens:
            # Reconstruct the token sequence with the merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == byte1 and tokens[i+1] == byte2:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            new_token_freq[tuple(new_tokens)] += freq
            # print("\n ==== _merge_tokens, merged: ", tuple(new_tokens))
        else:
            # If the pair isn't in this token, copy and skip
            new_token_freq[tokens] += freq

    # print("\n ==== _merge_tokens, merged size: ", len(new_token_freq))
    return new_token_freq

def process_chunk_worker(args):
    filename, start, end, special_tokens = args
    
    with open(filename, "rb") as f:
        f.seek(start)
        # Read only the assigned slice
        chunk_bytes = f.read(end - start)
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
    
    return _get_initial_token_counts(chunk_str, special_tokens)

def _get_initial_token_counts(text_chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Splits text by special tokens, applies regex pre-tokenization, 
    and returns initial byte-sequence counts.
    """
    token_freq = defaultdict(int)
    
    # 1. Handle Special Tokens (Prevent merging across them)
    escaped = [re.escape(t) for t in special_tokens]
    pattern = "|".join(escaped)
    # Split chunk so special tokens become isolated boundaries
    doc_list = re.split(pattern, text_chunk)
    
    # 2. Regex Pre-tokenization
    for doc in doc_list:
        matches = re.finditer(PRE_TOKEN_PAT, doc)
        for m in matches:
            m_str =  m.group()
            assert m_str != EOT_STRING
            token_bytes = m_str.encode("utf-8")
            # Convert bytes object to a tuple of single-byte objects
            # e.g. b'hi' -> (b'h', b'i')
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            token_freq[token_tuple] += 1

    return token_freq

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 10,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # --- 1. Initialization ---
    vocab, next_idx = _initialize_vocab(special_tokens)
    merges_bytes = []
    
    # --- 2. Process File ---
    with open(input_path, "rb") as f:
        # Assuming find_chunk_boundaries is defined elsewhere
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        worker_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            worker_args.append((input_path, start, end, special_tokens))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results_list = pool.map(process_chunk_worker, worker_args)

        token_freq = defaultdict(int)

        for local_counts in results_list:
            for token_tuple, count in local_counts.items():
                token_freq[token_tuple] += count
        # # save for debug
        # filename = f"token_freq_np{num_processes}.pkl"
        # with open(filename, "wb") as f:
        #     pickle.dump(token_freq, f)

        # --- 3. BPE Training Loop ---
        while len(vocab) < vocab_size:
            # A. Find best pair
            best_pair = _find_best_pair(token_freq)

            # B. Update vocab/merges
            merges_bytes.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            vocab[next_idx] = new_token
            next_idx += 1
            
            # C. Update stats for the next round
            token_freq = _merge_tokens(token_freq, best_pair)
            
    return vocab, merges_bytes



if __name__ == '__main__':
    # input_path = "/Users/xsarah/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = "/Users/xsarah/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    input_path = "/Users/xsarah/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = [EOT_STRING]
    vocab, merges = train_bpe(input_path=input_path, vocab_size=10_000, special_tokens=special_tokens, num_processes=10)
