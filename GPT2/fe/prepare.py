"""
Data preparation for number-aware GPT-2 training.

Tokenizes text using GPT-2 BPE, extracts numbers via regex and replaces
them with a single <NUM> token (id 50257). Produces two parallel binary files:
  - {split}.bin     : uint16 token IDs (with NUM_TOKEN_ID at number positions)
  - {split}_nums.bin: float32 values (float at NUM positions, 0.0 elsewhere)

The number extraction regex and process function are the only dataset-specific
components. To swap datasets, change load_dataset() and the map function.
"""

import os
import re
import math
import pickle

import numpy as np
import tiktoken
from tqdm import tqdm

# Number of workers for .map() and load_dataset()
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

NUM_TOKEN_ID = 50257  # GPT-2 vocab is 0..50256 (50256=EOT); 50257=<NUM>

# =============================================================================
# SME (Sign-Mantissa-Exponent) token constants
# =============================================================================
# Each number is encoded as exactly 5 tokens: Sign + Exponent + 3 Mantissa Digits
# Token IDs sit in the padded vocab range (50258-50303, all within uint16)

SME_SIGN_POS = 50258     # positive / zero
SME_SIGN_NEG = 50259     # negative

SME_EXP_BASE = 50260     # exponent tokens start here
SME_EXP_OFFSET = 5       # E-5 = 50260, E0 = 50265, E5 = 50270
SME_EXP_MIN = -5
SME_EXP_MAX = 5
SME_N_EXP = SME_EXP_MAX - SME_EXP_MIN + 1  # 11

SME_DIGIT_BASE = 50271   # D0 = 50271, D9 = 50280

SME_N_DIGITS = 3         # mantissa digits per number
SME_TOKENS_PER_NUM = 5   # S + E + D + D + D

# All SME token IDs (for quick membership check)
SME_ALL_TOKENS = set(range(SME_SIGN_POS, SME_DIGIT_BASE + 10))  # 50258-50280


def number_to_sme_tokens(value, n_digits=SME_N_DIGITS):
    """Convert a scalar number to a list of SME token IDs.

    Returns exactly (2 + n_digits) token IDs: [sign, exponent, d0, d1, ...].
    Default n_digits=3 gives 5 tokens per number.

    Examples:
        42   -> [S+, E1, D4, D2, D0]
       -17   -> [S-, E1, D1, D7, D0]
      3.14   -> [S+, E0, D3, D1, D4]
       0.5   -> [S+, E-1, D5, D0, D0]
         0   -> [S+, E0, D0, D0, D0]
    """
    # Sign
    sign_token = SME_SIGN_NEG if value < 0 else SME_SIGN_POS

    # Handle zero
    abs_val = abs(float(value))
    if abs_val == 0 or abs_val < 10 ** (SME_EXP_MIN):
        exp_token = SME_EXP_BASE + (0 + SME_EXP_OFFSET)  # E0
        digit_tokens = [SME_DIGIT_BASE + 0] * n_digits    # D0 D0 D0
        return [sign_token, exp_token] + digit_tokens

    # Exponent: floor(log10(abs_val))
    exp = int(math.floor(math.log10(abs_val)))
    exp = max(SME_EXP_MIN, min(SME_EXP_MAX, exp))
    exp_token = SME_EXP_BASE + (exp + SME_EXP_OFFSET)

    # Mantissa: normalize to [1, 10)
    mantissa = abs_val / (10.0 ** exp)
    # Clamp to valid range (handles floating point edge cases)
    mantissa = max(1.0, min(mantissa, 9.999))

    # Extract digits
    digit_tokens = []
    for _ in range(n_digits):
        d = int(mantissa)
        d = min(d, 9)  # safety clamp
        digit_tokens.append(SME_DIGIT_BASE + d)
        mantissa = (mantissa - d) * 10.0

    return [sign_token, exp_token] + digit_tokens


def sme_tokens_to_number(tokens):
    """Convert a list of SME token IDs back to a scalar number.

    Args:
        tokens: list of 5 ints [sign, exponent, d0, d1, d2]

    Returns:
        float value, or None if tokens are invalid
    """
    if len(tokens) < SME_TOKENS_PER_NUM:
        return None

    sign_tok = tokens[0]
    exp_tok = tokens[1]
    d_toks = tokens[2:2 + SME_N_DIGITS]

    # Parse sign
    if sign_tok == SME_SIGN_POS:
        sign = 1.0
    elif sign_tok == SME_SIGN_NEG:
        sign = -1.0
    else:
        return None

    # Parse exponent
    if not (SME_EXP_BASE <= exp_tok < SME_EXP_BASE + SME_N_EXP):
        return None
    exp = (exp_tok - SME_EXP_BASE) - SME_EXP_OFFSET

    # Parse mantissa digits
    mantissa = 0.0
    for i, dt in enumerate(d_toks):
        if not (SME_DIGIT_BASE <= dt <= SME_DIGIT_BASE + 9):
            return None
        digit = dt - SME_DIGIT_BASE
        mantissa += digit * (10.0 ** (-i))

    # Reconstruct value
    value = sign * mantissa * (10.0 ** exp)
    return value


# Match integers, decimals, scientific notation, optionally negative.
# Handles: 42, 3.14, -7, 1e5, 2.5e-3, .5, -.001
# Does NOT match numbers inside words (h2o, mp3) due to lookaround assertions.
NUMBER_PATTERN = re.compile(
    r'(?<![a-zA-Z0-9_])'           # not preceded by alphanumeric/underscore
    r'-?'                           # optional negative sign
    r'(?:'
        r'\d+\.?\d*'                # integer or decimal: 42, 3.14, 3.
        r'|\.\d+'                   # leading-dot decimal: .5, .001
    r')'
    r'(?:[eE][+-]?\d+)?'           # optional exponent: e5, E-3, e+02
    r'(?![a-zA-Z0-9_])'            # not followed by alphanumeric/underscore
)


def process_text_with_numbers(text):
    """Convert text to token IDs and parallel number values.

    Numbers detected by NUMBER_PATTERN are replaced with a single NUM_TOKEN_ID.
    Text segments between numbers are tokenized normally with GPT-2 BPE.

    Returns:
        ids:  list[int]   — token IDs (with NUM_TOKEN_ID for numbers)
        nums: list[float] — parallel to ids (float value at NUM positions, 0.0 elsewhere)
    """
    ids = []
    nums = []

    last_end = 0
    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()

        # Tokenize the text segment before this number
        if start > last_end:
            segment_ids = enc.encode_ordinary(text[last_end:start])
            ids.extend(segment_ids)
            nums.extend([0.0] * len(segment_ids))

        # Parse the number value
        num_str = match.group()
        try:
            value = float(num_str)
        except ValueError:
            # If parsing fails, treat as regular text
            segment_ids = enc.encode_ordinary(num_str)
            ids.extend(segment_ids)
            nums.extend([0.0] * len(segment_ids))
            last_end = end
            continue

        # Insert single <NUM> token with the float value
        ids.append(NUM_TOKEN_ID)
        nums.append(value)

        last_end = end

    # Tokenize remaining text after the last number
    if last_end < len(text):
        segment_ids = enc.encode_ordinary(text[last_end:])
        ids.extend(segment_ids)
        nums.extend([0.0] * len(segment_ids))

    # Append end-of-text token
    ids.append(enc.eot_token)  # 50256
    nums.append(0.0)

    return ids, nums


def process_example(example):
    """HuggingFace .map() function for a single document."""
    ids, nums = process_text_with_numbers(example['text'])
    return {'ids': ids, 'nums': nums, 'len': len(ids)}


if __name__ == '__main__':
    from datasets import load_dataset

    # --- Load dataset (swap this section for different datasets) ---
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # OpenWebText only has 'train' split; create a val split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    # --- Tokenize with number extraction ---
    tokenized = split_dataset.map(
        process_example,
        remove_columns=['text'],
        desc="tokenizing with number extraction",
        num_proc=num_proc,
    )

    # --- Write to binary files ---
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)

        tok_filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        num_filename = os.path.join(os.path.dirname(__file__), f'{split}_nums.bin')

        tok_arr = np.memmap(tok_filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        num_arr = np.memmap(num_filename, dtype=np.float32, mode='w+', shape=(arr_len,))

        total_batches = 1024
        idx = 0
        num_count = 0

        for batch_idx in tqdm(range(total_batches), desc=f'writing {tok_filename}'):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format('numpy')

            tok_batch = np.concatenate(batch['ids'])
            num_batch = np.concatenate(batch['nums'])

            tok_arr[idx:idx + len(tok_batch)] = tok_batch.astype(np.uint16)
            num_arr[idx:idx + len(num_batch)] = num_batch.astype(np.float32)

            num_count += int(np.sum(tok_batch == NUM_TOKEN_ID))
            idx += len(tok_batch)

        tok_arr.flush()
        num_arr.flush()

        print(f"{split}: {arr_len:,} tokens, {num_count:,} numbers "
              f"({num_count / arr_len * 100:.2f}% of tokens)")

    # --- Write meta file ---
    meta = {
        'vocab_size': 50304,
        'num_token_id': NUM_TOKEN_ID,
    }
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved meta to {meta_path}")
