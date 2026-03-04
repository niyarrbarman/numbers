"""
Data preparation for multi-position number-aware GPT-2 training.

Tokenizes text using GPT-2 BPE, extracts numbers via regex and replaces
each with NUM_POSITIONS consecutive <NUM> tokens (id 50257). Produces
three parallel binary files:
  - {split}.bin      : uint16 token IDs (with NUM_TOKEN_ID at number positions)
  - {split}_nums.bin : float32 values (float at NUM positions, 0.0 elsewhere)
  - {split}_pos.bin  : int8 position indices (-1 for non-NUM, 0..k-1 for positions)

Multi-position projection: instead of compressing each number into a single
token position, we project the encoder output to k positions, giving the
transformer multiple attention targets per number.
"""

import os
import re
import math
import pickle
from decimal import Decimal, InvalidOperation

import numpy as np

# Number of workers for .map() and load_dataset()
num_proc = 8
num_proc_load_dataset = num_proc

# Lazy tiktoken initialization — compute nodes may not have internet access
_enc = None

def _get_enc():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("gpt2")
    return _enc

NUM_TOKEN_ID = 50257  # GPT-2 vocab is 0..50256 (50256=EOT); 50257=<NUM>

# Multi-position: each input number occupies k consecutive <NUM> positions.
NUM_POSITIONS = 5

# =============================================================================
# SME (Sign-Mantissa-Exponent) token constants
# =============================================================================
# Variable-length SME grammar:
#   [SIGN] [EXP] [D0]...[Dk] [END], with 1 <= k <= SME_MAX_DIGITS
# Token IDs sit in the padded vocab range (50258-50303, all within uint16).
# Layout:
#   50258-50259 : sign (S+, S-)
#   50260-50278 : exponent (E-9..E9)
#   50279-50288 : digits (D0..D9)
#   50289       : END

SME_SIGN_POS = 50258     # positive / zero
SME_SIGN_NEG = 50259     # negative

SME_EXP_BASE = 50260     # exponent tokens start here
SME_EXP_OFFSET = 9       # E-9 = 50260, E0 = 50269, E9 = 50278
SME_EXP_MIN = -9
SME_EXP_MAX = 9
SME_N_EXP = SME_EXP_MAX - SME_EXP_MIN + 1  # 19

SME_DIGIT_BASE = 50279   # D0 = 50279, D9 = 50288
SME_END = 50289          # END token for variable-length mantissa

SME_MAX_DIGITS = 15
SME_MIN_DIGITS = 1

# Backward-compatible aliases used in diagnostics code.
SME_N_DIGITS = SME_MAX_DIGITS
SME_TOKENS_PER_NUM = 2 + SME_MAX_DIGITS + 1  # S + E + up to 15 digits + END

# All SME token IDs (for quick membership check)
SME_ALL_TOKENS = set(range(SME_SIGN_POS, SME_END + 1))  # 50258-50289


def is_sme_sign_token(tok):
    return tok == SME_SIGN_POS or tok == SME_SIGN_NEG


def is_sme_exp_token(tok):
    return SME_EXP_BASE <= tok < SME_EXP_BASE + SME_N_EXP


def is_sme_digit_token(tok):
    return SME_DIGIT_BASE <= tok <= SME_DIGIT_BASE + 9


def is_sme_token(tok):
    return SME_SIGN_POS <= tok <= SME_END


def _zero_sme_tokens(sign_token=SME_SIGN_POS, min_digits: int = SME_MIN_DIGITS):
    """Return an SME token sequence representing 0 with optional digit padding."""
    min_digits = max(1, int(min_digits))
    return (
        [sign_token, SME_EXP_BASE + SME_EXP_OFFSET]
        + [SME_DIGIT_BASE + 0] * min_digits
        + [SME_END]
    )


def number_to_sme_tokens(value, max_digits=SME_MAX_DIGITS, min_digits=SME_MIN_DIGITS):
    """Convert a scalar number to a list of SME token IDs.

    Returns variable-length tokens:
      [sign, exponent, d0, d1, ..., dK, END]
    with 1 <= K <= max_digits.

    Examples:
        42   -> [S+, E1, D4, D2, END]
       -17   -> [S-, E1, D1, D7, END]
      3.14   -> [S+, E0, D3, D1, D4, END]
       0.5   -> [S+, E-1, D5, END]
         0   -> [S+, E0, D0, END]
    """
    if max_digits < 1:
        raise ValueError("max_digits must be >= 1")
    max_digits = min(int(max_digits), SME_MAX_DIGITS)
    min_digits = max(1, int(min_digits))
    min_digits = min(min_digits, max_digits)

    # Parse and sanitize value.
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = 0.0
    if not math.isfinite(val):
        val = 0.0

    sign_token = SME_SIGN_NEG if val < 0 else SME_SIGN_POS
    abs_val = abs(val)

    if abs_val == 0 or abs_val < 10 ** (SME_EXP_MIN):
        return _zero_sme_tokens(sign_token=sign_token, min_digits=min_digits)

    # Stable float -> decimal conversion with capped significant digits.
    try:
        dec = Decimal(format(abs_val, f".{max_digits}g")).normalize()
    except (InvalidOperation, ValueError):
        return _zero_sme_tokens(sign_token=sign_token, min_digits=min_digits)

    if dec.is_zero():
        return _zero_sme_tokens(sign_token=sign_token, min_digits=min_digits)

    tup = dec.as_tuple()
    digits = list(tup.digits)
    if not digits:
        return _zero_sme_tokens(sign_token=sign_token, min_digits=min_digits)

    # IMPORTANT: exp is computed from the *un-padded* significant-digit length.
    # If we padded before computing exp, we'd shift the numeric value by 10^(pad).
    exp = int(tup.exponent + len(digits) - 1)

    # Saturate rare out-of-range values.
    if exp > SME_EXP_MAX:
        exp_token = SME_EXP_BASE + (SME_EXP_MAX + SME_EXP_OFFSET)
        max_digit_tokens = [SME_DIGIT_BASE + 9] * max_digits
        return [sign_token, exp_token] + max_digit_tokens + [SME_END]
    if exp < SME_EXP_MIN:
        return _zero_sme_tokens(sign_token=sign_token, min_digits=min_digits)

    exp_token = SME_EXP_BASE + (exp + SME_EXP_OFFSET)
    if len(digits) < min_digits:
        digits = digits + [0] * (min_digits - len(digits))
    digit_tokens = [SME_DIGIT_BASE + int(d) for d in digits[:max_digits]]
    return [sign_token, exp_token] + digit_tokens + [SME_END]


def parse_sme_number_tokens(tokens, start_idx=0, max_digits=SME_MAX_DIGITS):
    """Parse one SME number from a token stream.

    Args:
        tokens: sequence of token ids
        start_idx: index where a sign token is expected
        max_digits: maximum digits to consume before implicit END

    Returns:
        (parsed_tokens, next_idx), where parsed_tokens includes END.
        Returns (None, start_idx + 1) if parsing fails.
    """
    if max_digits < 1:
        raise ValueError("max_digits must be >= 1")
    max_digits = min(int(max_digits), SME_MAX_DIGITS)

    n = len(tokens)
    if start_idx + 2 > n:
        return None, start_idx + 1

    sign_tok = int(tokens[start_idx])
    exp_tok = int(tokens[start_idx + 1])
    if not is_sme_sign_token(sign_tok) or not is_sme_exp_token(exp_tok):
        return None, start_idx + 1

    out = [sign_tok, exp_tok]
    idx = start_idx + 2
    n_digits = 0

    while idx < n and n_digits < max_digits:
        tok = int(tokens[idx])
        if is_sme_digit_token(tok):
            out.append(tok)
            n_digits += 1
            idx += 1
            continue
        if tok == SME_END:
            if n_digits == 0:
                return None, start_idx + 1
            out.append(SME_END)
            return out, idx + 1
        break

    # Auto-append END at cap if the model did not emit it.
    if n_digits == max_digits:
        out.append(SME_END)
        return out, idx

    return None, start_idx + 1


def sme_tokens_to_number(tokens):
    """Convert a list of SME token IDs back to a scalar number.

    Args:
        tokens: list of SME tokens for one number or a stream starting with one

    Returns:
        float value, or None if tokens are invalid
    """
    parsed, _ = parse_sme_number_tokens(tokens, start_idx=0, max_digits=SME_MAX_DIGITS)
    if parsed is None:
        return None

    sign_tok = parsed[0]
    exp_tok = parsed[1]
    d_toks = parsed[2:-1]  # drop END

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
        if not is_sme_digit_token(dt):
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


def process_text_with_numbers(text, num_positions=NUM_POSITIONS):
    """Convert text to token IDs, number values, and position indices.

    Numbers detected by NUMBER_PATTERN are replaced with num_positions
    consecutive NUM_TOKEN_ID tokens, each carrying the same float value.
    This gives the transformer k attention targets per input number.

    Returns:
        ids:         list[int]   — token IDs (with k × NUM_TOKEN_ID per number)
        nums:        list[float] — parallel to ids (float value at NUM positions, 0.0 elsewhere)
        pos_indices: list[int]   — parallel to ids (-1 for non-NUM, 0..k-1 for positions)
    """
    enc = _get_enc()
    ids = []
    nums = []
    pos_indices = []

    last_end = 0
    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()

        # Tokenize the text segment before this number
        if start > last_end:
            segment_ids = enc.encode_ordinary(text[last_end:start])
            ids.extend(segment_ids)
            nums.extend([0.0] * len(segment_ids))
            pos_indices.extend([-1] * len(segment_ids))

        # Parse the number value
        num_str = match.group()
        try:
            value = float(num_str)
        except ValueError:
            # If parsing fails, treat as regular text
            segment_ids = enc.encode_ordinary(num_str)
            ids.extend(segment_ids)
            nums.extend([0.0] * len(segment_ids))
            pos_indices.extend([-1] * len(segment_ids))
            last_end = end
            continue

        # Insert k consecutive <NUM> tokens with the same float value
        for pos in range(num_positions):
            ids.append(NUM_TOKEN_ID)
            nums.append(value)
            pos_indices.append(pos)

        last_end = end

    # Tokenize remaining text after the last number
    if last_end < len(text):
        segment_ids = enc.encode_ordinary(text[last_end:])
        ids.extend(segment_ids)
        nums.extend([0.0] * len(segment_ids))
        pos_indices.extend([-1] * len(segment_ids))

    # Append end-of-text token
    ids.append(enc.eot_token)  # 50256
    nums.append(0.0)
    pos_indices.append(-1)

    return ids, nums, pos_indices


def process_example(example):
    """HuggingFace .map() function for a single document."""
    ids, nums, pos_indices = process_text_with_numbers(example['text'])
    return {'ids': ids, 'nums': nums, 'pos_indices': pos_indices, 'len': len(ids)}


if __name__ == '__main__':
    from datasets import load_dataset
    from tqdm import tqdm

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
        pos_filename = os.path.join(os.path.dirname(__file__), f'{split}_pos.bin')

        tok_arr = np.memmap(tok_filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        num_arr = np.memmap(num_filename, dtype=np.float32, mode='w+', shape=(arr_len,))
        pos_arr = np.memmap(pos_filename, dtype=np.int8, mode='w+', shape=(arr_len,))

        total_batches = 1024
        idx = 0
        num_count = 0

        for batch_idx in tqdm(range(total_batches), desc=f'writing {tok_filename}'):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format('numpy')

            tok_batch = np.concatenate(batch['ids'])
            num_batch = np.concatenate(batch['nums'])
            pos_batch = np.concatenate(batch['pos_indices'])

            tok_arr[idx:idx + len(tok_batch)] = tok_batch.astype(np.uint16)
            num_arr[idx:idx + len(num_batch)] = num_batch.astype(np.float32)
            pos_arr[idx:idx + len(pos_batch)] = pos_batch.astype(np.int8)

            num_count += int(np.sum(tok_batch == NUM_TOKEN_ID))
            idx += len(tok_batch)

        tok_arr.flush()
        num_arr.flush()
        pos_arr.flush()

        unique_nums = num_count // NUM_POSITIONS
        print(f"{split}: {arr_len:,} tokens, {unique_nums:,} numbers "
              f"({num_count:,} NUM tokens, {NUM_POSITIONS} positions each)")

    # --- Write meta file ---
    meta = {
        'vocab_size': 50304,
        'num_token_id': NUM_TOKEN_ID,
        'num_positions': NUM_POSITIONS,
    }
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved meta to {meta_path}")
