"""
Data preparation for number-aware GPT-2 with text output decoding.

Tokenizes input text using GPT-2 BPE, extracts numbers via regex and replaces
them with a single <NUM> token (id 50257). Output numbers remain as plain text
tokens (standard BPE encoding).

Produces two parallel binary files:
  - {split}.bin     : uint16 token IDs (with NUM_TOKEN_ID at input number positions)
  - {split}_nums.bin: float32 values (float at NUM positions, 0.0 elsewhere)
"""

import re

# Lazy tiktoken initialization — compute nodes may not have internet access
_enc = None

def _get_enc():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("gpt2")
    return _enc

NUM_TOKEN_ID = 50257  # GPT-2 vocab is 0..50256 (50256=EOT); 50257=<NUM>

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
    enc = _get_enc()
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
