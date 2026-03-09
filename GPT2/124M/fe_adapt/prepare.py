"""
Data preparation for Baby Luciole + NumberEncoder adapter.

Uses the luciole_50k SentencePiece tokenizer (NOT GPT-2 tiktoken).
Numbers in input are replaced with a single <NUM> token (id 50256).
Output numbers remain as plain text tokens.

Produces two parallel binary files:
  - {split}.bin     : uint16 token IDs (with NUM_TOKEN_ID at input number positions)
  - {split}_nums.bin: float32 values (float at NUM positions, 0.0 elsewhere)
"""

import re

# Lazy tokenizer initialization
_tokenizer = None
TOKENIZER_PATH = "/work/m24047/m24047brmn/tokenizers/luciole_50k"

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return _tokenizer

# Baby Luciole vocab is 0..50255 (50256 tokens); 50256 = <NUM>
NUM_TOKEN_ID = 50256
EOT_TOKEN_ID = 3  # </s> in SentencePiece

# Match integers, decimals, scientific notation, optionally negative.
NUMBER_PATTERN = re.compile(
    r'(?<![a-zA-Z0-9_])'
    r'-?'
    r'(?:'
        r'\d+\.?\d*'
        r'|\.\d+'
    r')'
    r'(?:[eE][+-]?\d+)?'
    r'(?![a-zA-Z0-9_])'
)


def process_text_with_numbers(text):
    """Convert text to token IDs and parallel number values.

    Numbers detected by NUMBER_PATTERN are replaced with a single NUM_TOKEN_ID.
    Text segments between numbers are tokenized with the luciole_50k tokenizer.

    Returns:
        ids:  list[int]   - token IDs (with NUM_TOKEN_ID for numbers)
        nums: list[float] - parallel to ids (float value at NUM positions, 0.0 elsewhere)
    """
    tokenizer = _get_tokenizer()
    ids = []
    nums = []

    last_end = 0
    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()

        # Tokenize the text segment before this number
        if start > last_end:
            segment = text[last_end:start]
            segment_ids = tokenizer.encode(segment, add_special_tokens=False)
            ids.extend(segment_ids)
            nums.extend([0.0] * len(segment_ids))

        # Parse the number value
        num_str = match.group()
        try:
            value = float(num_str)
        except ValueError:
            segment_ids = tokenizer.encode(num_str, add_special_tokens=False)
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
        segment = text[last_end:]
        segment_ids = tokenizer.encode(segment, add_special_tokens=False)
        ids.extend(segment_ids)
        nums.extend([0.0] * len(segment_ids))

    # Append end-of-sequence token
    ids.append(EOT_TOKEN_ID)
    nums.append(0.0)

    return ids, nums
