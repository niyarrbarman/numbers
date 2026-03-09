"""Download tulu-3-sft-personas-math-grade from HuggingFace.

Run on a login node (with internet) inside apptainer:
  apptainer shell --bind /tmpdir,/work <image>
  python download_tulu.py
"""

import os
from datasets import load_dataset

OUT_DIR = "/tmpdir/m24047brmn/numbers/data/tulu3_math_grade/raw"

print("Downloading allenai/tulu-3-sft-personas-math-grade...")
ds = load_dataset("allenai/tulu-3-sft-personas-math-grade")

os.makedirs(OUT_DIR, exist_ok=True)
ds.save_to_disk(OUT_DIR)

print(f"\nSaved to {OUT_DIR}")
for split in ds:
    print(f"  {split}: {len(ds[split])} examples")
    if len(ds[split]) > 0:
        ex = ds[split][0]
        print(f"    columns: {list(ex.keys())}")
        if 'messages' in ex:
            for msg in ex['messages'][:2]:
                preview = msg['content'][:80] + ('...' if len(msg['content']) > 80 else '')
                print(f"    {msg['role']}: {preview}")
