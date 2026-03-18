"""Download GSM8K dataset from HuggingFace.

Run this from a node with internet access:
  python3 download_gsm8k.py --out_dir /path/to/data/gsm8k
"""

import os
import json
import argparse


def download(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")

    for split in ["train", "test"]:
        data = list(ds[split])
        out_path = os.path.join(out_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {split}: {len(data)} samples -> {out_path}")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    download(args.out_dir)
