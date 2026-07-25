from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyset_model import HYSETModel


QUERY = (
    "I'm planning a surprise party for my best friend, and I want to include meaningful quotes "
    "in the decorations. Can you provide me with random love, success, and motivation quotes? "
    "It would be great to have quotes that can celebrate love, success, and inspire everyone "
    "at the party. Thank you so much for your help!"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.getenv("HYSET_CHECKPOINT_PATH", ""))
    parser.add_argument("--encoder", default=os.getenv("HYSET_ENCODER_PATH", ""))
    parser.add_argument("--device", default=os.getenv("HYSET_DEVICE", "cpu"))
    parser.add_argument("--k_pool", type=int, default=100)
    parser.add_argument("--k_return", type=int, default=5)
    parser.add_argument("--query", default=QUERY)
    args = parser.parse_args()

    if not args.ckpt or not args.encoder:
        parser.error("--ckpt and --encoder are required "
                     "(or set HYSET_CHECKPOINT_PATH / HYSET_ENCODER_PATH)")

    model = HYSETModel.load(args.ckpt, args.encoder, device=args.device)
    model.eval()
    print(f"Loaded: N_T={model.N_T}  d={model.d}  m_max={model.m_max}")

    selected = model.retrieve(
        args.query,
        k_pool=args.k_pool,
        k_return=args.k_return,
    )
    print("Selected tools:")
    for fn in selected:
        print(f"  {fn}")


if __name__ == "__main__":
    main()
