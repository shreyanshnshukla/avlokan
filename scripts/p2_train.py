"""Phase 2, step 2: train the built-up U-Net.

    python scripts/p2_train.py --data data/p2/indore_2021.npz --epochs 30

Uses the GPU if PyTorch can see it. Saves models/builtup_unet.pt and
models/metrics.json. On a 4 GB RTX 3050 keep --batch at 8 or below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/p2/indore_2021.npz")
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=24, help="U-Net width; 16 if you run out of GPU memory")
    a = p.parse_args()
    try:
        from avlokan.construction.train import train
    except ImportError:
        print("PyTorch isn't installed. See requirements-ml.txt.")
        return 1
    m = train(a.data, a.out, epochs=a.epochs, batch_size=a.batch, lr=a.lr, base=a.base)
    print(json.dumps({k: v for k, v in m.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
