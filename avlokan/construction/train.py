"""Train the built-up U-Net on patches from dataset.make_patches."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import UNet, dice_bce_loss, iou_counts


class PatchSet(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool):
        self.X, self.Y, self.augment = X, Y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        if self.augment:  # satellite images have no "up": flips and 90° turns are free data
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x))
        y = np.ascontiguousarray(y)
        valid = torch.from_numpy(y != 255)
        target = torch.from_numpy((y == 1).astype("float32"))
        return x, target, valid


def train(
    data_npz: str | Path,
    out_dir: str | Path = "models",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    patience: int = 6,
    base: int = 24,
    workers: int = 0,
    log=print,
) -> dict:
    """Train with early stopping on validation IoU. Saves the best model to
    out_dir/builtup_unet.pt (weights + band normalisation) and metrics.json."""
    d = np.load(data_npz)
    X, Y, split = d["X"], d["Y"], d["split"]
    mean, std = d["mean"], d["std"]
    tr, va = split == "train", split == "val"
    log(f"patches: {tr.sum()} train, {va.sum()} val, {X.shape[1]} bands, {X.shape[-1]}px")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else
                            " — no GPU found; training will be slow"))
    model = UNet(in_ch=X.shape[1], base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    tl = DataLoader(PatchSet(X[tr], Y[tr], True), batch_size=batch_size, shuffle=True, num_workers=workers)
    vl = DataLoader(PatchSet(X[va], Y[va], False), batch_size=batch_size, num_workers=workers)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, bad, history = -1.0, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y, v in tl:
            x, y, v = x.to(dev), y.to(dev), v.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = dice_bce_loss(model(x), y, v)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(x)
        sched.step()
        model.eval()
        inter = union = 0
        with torch.no_grad():
            for x, y, v in vl:
                i, u = iou_counts(model(x.to(dev)), y.to(dev), v.to(dev))
                inter, union = inter + i, union + u
        iou = inter / union if union else 0.0
        history.append({"epoch": ep, "train_loss": round(tot / tr.sum(), 4), "val_iou": round(iou, 4)})
        log(f"epoch {ep:2d}  loss {tot / tr.sum():.4f}  val IoU {iou:.3f}  ({time.time() - t0:.0f}s)")
        if iou > best:
            best, bad = iou, 0
            torch.save({"state": model.state_dict(), "in_ch": X.shape[1], "base": base,
                        "mean": mean, "std": std, "bands": list(d["bands"])}, out / "builtup_unet.pt")
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop: no improvement for {patience} epochs")
                break
    metrics = {"best_val_iou": round(best, 4), "epochs_run": len(history), "history": history,
               "device": str(dev), "train_patches": int(tr.sum()), "val_patches": int(va.sum())}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(path: str | Path, device: str | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = UNet(in_ch=ck["in_ch"], base=ck["base"])
    model.load_state_dict(ck["state"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(dev).eval(), ck["mean"], ck["std"], dev


@torch.no_grad()
def predict_probability(model, x_norm: np.ndarray, device, tile: int = 256, overlap: int = 32) -> np.ndarray:
    """Sliding-window inference over a whole (C, Y, X) normalised image.
    Overlapping tiles, keeping only each tile's centre, avoid seams."""
    c, h, w = x_norm.shape
    step = tile - 2 * overlap
    ph = (-h) % step  # pad so the output grid divides evenly into steps
    pw = (-w) % step
    xp = np.pad(x_norm, ((0, 0), (overlap, overlap + ph), (overlap, overlap + pw)), mode="reflect")
    prob = np.zeros((h + ph, w + pw), dtype="float32")
    for r in range(0, h + ph, step):
        for q in range(0, w + pw, step):
            t = torch.from_numpy(xp[None, :, r:r + tile, q:q + tile].copy()).to(device)
            p = torch.sigmoid(model(t))[0].float().cpu().numpy()
            prob[r:r + step, q:q + step] = p[overlap:overlap + step, overlap:overlap + step]
    return prob[:h, :w]
