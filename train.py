"""Train một biến thể U-Net trên ISBI 2012 EM (CPU).

Dữ liệu: 30 lát cắt liên tiếp 512x512. Chia theo thứ tự lát cắt để tránh rò rỉ
giữa các lát gần nhau: lát 0-23 train, 24-29 val.
Foreground = màng tế bào (pixel đen trong label).

Usage: python train.py <variant> [--seed 0] [--steps 1000] [--crop 128] [--bs 4]
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import unet_ablation as U

DATA = "data/src/data/membrane/train"


def load_split():
    def load(i):
        im = np.array(Image.open(f"{DATA}/image/{i}.png"), dtype=np.float32) / 255.0
        lb = (np.array(Image.open(f"{DATA}/label/{i}.png")) < 128).astype(np.float32)  # 1 = màng
        return im, lb
    tr = [load(i) for i in range(24)]
    va = [load(i) for i in range(24, 30)]
    return tr, va


def sample_batch(train, bs, crop, rng):
    xs, ys = [], []
    for _ in range(bs):
        im, lb = train[rng.integers(len(train))]
        r, c = rng.integers(0, 512 - crop + 1, size=2)
        a, b = im[r:r + crop, c:c + crop], lb[r:r + crop, c:c + crop]
        k = rng.integers(4)                    # augment: xoay 90 độ + lật
        a, b = np.rot90(a, k), np.rot90(b, k)
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        xs.append(a.copy()); ys.append(b.copy())
    x = torch.from_numpy(np.stack(xs))[:, None]
    y = torch.from_numpy(np.stack(ys))[:, None]
    return (x - 0.5) / 0.5, y


@torch.no_grad()
def evaluate(model, val):
    model.eval()
    tp = fp = fn = tn = 0
    loss = 0.0
    for im, lb in val:
        x = (torch.from_numpy(im)[None, None] - 0.5) / 0.5
        y = torch.from_numpy(lb)[None, None]
        logit = model(x)
        loss += F.binary_cross_entropy_with_logits(logit, y).item()
        p = (logit > 0).float()
        tp += (p * y).sum().item(); fp += (p * (1 - y)).sum().item()
        fn += ((1 - p) * y).sum().item(); tn += ((1 - p) * (1 - y)).sum().item()
    model.train()
    return dict(val_loss=loss / len(val),
                dice=2 * tp / (2 * tp + fp + fn),
                iou=tp / (tp + fp + fn),
                pixel_error=(fp + fn) / (tp + fp + fn + tn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    a = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train, val = load_split()

    model = U.build(a.variant)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    hist, t0, run_loss = [], time.perf_counter(), 0.0
    for step in range(1, a.steps + 1):
        x, y = sample_batch(train, a.bs, a.crop, rng)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run_loss += loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, val)
            m.update(step=step, train_loss=run_loss / a.eval_every,
                     elapsed_min=round((time.perf_counter() - t0) / 60, 1))
            run_loss = 0.0
            hist.append(m)
            print(a.variant, a.seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)

    os.makedirs("results", exist_ok=True)
    out = dict(variant=a.variant, seed=a.seed, args=vars(a),
               params=sum(p.numel() for p in model.parameters()),
               final=hist[-1], history=hist)
    json.dump(out, open(f"results/{a.variant}_s{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
