import os
import sys
import math
import time
import contextlib
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model_50M import GPT, GPTConfig
from dataloader_wikitext import DataLoaderWikitext

CFG = dict(
    total_steps    = 1000,      
    mini_batch_size= 4,         
    grad_accum     = 1,
    max_lr         = 3e-4,
    min_lr         = 3e-5,
    warmup_steps   = 50,
    weight_decay   = 0.1,
    eval_freq      = 250,        
    data_dir  = os.path.join(ROOT, "data", "wikitext103"),
    log_dir   = os.path.join(ROOT, "logs"),
    ckpt_name = "model_50M_wikitext.pt",
)

LOG_DIR   = CFG["log_dir"]
DATA_DIR  = CFG["data_dir"]
CKPT_PATH = os.path.join(LOG_DIR, CFG["ckpt_name"])

os.makedirs(LOG_DIR,  exist_ok=True)

def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / (total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"\n[device] Using device: {device}")
    torch.manual_seed(1337)
    np.random.seed(1337)
    B   = CFG["mini_batch_size"]
    T   = 512 # matching model context length
    grad_accum = CFG["grad_accum"]
    print(f"[batch] mini_batch={B}, context={T}, grad_accum={grad_accum}")
    train_loader = DataLoaderWikitext(B=B, T=T, split="train", data_root=DATA_DIR)
    val_loader   = DataLoaderWikitext(B=B, T=T, split="val",   data_root=DATA_DIR)
    config = GPTConfig() # defaults to 50M
    model = GPT(config).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=CFG["weight_decay"],
        lr=CFG["max_lr"],
        device_type=device_type,
        master_process=True,
    )
    total_steps = CFG["total_steps"]
    eval_freq   = CFG["eval_freq"]

    print(f"\n{'='*60}")
    print(f"  Starting training: {total_steps} steps")
    print(f"  Checkpoint       : {CKPT_PATH}")
    print(f"{'='*60}\n")

    wall_start = time.time()

    for step in range(total_steps):
        is_last = (step == total_steps - 1)
        t0 = time.time()
        if step % eval_freq == 0 or is_last:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(10):
                    inp, tar = val_loader.next_batch()
                    inp, tar = inp.to(device), tar.to(device)
                    ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) \
                          if device_type == "cuda" else contextlib.nullcontext()
                    with ctx:
                        _, loss = model(inp, tar)
                    val_loss += loss.detach() / 10
            val_loss = val_loss.item()
            print(f"\n[step {step:5d}] val_loss={val_loss:.4f}\n")
        model.train()
        optimizer.zero_grad()
        batch_loss = 0.0

        for mini in range(grad_accum):
            inp, tar = train_loader.next_batch()
            inp, tar = inp.to(device), tar.to(device)
            ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) \
                  if device_type == "cuda" else contextlib.nullcontext()
            with ctx:
                _, loss = model(inp, tar)
            loss = loss / grad_accum
            batch_loss += loss.detach()
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, CFG["warmup_steps"], total_steps, CFG["max_lr"], CFG["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = (time.time() - t0) * 1000
        tps = (B * T * grad_accum) / (dt / 1000)
        print(f"step {step:5d} | loss {batch_loss.item():.4f} | lr {lr:.2e} | dt {dt:.0f}ms | tok/s {tps:.1f}")
        if is_last or step > 0 and step % 5000 == 0:
            ckpt = {
                "model":     model.state_dict(),
                "config":    model.config,
                "step":      step,
            }
            torch.save(ckpt, CKPT_PATH)
            print(f"[done] Saved checkpoint -> {CKPT_PATH}")

    print(f"Training complete in {(time.time() - wall_start) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
