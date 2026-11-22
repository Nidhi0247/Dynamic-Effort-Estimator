"""E10 (baseline B8): fine-tuned transformer on requirement text (GPT2SP-style), same protocol as E1.

For each rolling origin a pretrained encoder (default gpt2, as in GPT2SP; or roberta-base / deberta-v3) is
fine-tuned end to end on the training window to regress log(hours / SP) from the as-of title + description,
with story points appended as text ("[SP 3]").  Early stopping and the MAE back-transform use the validation
window; the 95% interval uses the same normalised Mondrian conformal recipe as every other E1 model.
Rows are written in the E1 raw format (results/e10_text_<model>_raw.csv) so E1's analysis can merge them.

Needs the CUDA environment:  venv-gpu/Scripts/python.exe e10_text_baseline.py --model gpt2
"""
import time

from common import args, fitted_model, log, origin_splits, save
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from dynest.calibration import mae_shift
from e1_accuracy import baseline_intervals, per_project, with_features


class Regressor(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.last_token = "gpt" in name.lower()
        hs = self.enc.config.hidden_size                     # LayerNorm tames GPT-2's outlier activations
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hs), torch.nn.Dropout(0.1), torch.nn.Linear(hs, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.last_token:                                     # GPT2SP: representation of the last real token
            pooled = h[torch.arange(len(h), device=h.device), mask.sum(1) - 1]
        else:                                                   # encoders: masked mean pooling
            pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)


def texts(df):
    return [f"[SP {sp:g}] {t}" for sp, t in zip(df.sp.values, df.text.fillna("").values)]


def encode(tok, df, max_len):
    e = tok(texts(df), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def predict_z(model, ids, mask, dev, bs=64, lo=-np.inf, hi=np.inf):
    """Standardised predictions, clipped to the range observed in the training targets."""
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(ids), bs):
            out.append(model(ids[i:i + bs].to(dev), mask[i:i + bs].to(dev)).float().cpu())
    return np.clip(torch.cat(out).numpy(), lo, hi)


class TextB:
    """Adapter so E1's interval helper can call .predict(df) -> hours."""
    def __init__(self, name, hours_by_issue):
        self.name, self.h = name, hours_by_issue

    def predict(self, df):
        return df.issue_id.map(self.h).values


def run_origin(k, tr, va, te, a, dev):
    torch.manual_seed(k)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Regressor(a.model).to(dev)
    if model.last_token:
        model.enc.config.pad_token_id = tok.pad_token_id
    y_tr = np.log(tr.effort_h.values / tr.sp.values)
    mu, sd = y_tr.mean(), y_tr.std()
    lo, hi = (y_tr.min() - mu) / sd, (y_tr.max() - mu) / sd
    I_tr, M_tr = encode(tok, tr, a.max_len)
    I_va, M_va = encode(tok, va, a.max_len)
    I_te, M_te = encode(tok, te, a.max_len)
    dl = DataLoader(TensorDataset(I_tr, M_tr, torch.tensor((y_tr - mu) / sd, dtype=torch.float32)),
                    batch_size=a.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best = (np.inf, None, None)
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        for ids, mask, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.l1_loss(model(ids.to(dev), mask.to(dev)).float(), y.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        z_va = predict_z(model, I_va, M_va, dev, lo=lo, hi=hi) * sd + mu
        shift = mae_shift(va.effort_h.values, va.sp.values * np.exp(z_va))
        mae = float(np.abs(va.sp.values * np.exp(z_va + shift) - va.effort_h.values).mean())
        log(f"  origin {k} epoch {ep}: val MAE {mae:.2f} h ({time.time() - t0:.0f}s)")
        if mae < best[0]:
            best = (mae, {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}, shift)
    model.load_state_dict(best[1])
    shift = best[2]
    hours = {}
    for df, I, M in ((va, I_va, M_va), (te, I_te, M_te)):
        z = predict_z(model, I, M, dev, lo=lo, hi=hi) * sd + mu
        log(f"  log-density quantiles (1/50/99%): {np.round(np.quantile(z, [0.01, 0.5, 0.99]), 2)} "
            f"train range [{y_tr.min():.2f}, {y_tr.max():.2f}]")
        hours.update(dict(zip(df.issue_id.values, df.sp.values * np.exp(z + shift))))
    del model
    torch.cuda.empty_cache()
    return hours


def main():
    a = args([(["--model"], dict(default="gpt2")), (["--epochs"], dict(type=int, default=4)),
              (["--lr"], dict(type=float, default=3e-5)), (["--batch"], dict(type=int, default=16)),
              (["--max-len"], dict(type=int, default=256, dest="max_len"))])
    dev = "cuda"
    assert torch.cuda.is_available(), "run with the CUDA environment (venv-gpu)"
    name = f"B8 fine-tuned {a.model.split('/')[-1]}"
    rows = []
    for k, (f, tr, va, te) in enumerate(origin_splits(a.smoke)):
        if a.smoke:
            tr, va, te = tr.sample(800, random_state=0), va.head(300), te.head(300)
        log(f"origin {k}: train={len(tr)} val={len(va)} test={len(te)} model={a.model}")
        hours = run_origin(k, tr, va, te, a, dev)
        b = TextB(name, hours)
        m = fitted_model(k, tr, va, a.backend, a.smoke)              # only for the shared interval features
        VA, TE = with_features(m, va), with_features(m, te)
        pt, lo, hi = baseline_intervals(m, b, VA, TE, seed=k)
        rows += per_project(name, k, te, pt, tr, lo, hi)
    R = pd.DataFrame(rows)
    save(R, f"e10_text_{a.model.split('/')[-1]}{'_smoke' if a.smoke else ''}_raw.csv")
    print(R[R.project_id == "ALL"][["origin", "MAE", "MdAE", "LMAE", "SA", "coverage", "rel_width"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
