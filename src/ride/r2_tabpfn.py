"""Round 2 TabPFN: one fit per retrieval context, predicting every test row with that key.

Usage (from the repo root). The driver is CPU-only and starts one short GPU worker at a
time; each worker takes the shared GPU lock itself (fcntl.flock on
/tmp/snooze-oracle-gpu.lock) after its imports, works for about --budget seconds, frees
the GPU by exiting, and the driver waits a little before the next worker, so the lock is
held for minutes at a time and never while idle:
    python -m src.ride.r2_tabpfn drive
One worker by hand:
    python -m src.ride.r2_tabpfn work --budget 150

Outputs under data/ride2/: tabpfn/<ctx_key>.parquet (test_id + quantiles), one line per
context in tabpfn_stats.jsonl, and preds_tabpfn.parquet once every context is done.
Finished contexts are skipped on rerun.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

from .common import CATEGORICAL, FEATURES, QUANTILES, TARGET, qcol
from .retrieval import OUT2, context_rows

PER_KEY = OUT2 / "tabpfn"
STATS = OUT2 / "tabpfn_stats.jsonl"
# Redo mode (SNOOZE_R2_REDO=1): rerun on the GPU the contexts that the main run computed
# on the CPU, into separate files, to measure the CPU/GPU difference and, if all finish,
# replace the CPU results.
REDO = os.environ.get("SNOOZE_R2_REDO") == "1"
MAIN_STATS = STATS
if REDO:
    PER_KEY = OUT2 / "tabpfn_gpu_redo"
    STATS = OUT2 / "tabpfn_stats_gpu_redo.jsonl"
LOCK = "/tmp/snooze-oracle-gpu.lock"
# Rows per predict call. Each call re-runs the context through the model, a fixed cost
# of up to ~30 s for a 10k-row context on the shared card, so calls are made as large as
# memory allows; halved on CUDA OOM.
PREDICT_CHUNK = 1000
# On WSL an over-full GPU does not raise OOM: it pages GPU memory to host RAM and a
# 1-second predict took 650 s. So before each context the worker reads the card's free
# memory from nvidia-smi (torch.cuda.mem_get_info misreports it under WSL) and, if there
# is not enough, exits with BUSY so the driver releases the lock and retries later.
# Measured footprint: CUDA context ~300 MiB + weights ~850 MiB + activations; torch
# reserved ~900 + 0.68 * context rows MiB for small contexts, ~3.6 GB at 10k (chunked).
LARGE_CTX = 3000


def need_free_mib(n_ctx: int) -> float:
    return min(1150 + 0.7 * n_ctx, 4000)
# Watchdog: a context that runs longer than this is stuck in paging; the worker kills
# itself (frees GPU and lock) with STUCK and the driver retries later.
MAX_CONTEXT_S = {"small": 60, "large": 300}
BUSY, STUCK = 3, 4
MAX_BACKOFFS = 40
# Some GPU predicts hang with plenty of free memory (the same context ran in 7 s on a
# later try). A context that stalls MAX_GPU_STALLS times is run on the CPU instead, by
# the driver, without the GPU lock. Same model and settings; only the device differs.
MAX_GPU_STALLS = 2
# After the first ~110 contexts the free memory on the shared card shrank and every
# context with >= ~1,000 rows (later >= ~700) stalled; from then on such contexts go to
# the CPU directly.
CPU_MIN_CTX = 700
STALLS = OUT2 / ("tabpfn_stalls_gpu_redo.json" if REDO else "tabpfn_stalls.json")
CPU_THREADS = 4

# Keep one built model per worker instead of rebuilding it from the checkpoint for every
# context (the package's own switch; weights are not changed by inference). The cached
# CPU checkpoint is dropped after the first build to keep host RSS down.
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")


def make_model(device: str = "cuda"):
    from tabpfn import TabPFNRegressor

    # Package defaults (TabPFN-3.5, n_estimators="auto", memory_saving_mode="auto",
    # random_state=0). Only the device and which columns are categorical are set.
    return TabPFNRegressor(device=device,
                           categorical_features_indices=[FEATURES.index(c) for c in CATEGORICAL])


def fit_context(ctx: pd.DataFrame, device: str = "cuda"):
    m = make_model(device)
    m.fit(ctx[FEATURES].to_numpy(np.float64), ctx[TARGET].to_numpy(np.float64))
    from tabpfn import model_loading

    model_loading._load_checkpoint_cached.cache_clear()
    gc.collect()
    return m


def key_tuple(g: pd.DataFrame) -> tuple:
    r = g.iloc[0]
    place = int(r.pu_zone) if r.ctx_level == "route" else str(r.pu_borough)
    return (r.ctx_level, place, int(r.do_zone), int(r.req_hour))


def stalls() -> dict:
    return json.loads(STALLS.read_text()) if STALLS.exists() else {}


def cpu_keys() -> set:
    s = pd.read_json(MAIN_STATS, lines=True).drop_duplicates("ctx_key", keep="last")
    return set(s.loc[s.get("device", pd.Series(index=s.index, dtype=object)) == "cpu",
                     "ctx_key"])


def todo_groups(test: pd.DataFrame, device: str = "cuda"):
    st = stalls()
    redo = cpu_keys() if REDO else None
    for name, g in test.groupby("ctx_key", sort=True):
        if (PER_KEY / f"{name}.parquet").exists():
            continue
        if REDO:
            if device == "cuda" and name in redo and st.get(name, 0) < MAX_GPU_STALLS:
                yield name, g
            continue
        to_cpu = (st.get(name, 0) >= MAX_GPU_STALLS
                  or len(context_rows(key_tuple(g))[0]) >= CPU_MIN_CTX)
        if to_cpu == (device == "cpu"):
            yield name, g


def gpu_free_mib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout.split(",")
    return float(out[0]) - float(out[1])


def run_context(name: str, g: pd.DataFrame, device: str, free_mib: float | None,
                limit: float | None = None) -> None:
    import torch

    cuda = device == "cuda"
    ctx, n_match = context_rows(key_tuple(g))
    finished = threading.Event()

    def watchdog():
        if not finished.wait(limit):
            st = stalls()
            st[name] = st.get(name, 0) + 1
            STALLS.write_text(json.dumps(st, indent=1))
            print(f"worker: {name} exceeded {limit}s, exiting", flush=True)
            import faulthandler
            faulthandler.dump_traceback(all_threads=True)
            os._exit(STUCK)

    if limit:
        threading.Thread(target=watchdog, daemon=True).start()
    if cuda and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = fit_context(ctx, device)
    if cuda:
        torch.cuda.synchronize()
    fit_s = time.time() - t0
    X = g[FEATURES].to_numpy(np.float64)
    chunk, parts, ooms, i = PREDICT_CHUNK, [], 0, 0
    while i < len(X):
        try:
            qs = m.predict(X[i:i + chunk], output_type="quantiles", quantiles=QUANTILES)
        except torch.cuda.OutOfMemoryError:
            ooms += 1
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            continue
        parts.append(np.column_stack(qs))
        i += chunk
    if cuda:
        torch.cuda.synchronize()
    finished.set()
    pred_s = time.time() - t0 - fit_s
    q = np.vstack(parts)
    pd.DataFrame({"test_id": g["test_id"].to_numpy(),
                  **{qcol(lv): q[:, j] for j, lv in enumerate(QUANTILES)}}
                 ).to_parquet(PER_KEY / f"{name}.parquet", index=False)
    rec = {"ctx_key": name, "level": g["ctx_level"].iloc[0], "device": device,
           "n_ctx": len(ctx), "n_match": n_match, "capped": n_match > len(ctx),
           "n_queries": len(g), "fit_s": round(fit_s, 3), "predict_s": round(pred_s, 3),
           "final_chunk": chunk, "ooms": ooms,
           "free_mib_at_start": None if free_mib is None else round(free_mib),
           "n_estimators": getattr(m, "n_estimators_", m.n_estimators),
           "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else 0,
           "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1) if cuda else 0,
           "worker_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                        / 1024)}
    with open(STATS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    del m
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


def work(budget: float, device: str = "cuda", reverse: bool = False) -> None:
    import torch

    test = pd.read_parquet(OUT2 / "test.parquet")
    PER_KEY.mkdir(parents=True, exist_ok=True)
    todo = list(todo_groups(test, device))
    if not todo:
        return
    if device == "cpu":
        # Several CPU workers may run at once (--reverse for the second one); a claim
        # file created with O_EXCL makes sure each context is computed once.
        torch.set_num_threads(CPU_THREADS)
        for name, g in (todo[::-1] if reverse else todo):
            try:
                os.close(os.open(PER_KEY / f"{name}.claim", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            if not (PER_KEY / f"{name}.parquet").exists():
                run_context(name, g, "cpu", None)
        return
    lock = open(LOCK, "w")
    t_wait = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    print(f"worker: lock after {time.time() - t_wait:.0f}s wait", flush=True)
    t_start = time.time()
    done = 0
    for name, g in todo:
        if time.time() - t_start > budget:
            break
        ctx, n_match = context_rows(key_tuple(g))
        size = "small" if len(ctx) < LARGE_CTX else "large"
        # This process's CUDA context stays; its cached weights/activations count as free.
        mine = torch.cuda.memory_reserved() / 2**20 if torch.cuda.is_initialized() else 0
        free_mib = gpu_free_mib() + mine
        need = need_free_mib(len(ctx)) - (300 if torch.cuda.is_initialized() else 0)
        if free_mib < need:
            print(f"worker: {free_mib:.0f} MiB GPU free, need {need:.0f} for {name}; "
                  "backing off", flush=True)
            sys.exit(BUSY)

        run_context(name, g, "cuda", free_mib, MAX_CONTEXT_S[size])
        done += 1
    print(f"worker: {done} contexts in {time.time() - t_start:.0f}s", flush=True)


def merge() -> None:
    test = pd.read_parquet(OUT2 / "test.parquet")
    pred = pd.concat([pd.read_parquet(PER_KEY / f"{k}.parquet")
                      for k in sorted(test["ctx_key"].unique())]).sort_values("test_id")
    assert (pred["test_id"].to_numpy() == test["test_id"].to_numpy()).all()
    pred.to_parquet(OUT2 / "preds_tabpfn.parquet", index=False)
    print(f"merged {len(pred):,} rows -> {OUT2 / 'preds_tabpfn.parquet'}")


def drive(budget: float) -> None:
    test = pd.read_parquet(OUT2 / "test.parquet")
    backoffs = 0
    while any(True for _ in todo_groups(test)):
        r = subprocess.run([sys.executable, "-u", "-m", "src.ride.r2_tabpfn", "work",
                            "--budget", str(budget)])
        if r.returncode in (BUSY, STUCK):
            backoffs += 1
            if backoffs > MAX_BACKOFFS:
                sys.exit("too many back-offs; GPU stays too full")
            time.sleep(120 if r.returncode == BUSY else 30)  # lock released meanwhile
            continue
        if r.returncode != 0:
            sys.exit(f"worker failed with code {r.returncode}")
        backoffs = 0
        time.sleep(10)  # let another lock holder in between workers
    if REDO:
        return
    if not REDO and any(True for _ in todo_groups(test, "cpu")):
        r = subprocess.run([sys.executable, "-u", "-m", "src.ride.r2_tabpfn", "work",
                            "--device", "cpu"])
        if r.returncode != 0:
            sys.exit(f"cpu worker failed with code {r.returncode}")
    merge()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["drive", "work", "merge"])
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    {"drive": lambda: drive(a.budget), "work": lambda: work(a.budget, a.device, a.reverse),
     "merge": merge}[a.cmd]()
