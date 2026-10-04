"""Run the C2G ablation matrix (4 configs x 3 seeds) with bounded parallelism."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_DATA = HERE.parent / "repo" / "parameter-golf-main" / "data"
TOK = REPO_DATA / "tokenizers" / "fineweb_1024_bpe.model"
RUNS = HERE / "runs"

CONFIGS = [
    ("base", {"opt": "adamw"}),
    ("muon", {"opt": "muon"}),
    ("recur", {"opt": "adamw", "depth_share": 2}),
    ("tie", {"opt": "adamw", "tie": 1}),
]
SEEDS = [1337, 2024, 7]


def build_cmd(tag, extra, seed, common):
    cmd = [sys.executable, str(HERE / "c2g_train.py"),
           "--data-dir", str(REPO_DATA), "--tok", str(TOK),
           "--tag", tag, "--seed", str(seed),
           "--out", str(RUNS / f"{tag}_seed{seed}.json")]
    for k, v in common.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    for k, v in extra.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--tags", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s] or SEEDS
    tags = set(t for t in args.tags.split(",") if t)

    common = {"threads": args.threads, "max_seconds": args.max_seconds, "steps": args.steps}
    jobs = []
    for tag, extra in CONFIGS:
        if tags and tag not in tags:
            continue
        for seed in seeds:
            jobs.append((tag, seed, build_cmd(tag, extra, seed, common)))

    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"launching {len(jobs)} jobs, max_parallel={args.max_parallel}", flush=True)
    running = []
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < args.max_parallel:
            tag, seed, cmd = jobs.pop(0)
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append((tag, seed, p))
        time.sleep(3)
        still = []
        for tag, seed, p in running:
            if p.poll() is None:
                still.append((tag, seed, p))
            else:
                print(f"  [{time.time() - t0:7.1f}s] {tag}_seed{seed} exit={p.returncode}", flush=True)
        running = still

    rows = []
    for tag, _ in CONFIGS:
        for seed in seeds:
            f = RUNS / f"{tag}_seed{seed}.json"
            if f.exists():
                rows.append(json.loads(f.read_text(encoding="utf-8")))
            else:
                print(f"MISSING {f}")
    summary = RUNS / "summary.json"
    summary.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {summary} with {len(rows)} runs in {time.time() - t0:.1f}s\n")
    print(f"{'tag':6} {'seed':6} {'steps':6} {'tok':>9} {'wall':>7} {'params':>9} "
          f"{'val_bpb':>9} {'int8_bpb':>9} {'art_B':>10}")
    for r in sorted(rows, key=lambda r: (r["tag"], r["seed"])):
        print(f"{r['tag']:6} {r['seed']:<6} {r['steps']:<6} {r['tokens_seen']:>9} "
              f"{r['wall_seconds']:>7.1f} {r['n_params']:>9} {r['val_bpb']:>9.4f} "
              f"{r['int8_val_bpb']:>9.4f} {r['artifact_bytes_zlib_int8']:>10}")


if __name__ == "__main__":
    main()
