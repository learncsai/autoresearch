
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ORDER = [
    "baseline",
    "exp01", "exp02", "exp03", "exp04", "exp05",
    "exp06", "exp07", "exp08", "exp09", "exp10",
    "exp11", "exp12", "exp13", "exp14", "exp15",
    "exp16", "exp17", "exp18", "exp19", "exp20",
]

# Per-experiment wall clock limit for the whole Python process.
# Slightly above the 5-minute training budget to allow startup/compile/eval.
TIMEOUT_SECONDS = int(os.environ.get("EXPERIMENT_TIMEOUT_SECONDS", "600"))

SCRIPT = Path(__file__).with_name("train_experiments.py")
RESULTS_DIR = Path(__file__).with_name("experiment_logs")
RESULTS_DIR.mkdir(exist_ok=True)

PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "total_seconds": re.compile(r"^total_seconds:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "mfu_percent": re.compile(r"^mfu_percent:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "total_tokens_M": re.compile(r"^total_tokens_M:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "num_steps": re.compile(r"^num_steps:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "num_params_M": re.compile(r"^num_params_M:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
    "depth": re.compile(r"^depth:\s+([0-9eE.+-]+)\s*$", re.MULTILINE),
}

def parse_metrics(log_text: str):
    metrics = {}
    for key, pattern in PATTERNS.items():
        m = pattern.search(log_text)
        metrics[key] = m.group(1) if m else ""
    return metrics

def main():
    summary_path = RESULTS_DIR / "summary.tsv"
    results_path = RESULTS_DIR / "results.tsv"

    summary_header = "index\texperiment_id\tstatus\treturncode\telapsed_seconds\tlog_path\n"
    results_header = (
        "index\texperiment_id\tstatus\treturncode\telapsed_seconds\tlog_path\t"
        "val_bpb\ttraining_seconds\ttotal_seconds\tpeak_vram_mb\tmfu_percent\t"
        "total_tokens_M\tnum_steps\tnum_params_M\tdepth\n"
    )

    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(summary_header)

    with open(results_path, "w", encoding="utf-8") as rf:
        rf.write(results_header)

    print("Running experiments in order:")
    for i, exp_id in enumerate(ORDER, start=1):
        print(f"  {i:02d}. {exp_id}")
    print()
    print(f"Per-run timeout: {TIMEOUT_SECONDS}s")
    print(f"Logs directory: {RESULTS_DIR}")
    print()

    for i, exp_id in enumerate(ORDER, start=1):
        log_path = RESULTS_DIR / f"{i:02d}_{exp_id}.log"
        start = time.time()

        print("=" * 100)
        print(f"[{i:02d}/{len(ORDER)}] Starting {exp_id}")
        print("=" * 100)

        env = os.environ.copy()
        env["EXPERIMENT_ID"] = exp_id

        status = "ok"
        returncode = 0

        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"=== Experiment {exp_id} ===\n")
            logf.write(f"Timeout seconds: {TIMEOUT_SECONDS}\n\n")
            logf.flush()

            try:
                result = subprocess.run(
                    [sys.executable, str(SCRIPT)],
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=TIMEOUT_SECONDS,
                )
                returncode = result.returncode
                if returncode != 0:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
                returncode = 124
                logf.write("\n\n=== TIMEOUT ===\n")
                logf.write(f"Process exceeded {TIMEOUT_SECONDS} seconds and was terminated.\n")
            except Exception as e:
                status = "runner_error"
                returncode = 1
                logf.write("\n\n=== RUNNER ERROR ===\n")
                logf.write(repr(e) + "\n")

        elapsed = time.time() - start
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = parse_metrics(log_text)

        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(f"{i}\t{exp_id}\t{status}\t{returncode}\t{elapsed:.1f}\t{log_path.name}\n")

        with open(results_path, "a", encoding="utf-8") as rf:
            rf.write(
                f"{i}\t{exp_id}\t{status}\t{returncode}\t{elapsed:.1f}\t{log_path.name}\t"
                f"{metrics['val_bpb']}\t{metrics['training_seconds']}\t{metrics['total_seconds']}\t"
                f"{metrics['peak_vram_mb']}\t{metrics['mfu_percent']}\t{metrics['total_tokens_M']}\t"
                f"{metrics['num_steps']}\t{metrics['num_params_M']}\t{metrics['depth']}\n"
            )

        metric_msg = f"val_bpb={metrics['val_bpb']}" if metrics["val_bpb"] else "val_bpb=N/A"
        print(
            f"Finished {exp_id} | status={status} | elapsed={elapsed:.1f}s | "
            f"{metric_msg} | log={log_path.name}"
        )
        print()

    print("=" * 100)
    print("All experiments attempted.")
    print(f"Summary written to: {summary_path}")
    print(f"Parsed results written to: {results_path}")
    print("=" * 100)

if __name__ == "__main__":
    main()
