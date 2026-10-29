"""Execute corrected standard runs without replacing historical artifact directories."""

import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    evidence = Path("reports/release")
    evidence.mkdir(parents=True, exist_ok=True)
    existing_file = evidence / "pipeline_execution.json"
    rows = json.loads(existing_file.read_text()) if existing_file.exists() else []
    completed = {(r["mode"], r["stage"]): r for r in rows if r["exit_code"] == 0}

    # Import completed intermittent execution rows if not already present
    intermittent_exec_file = evidence / "intermittent_execution.json"
    if intermittent_exec_file.exists():
        inter_rows = json.loads(intermittent_exec_file.read_text())
        for r in inter_rows:
            stage = r["stage"]
            if ("m5_intermittent", stage) not in completed:
                log_src = Path(r["log"])
                log_dst = evidence / f"m5_intermittent_{stage}.log"
                if log_src.exists() and not log_dst.exists():
                    log_dst.write_text(log_src.read_text())
                command = [sys.executable, "-m", "inventory_di.cli", stage, "--config", "configs/m5_intermittent.yaml"]
                if stage == "train":
                    command += ["--model", "all"]
                entry = dict(
                    mode="m5_intermittent",
                    stage=stage,
                    command=command,
                    seconds=r["seconds"],
                    exit_code=r["exit_code"],
                    log=str(log_dst),
                )
                completed[("m5_intermittent", stage)] = entry

    stages = ["prepare", "train", "backtest", "report"]
    modes = ["demo", "m5", "m5_intermittent"]

    for mode in modes:
        for stage in stages:
            if (mode, stage) in completed:
                print(f"Skipping already completed {mode}: {stage}", flush=True)
                continue
            command = [sys.executable, "-u", "-m", "inventory_di.cli", stage, "--config", f"configs/{mode}.yaml"]
            if stage == "train":
                command += ["--model", "all"]
            log = evidence / f"{mode}_{stage}.log"
            print(f"Starting {mode}: {stage}", flush=True)
            started = time.perf_counter()
            with log.open("w") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
            entry = dict(
                mode=mode,
                stage=stage,
                command=command,
                seconds=time.perf_counter() - started,
                exit_code=result.returncode,
                log=str(log),
            )
            completed[(mode, stage)] = entry

            # Write rows in canonical order
            ordered_rows = [completed[(m, s)] for m in modes for s in stages if (m, s) in completed]
            existing_file.write_text(json.dumps(ordered_rows, indent=2))
            print(f"Completed {mode}: {stage}, exit={result.returncode}, seconds={entry['seconds']:.1f}", flush=True)
            if result.returncode:
                print(log.read_text()[-3000:], flush=True)
                return result.returncode

    ordered_rows = [completed[(m, s)] for m in modes for s in stages if (m, s) in completed]
    existing_file.write_text(json.dumps(ordered_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
