"""Poll cup-hanging jobs and compare queue estimates without changing any jobs."""

import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
DIRECTORY = REPO / "tmp/cup_hanging_0913_monitor"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def command(args):
    result = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=40, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout + result.stderr


def snapshot(jobs):
    ids = ",".join(str(job["job_id"]) for job in jobs)
    accounting = command(["sacct", "-X", "-n", "-P", "-j", ids, "-o", "JobID,State,ExitCode,Elapsed,NodeList"])
    states = {line.split("|")[0]: line.split("|")[1:5] for line in accounting.splitlines() if "|" in line}
    all_terminal = len(states) == len(jobs) and all(value[0].split()[0] in TERMINAL for value in states.values())
    queue = "" if all_terminal else command(["squeue", "-h", "-j", ids, "-o", "%i|%T|%S|%R"])
    live = {line.split("|")[0]: line.split("|")[1:] for line in queue.splitlines() if "|" in line}
    records = []
    for job in jobs:
        jid = str(job["job_id"])
        status = states.get(jid, ["UNKNOWN", "", "", ""])
        record = {**job, "state": status[0], "exit_code": status[1], "elapsed": status[2], "node": status[3]}
        if jid in live:
            record.update(state=live[jid][0], estimated_start=live[jid][1], reason_or_node=live[jid][2])
        logfile = REPO / job["log"]
        if logfile.exists():
            with logfile.open("rb") as stream:
                stream.seek(max(0, logfile.stat().st_size - 128_000))
                tail = stream.read().decode(errors="replace")
            steps = re.findall(r"Step (\d+):[^\r\n]*", tail)
            progress = re.findall(r"Progress on:[^\r\n]*", tail)
            record["last_logged_step"] = int(steps[-1]) if steps else None
            record["last_progress"] = progress[-1] if progress else None
            record["error_in_log_tail"] = bool(re.search(r"Traceback|CUDA_ERROR_ECC|OutOfMemoryError", tail))
            record["log_modified"] = datetime.datetime.fromtimestamp(logfile.stat().st_mtime).astimezone().isoformat()
        records.append(record)
    probes = {}
    if any(record["state"] == "PENDING" for record in records):
        for account, partition, node, gpu in (
            ("iliad", "iliad", "iliad-hgx-1", "h200"),
            ("iris", "iris-hi", "iris-hgx-1", "h100"),
            ("iris", "iris-hi", "iris-hgx-2", "h200"),
        ):
            output = command(
                [
                    "sbatch",
                    "--test-only",
                    f"--account={account}",
                    f"--partition={partition}",
                    f"--nodelist={node}",
                    f"--gres=gpu:{gpu}:1",
                    f"--export=ALL,CONFIG={jobs[0]['config']}",
                    "examples/droid/cup_hanging/train_0913.slurm",
                ]
            )
            start = re.search(r"to start at (\d{4}-\d{2}-\d{2}T[\d:]+)", output)
            probes[node] = {"estimated_start": start[1] if start else None, "scheduler_output": output.strip()}
    available = {node: value["estimated_start"] for node, value in probes.items() if value["estimated_start"]}
    best = min(available, key=available.get) if available else None
    return {
        "checked_at": datetime.datetime.now().astimezone().isoformat(),
        "jobs": records,
        "equivalent_new_job_probes": probes,
        "earliest_new_job_node": best,
        "recommendation": (
            "Keep queued jobs on Iliad"
            if best == "iliad-hgx-1"
            else "Review alternate queue estimates"
            if best
            else "No comparable pending-job estimates"
        ),
        "caveat": "Probes model new submissions, not guaranteed start times for existing jobs; no jobs are modified.",
    }


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = json.loads((Path(__file__).parent / "submitted_jobs_0913.json").read_text())["jobs"]
        while True:
            try:
                result = snapshot(jobs)
                temporary = DIRECTORY / "latest.json.tmp"
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(DIRECTORY / "latest.json")
                with (DIRECTORY / "history.jsonl").open("a") as history:
                    history.write(json.dumps(result) + "\n")
                print(result["checked_at"], result["recommendation"], flush=True)
                if all(job["state"].split()[0] in TERMINAL for job in result["jobs"]):
                    return
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                print(datetime.datetime.now().astimezone().isoformat(), str(error), flush=True)
            for _ in range(5):
                time.sleep(60)


if __name__ == "__main__":
    main()
