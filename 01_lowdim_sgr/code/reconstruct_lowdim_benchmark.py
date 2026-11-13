#!/usr/bin/env python3
"""Extract the 27 SGR checkpoints over their 10%-padded input domains."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def discover(repository_root: Path, asset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration in sorted(
        (repository_root / "example" / "lowdim").glob(
            "example_*/config/training.json"
        )
    ):
        example = configuration.parent.parent
        payload = load_json(configuration)
        extraction = payload.get("extraction_parameters", {})
        if float(extraction.get("bbox_pad_fraction_each_side", -1.0)) != 0.10:
            raise ValueError(f"Stale extraction domain: {configuration}")
        if int(extraction.get("grid_resolution", -1)) != 360:
            raise ValueError(f"Stale extraction resolution: {configuration}")
        if str(extraction.get("component_policy")) != "all_components":
            raise ValueError(f"Stale component policy: {configuration}")
        point_cloud = (
            asset_root / str(payload["training_parameters"]["point_cloud"])
        ).resolve()
        checkpoint = example / "result" / "implicit_function" / "best.pt"
        if not point_cloud.is_file():
            raise FileNotFoundError(point_cloud)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows.append(
            {
                "dataset": str(payload["dataset"]),
                "example": example,
                "configuration": configuration,
                "point_cloud": point_cloud,
                "checkpoint": checkpoint,
            }
        )
    if len(rows) != 27:
        raise RuntimeError(f"Expected 27 examples, found {len(rows)}")
    return rows


def valid_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        stats = load_json(Path(payload["stats"]))
        return (
            payload.get("status") == "ok"
            and Path(payload["mesh"]).is_file()
            and int(stats.get("grid_n", -1)) == 360
            and float(stats.get("bbox_pad_fraction_each_side", -1.0)) == 0.10
            and str(stats.get("component_policy")) == "all_components"
        )
    except Exception:
        return False


def run_one(
    row: dict[str, Any],
    gpu: int,
    *,
    python: Path,
    extractor: Path,
    output_root: Path,
    batch_size: int,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    dataset = row["dataset"]
    directory = output_root / dataset
    mesh = directory / "zero_mesh.npz"
    stats = directory / "stats.json"
    done = directory / "DONE.json"
    log = directory / "extract.log"
    if valid_done(done):
        print(f"reuse dataset={dataset}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-u",
        str(extractor),
        "--checkpoint",
        str(row["checkpoint"]),
        "--point-cloud",
        str(row["point_cloud"]),
        "--bbox-pad-fraction-each-side",
        "0.10",
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--out-mesh-npz",
        str(mesh),
        "--out-stats",
        str(stats),
    ]
    with lock:
        state["running"][dataset] = {"gpu": gpu, "started_at": utc_now()}
        write_json(state_path, state)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
    elapsed = time.time() - started
    payload = {
        "dataset": dataset,
        "status": "ok" if completed.returncode == 0 and mesh.is_file() and stats.is_file() else "failed",
        "gpu": gpu,
        "runtime_seconds": elapsed,
        "checkpoint": str(row["checkpoint"]),
        "configuration": str(row["configuration"]),
        "point_cloud": str(row["point_cloud"]),
        "mesh": str(mesh),
        "stats": str(stats),
        "log": str(log),
        "command": command,
        "return_code": completed.returncode,
        "finished_at": utc_now(),
    }
    write_json(done if payload["status"] == "ok" else directory / "FAILED.json", payload)
    with lock:
        state["running"].pop(dataset, None)
        state["completed" if payload["status"] == "ok" else "failures"][dataset] = payload
        write_json(state_path, state)
    print(
        f"{payload['status']} dataset={dataset} gpu={gpu} elapsed={elapsed:.1f}s",
        flush=True,
    )


def worker(
    tasks: queue.Queue[dict[str, Any]],
    gpu: int,
    **kwargs: Any,
) -> None:
    while True:
        try:
            row = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            run_one(row, gpu, **kwargs)
        finally:
            tasks.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=131072)
    args = parser.parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    python = Path(args.python).expanduser()
    extractor = repository_root / "01_lowdim_sgr" / "code" / "reconstruct.py"
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    rows = discover(repository_root, asset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    state: dict[str, Any] = {
        "schema": "sgr-lowdim-bbox10-extraction/v1",
        "started_at": utc_now(),
        "grid_resolution": 360,
        "bbox_pad_fraction_each_side": 0.10,
        "component_policy": "all_components",
        "datasets": len(rows),
        "gpus": gpus,
        "running": {},
        "completed": {},
        "failures": {},
    }
    write_json(state_path, state)
    tasks: queue.Queue[dict[str, Any]] = queue.Queue()
    for row in rows:
        if not valid_done(output_root / row["dataset"] / "DONE.json"):
            tasks.put(row)
    common = {
        "python": python,
        "extractor": extractor,
        "output_root": output_root,
        "batch_size": int(args.batch_size),
        "state": state,
        "state_path": state_path,
    }
    threads = [
        threading.Thread(target=worker, args=(tasks, gpu), kwargs=common)
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        state["finished_at"] = utc_now()
        state["status"] = "complete" if not state["failures"] else "complete_with_failures"
        write_json(state_path, state)
    if state["failures"]:
        raise SystemExit(f"{len(state['failures'])} extraction jobs failed")
    print(f"complete datasets={len(rows)} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
