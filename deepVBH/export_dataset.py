from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-export DeepVBSCF traces")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input .xmi files and/or directories containing .xmi files",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Additional glob pattern(s) for .xmi inputs, e.g. 'src/test_molecule/*.xmi'",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help="Text file containing one .xmi path per line",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("deepvbscf_data"),
        help="Output dataset root",
    )
    parser.add_argument(
        "--xmvb-exe",
        type=Path,
        default=Path("build/src/xmvb-cpp.exe"),
        help="Path to the standalone xmvb-cpp executable",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=["lbfgspp", "legacy_fortran"],
        default="lbfgspp",
    )
    parser.add_argument(
        "--algorithm",
        choices=["original", "biorthogonal"],
        default="original",
    )
    parser.add_argument(
        "--omp-num-threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing sample directory with the same exported sample name",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Treat unconverged runs as failures and discard the exported sample",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSONL manifest path; defaults to <data-root>/export_manifest.jsonl",
    )
    return parser.parse_args()


def _iter_xmi_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".xmi":
            yield path.resolve()
        return
    if path.is_dir():
        for child in sorted(path.rglob("*.xmi")):
            if child.is_file():
                yield child.resolve()


def _collect_input_paths(args: argparse.Namespace) -> list[Path]:
    collected: set[Path] = set()
    for item in args.inputs:
        collected.update(_iter_xmi_paths(item))
    for pattern in args.input_glob:
        collected.update(path.resolve() for path in sorted(Path().glob(pattern)) if path.is_file())
    if args.input_list is not None:
        for raw_line in args.input_list.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            collected.update(_iter_xmi_paths(Path(line)))
    if not collected:
        raise ValueError("no .xmi inputs found")
    return sorted(collected)


def _sanitize_component(value: str) -> str:
    sanitized = []
    previous_was_separator = False
    for character in value:
        if character.isalnum():
            sanitized.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            sanitized.append("_")
            previous_was_separator = True
    result = "".join(sanitized).strip("_")
    return result or "sample"


def _unique_sample_names(paths: list[Path]) -> dict[Path, str]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(_sanitize_component(path.stem), []).append(path)

    result: dict[Path, str] = {}
    for base_name, group in grouped.items():
        if len(group) == 1:
            result[group[0]] = base_name
            continue

        remaining = sorted(group)
        assigned: dict[Path, str] = {}
        depth = 1
        while remaining:
            trial_groups: dict[str, list[Path]] = {}
            for path in remaining:
                parts = path.with_suffix("").parts
                prefix_parts = parts[-(depth + 1) :]
                candidate = _sanitize_component("_".join(prefix_parts))
                trial_groups.setdefault(candidate, []).append(path)
            remaining = []
            for candidate, candidate_paths in trial_groups.items():
                if len(candidate_paths) == 1:
                    assigned[candidate_paths[0]] = candidate
                else:
                    remaining.extend(candidate_paths)
            depth += 1

        for path in group:
            result[path] = assigned[path]
    return result


def _sample_completed(sample_dir: Path) -> bool:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    return str(metadata.get("status")) == "completed"


def _load_sample_metadata(sample_dir: Path) -> dict[str, object]:
    return json.loads((sample_dir / "metadata.json").read_text())


def _rewrite_sample_name(sample_dir: Path, sample_name: str) -> None:
    metadata_path = sample_dir / "metadata.json"
    metadata = _load_sample_metadata(sample_dir)
    metadata["sample_name"] = sample_name
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def _run_export(
    *,
    xmvb_exe: Path,
    input_path: Path,
    dataset_root: Path,
    optimizer_backend: str,
    algorithm: str,
    omp_num_threads: int,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(xmvb_exe.resolve()),
        str(input_path),
        "--optimizer-backend",
        optimizer_backend,
        "--algorithm",
        algorithm,
        "--dump-trace-dir",
        str(dataset_root),
    ]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(omp_num_threads)
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )


def _extract_generated_sample_dir(temp_root: Path) -> Path:
    candidates = [path for path in temp_root.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one generated sample under {temp_root}, found {len(candidates)}"
        )
    return candidates[0]


def _append_manifest_record(manifest_path: Path, record: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    input_paths = _collect_input_paths(args)
    sample_names = _unique_sample_names(input_paths)

    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else data_root / "export_manifest.jsonl"
    )

    summary = {
        "requested_inputs": len(input_paths),
        "exported": 0,
        "skipped": 0,
        "failed": 0,
    }

    for input_path in input_paths:
        sample_name = sample_names[input_path]
        target_dir = data_root / sample_name
        start_time = time.time()

        if target_dir.exists() and not args.overwrite and _sample_completed(target_dir):
            summary["skipped"] += 1
            _append_manifest_record(
                manifest_path,
                {
                    "sample_name": sample_name,
                    "input_path": str(input_path),
                    "status": "skipped_existing",
                    "elapsed_seconds": time.time() - start_time,
                },
            )
            print(
                json.dumps(
                    {
                        "sample_name": sample_name,
                        "input_path": str(input_path),
                        "status": "skipped_existing",
                    },
                    ensure_ascii=False,
                )
            )
            continue

        with tempfile.TemporaryDirectory(prefix="deepvbh_export_", dir=str(data_root)) as temp_dir_name:
            temp_root = Path(temp_dir_name)
            process = _run_export(
                xmvb_exe=args.xmvb_exe,
                input_path=input_path,
                dataset_root=temp_root,
                optimizer_backend=args.optimizer_backend,
                algorithm=args.algorithm,
                omp_num_threads=args.omp_num_threads,
            )

            if process.returncode not in {0, 2}:
                summary["failed"] += 1
                _append_manifest_record(
                    manifest_path,
                    {
                        "sample_name": sample_name,
                        "input_path": str(input_path),
                        "status": "failed_process",
                        "returncode": process.returncode,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "elapsed_seconds": time.time() - start_time,
                    },
                )
                print(
                    json.dumps(
                        {
                            "sample_name": sample_name,
                            "input_path": str(input_path),
                            "status": "failed_process",
                            "returncode": process.returncode,
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            generated_dir = _extract_generated_sample_dir(temp_root)
            generated_metadata = _load_sample_metadata(generated_dir)
            converged = bool(generated_metadata.get("converged", False))
            if args.require_converged and not converged:
                summary["failed"] += 1
                _append_manifest_record(
                    manifest_path,
                    {
                        "sample_name": sample_name,
                        "input_path": str(input_path),
                        "status": "failed_unconverged",
                        "returncode": process.returncode,
                        "converged": converged,
                        "termination_reason": generated_metadata.get("termination_reason"),
                        "elapsed_seconds": time.time() - start_time,
                    },
                )
                print(
                    json.dumps(
                        {
                            "sample_name": sample_name,
                            "input_path": str(input_path),
                            "status": "failed_unconverged",
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(generated_dir), str(target_dir))
            _rewrite_sample_name(target_dir, sample_name)

            summary["exported"] += 1
            _append_manifest_record(
                manifest_path,
                {
                    "sample_name": sample_name,
                    "input_path": str(input_path),
                    "status": "exported",
                    "returncode": process.returncode,
                    "converged": converged,
                    "accepted_iteration_count": generated_metadata.get("accepted_iteration_count"),
                    "final_total_energy": generated_metadata.get("final_total_energy"),
                    "elapsed_seconds": time.time() - start_time,
                },
            )
            print(
                json.dumps(
                    {
                        "sample_name": sample_name,
                        "input_path": str(input_path),
                        "status": "exported",
                        "converged": converged,
                        "accepted_iteration_count": generated_metadata.get("accepted_iteration_count"),
                    },
                    ensure_ascii=False,
                )
            )

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
