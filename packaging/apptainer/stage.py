#!/usr/bin/env python3
"""Stage the program and dependency files for a split Apptainer release."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


SOURCE = Path(__file__).resolve().parents[2]
SYSTEM_LIBS = {
    "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1",
    "libutil.so.1", "libresolv.so.2", "ld-linux-x86-64.so.2",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def manifest(directory):
    return "".join(
        f"{digest(path)}  {path.relative_to(directory)}\n"
        for path in sorted(directory.rglob("*")) if path.is_file()
    )


def stage(args):
    prefix = Path("/opt/xmvb/deps")
    program, runtime = args.stage / "program", args.stage / "runtime"
    cache = dict(re.findall(r"^(\w+):[^=\n]+=(.*)$",
                            (args.build / "CMakeCache.txt").read_text(), re.M))
    installed = program / "bin/xmvb-cpp.exe"
    records = [json.loads(path.read_text()) for path in (args.stage / "conda-meta").glob("*.json")]
    owners = {name: record for record in records for name in record.get("files", [])}
    used = {record["name"]: record for record in records if record["name"] == "eigen"}
    for line in (args.stage / "dependencies.txt").read_text().splitlines():
        if "=> not found" in line:
            raise ValueError(line.strip())
        match = re.fullmatch(r"\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-f]+\)\s*", line)
        if not match:
            continue
        name, filename = match.groups()
        if name in SYSTEM_LIBS:
            continue
        path = Path(filename)
        if path.parent != prefix / "lib":
            raise ValueError(f"Dependency resolved outside the builder prefix: {line}")
        record = owners.get(f"lib/{path.name}")
        if record is None:
            raise ValueError(f"Missing package metadata for {path}")
        used[record["name"]] = record
        if not (runtime / "lib" / name).is_file():
            raise ValueError(f"Missing exported builder library: {name}")

    for name, record in sorted(used.items()):
        dest = runtime / "licenses" / name
        dest.mkdir(parents=True)
        cached = args.stage / "release-licenses" / name
        if cached.is_dir():
            shutil.copytree(cached, dest, dirs_exist_ok=True)
        for filename in record.get("files", []):
            if filename.startswith("share/licenses/") and (args.stage / filename).is_file():
                shutil.copy2(args.stage / filename, dest / Path(filename).name)
        if not any(dest.iterdir()):
            raise ValueError(f"Missing license files for {name}")
    runtime_info = {
        "base_sha256": args.base_sha,
        "runtime_recipe_sha256": hashlib.sha256(b"".join(
            (SOURCE / "packaging/apptainer" / name).read_bytes()
            for name in ("image.sh", "runtime.env", "runtime.sh")
        )).hexdigest(),
        "packages": {name: f'{record["version"]}={record["build"]}'
                     for name, record in sorted(used.items())},
    }
    (runtime / "runtime.json").write_text(json.dumps(runtime_info, indent=2) + "\n")
    content = manifest(runtime)
    (runtime / "SHA256SUMS").write_text(content)
    runtime_id = hashlib.sha256(content.encode()).hexdigest() + "\n"
    (runtime / "runtime-id").write_text(runtime_id)
    share = program / "share"
    share.mkdir()
    (share / "RUNTIME_ID").write_text(runtime_id)
    shutil.copytree(SOURCE / "basis", share / "basis")
    shutil.copytree(SOURCE / "licenses", share / "licenses")
    for filename in ("LICENSE", "NOTICE", "BUILDING.md"):
        shutil.copy2(SOURCE / filename, share / filename)
    launcher = program / "bin/xmvb"
    shutil.copyfile(SOURCE / "packaging/apptainer/xmvb", launcher)
    launcher.chmod(0o755)
    (share / "examples").mkdir()
    example = (SOURCE / "testdata/vbscf/F2.xmi").read_text()
    (share / "examples/F2.xmi").write_text(example.replace("ISCF=7", "ISCF=9"))
    metadata = {
        "version": args.version,
        "source_revision": args.revision,
        "builder_sha256": args.builder_sha,
        "build_cache_key": args.build_key,
        "backend": "cpu",
        "runtime_id": runtime_id.strip(),
        "program_sha256": digest(installed),
        "compiler": (args.stage / "compiler.txt").read_text().splitlines()[0],
        "cmake": (args.stage / "cmake.txt").read_text().splitlines()[0],
        "build_settings": {key: value for key, value in sorted(cache.items())
                           if key.startswith(("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS",
                                              "CMAKE_Fortran_FLAGS", "CMAKE_EXE_LINKER_FLAGS", "XMVB_"))
                           or key in ("CMAKE_BUILD_TYPE", "CMAKE_GENERATOR")},
        "conda_packages": sorted(f'{record["name"]}={record["version"]}={record["build"]}'
                                 for record in records),
    }
    (share / "build-info.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--builder-sha", required=True)
    parser.add_argument("--build-key", required=True)
    stage(parser.parse_args())
