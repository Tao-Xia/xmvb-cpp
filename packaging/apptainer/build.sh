#!/usr/bin/env bash
# Usage: build.sh EXPORTED_SOURCE REVISION VERSION WORK_DIR RUNTIME_SIF PROGRAM_TAR_GZ
set -euo pipefail
src=$(realpath "$1")
revision=$2
version=$3
work=$(realpath -m "$4")
runtime=$(realpath -m "$5")
output=$(realpath -m "$6")
here=$(dirname "$(realpath "$0")")
cache=$(realpath -m "${XMVB_RELEASE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/xmvb-release}")
: "${XMVB_RELEASE_BASE_SIF:?Set the pinned Ubuntu 24.04 base SIF path}"
base=$(realpath "$XMVB_RELEASE_BASE_SIF")
[[ $revision =~ ^[0-9a-f]{40}$ && $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
[[ -f $src/CMakeLists.txt && $runtime != "$output" ]]
case "$cache/" in "$src/"*) echo 'Cache must be outside the exported source' >&2; exit 2;; esac
case "$src/" in "$cache/"*) echo 'Export source outside the build cache' >&2; exit 2;; esac
case "$work/" in "$src/"*|"$cache/"*) echo 'Use a separate release work directory' >&2; exit 2;; esac
mkdir -p "$cache" "$work" "$(dirname "$runtime")" "$(dirname "$output")"
exec 8>"$work/.lock"
flock 8
[[ ! -e $output && ! -e $output.sha256 && ! -e $work/stage ]]
base_sha=$(sha256sum "$base" | cut -d ' ' -f 1)

if [[ -n ${XMVB_RELEASE_BUILDER_SIF:-} ]]; then
    builder=$(realpath "$XMVB_RELEASE_BUILDER_SIF")
else
    : "${XMVB_RELEASE_DEPENDENCY_PREFIX:?Provide the dependency-complete Conda compiler prefix}"
    deps=$(realpath "$XMVB_RELEASE_DEPENDENCY_PREFIX")
    deps_sha=$(cd "$deps" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1)
    recipe_sha=$(cat "$here/image.sh" "$here/builder.env" | sha256sum | cut -d ' ' -f 1)
    builder_key=$(printf '%s\n' "$(uname -m)" "$base_sha" "$deps_sha" "$recipe_sha" | sha256sum | cut -d ' ' -f 1)
    builder_dir="$cache/builders/$builder_key"
    mkdir -p "$builder_dir"
    builder="$builder_dir/builder.sif"
    (
        flock 9
        if [[ ! -f $builder ]]; then
            temporary=$(mktemp -d "$builder_dir/.build.XXXXXX")
            trap 'rm -rf -- "$temporary"' EXIT
            mkdir "$temporary/deps"
            rsync -rl "$deps/" "$temporary/deps/"
            python3 - "$deps" "$temporary/deps" <<'PY'
import json
from pathlib import Path
import shutil
import sys

source, dest = map(Path, sys.argv[1:])
for path in (source / "conda-meta").glob("*.json"):
    record = json.loads(path.read_text())
    licenses = Path(record.get("link", {}).get("source", "")) / "info/licenses"
    if licenses.is_dir():
        shutil.copytree(licenses, dest / "release-licenses" / record["name"])
PY
            bash "$here/image.sh" "$base" "$temporary/deps" /opt/xmvb/deps \
                "$here/builder.env" - "$temporary/builder.sif"
            mv -- "$temporary/builder.sif" "$builder"
        fi
    ) 9>"$builder_dir/.lock"
fi

builder_sha=$(sha256sum "$builder" | cut -d ' ' -f 1)
recipe_sha=$(cat "$here/build.sh" "$here/stage.py" "$src/CMakePresets.json" | sha256sum | cut -d ' ' -f 1)
build_key=$(printf '%s\n' "$(uname -m)" cpu "$builder_sha" "$recipe_sha" | sha256sum | cut -d ' ' -f 1)
build_root="$cache/builds/$build_key"
mkdir -p "$build_root/source"
printf '%s\n' "$builder" > "$work/builder-path.txt"
printf '%s\n' "$build_root" > "$work/build-cache-path.txt"
mkdir "$work/stage"
(
    flock 9
    # This directory is owned by this cache key. Content comparison preserves
    # unchanged Ninja inputs even when the source came from a new git archive.
    rsync -rlc --delete --exclude='/.git' "$src/" "$build_root/source/"
    apptainer exec --cleanenv --bind "$build_root" --bind "$work" "$builder" \
        /bin/bash -euo pipefail -c '
            root=$1 work=$2 revision=$3 version=$4 base_sha=$5 builder_sha=$6 build_key=$7 jobs=$8
            export CONDA_PREFIX=/opt/xmvb/deps
            export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
            export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
            export FC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran
            cd "$root/source"
            cmake --preset cpu-release -B "$root/build" -DXMVB_ENABLE_CCACHE=OFF > "$work/configure.log" 2>&1
            cmake --build "$root/build" --target xmvb --parallel "$jobs" > "$work/build.log" 2>&1
            cmake --install "$root/build" --prefix "$work/stage/program" --component xmvb-program > "$work/stage.log" 2>&1
            cmake -DPROGRAM="$work/stage/program/bin/xmvb-cpp.exe" -P packaging/apptainer/remove-rpath.cmake >> "$work/stage.log" 2>&1
            ldd "$work/stage/program/bin/xmvb-cpp.exe" > "$work/stage/dependencies.txt"
            mkdir -p "$work/stage/runtime/lib" "$work/stage/conda-meta" "$work/stage/share"
            while read -r name arrow path remainder; do
                if [[ $arrow == "=>" && $path == /opt/xmvb/deps/lib/* ]]; then
                    cp -L "$path" "$work/stage/runtime/lib/$name"
                fi
            done < "$work/stage/dependencies.txt"
            cp /opt/xmvb/deps/conda-meta/*.json "$work/stage/conda-meta/"
            cp -r /opt/xmvb/deps/release-licenses "$work/stage/"
            cp -r /opt/xmvb/deps/share/licenses "$work/stage/share/"
            "$CXX" --version > "$work/stage/compiler.txt"
            cmake --version > "$work/stage/cmake.txt"
        ' -- "$build_root" "$work" "$revision" "$version" "$base_sha" "$builder_sha" "$build_key" "${SLURM_CPUS_PER_TASK:-8}"
) 9>"$build_root/.lock"
python3 "$here/stage.py" --build "$build_root/build" --stage "$work/stage" \
    --revision "$revision" --version "$version" --base-sha "$base_sha" \
    --builder-sha "$builder_sha" --build-key "$build_key" >> "$work/stage.log" 2>&1

(
    flock 9
    if [[ ! -f $runtime ]]; then
        temporary=$(mktemp -d "$(dirname "$runtime")/.runtime.XXXXXX")
        trap 'rm -rf -- "$temporary"' EXIT
        bash "$here/image.sh" "$base" "$work/stage/runtime" /opt/xmvb \
            "$here/runtime.env" "$here/runtime.sh" "$temporary/runtime.sif"
        mv -- "$temporary/runtime.sif" "$runtime"
    fi
    (cd "$(dirname "$runtime")" && sha256sum "$(basename "$runtime")") > "$runtime.sha256"
) 9>"$runtime.lock" > "$work/runtime-build.log" 2>&1

apptainer exec --cleanenv --bind "$work/stage:/release:ro" "$runtime" /bin/bash -euo pipefail -c '
    cmp /opt/xmvb/runtime-id /release/program/share/RUNTIME_ID
    for lib in /release/runtime/lib/*; do cmp "$lib" "/opt/xmvb/lib/${lib##*/}"; done
    dependencies=$(ldd /release/program/bin/xmvb-cpp.exe)
    printf "%s\n" "$dependencies"
    if grep -q "not found" <<< "$dependencies"; then exit 1; fi
' > "$work/linkage.log" 2>&1
runtime_sha=$(cut -d ' ' -f 1 "$runtime.sha256")
printf '%s\n' "$runtime_sha" > "$work/stage/program/share/RUNTIME_SHA256"
python3 - "$work/stage/program/share/build-info.json" "$(basename "$runtime")" "$runtime_sha" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
metadata = json.loads(path.read_text())
metadata.update(runtime_file=sys.argv[2], runtime_sha256=sys.argv[3])
path.write_text(json.dumps(metadata, indent=2) + "\n")
PY
(cd "$work/stage/program" && find bin share -type f -print0 | sort -z | xargs -0 sha256sum) \
    > "$work/stage/program/SHA256SUMS"
temporary=$(mktemp -d "$(dirname "$output")/.program.XXXXXX")
trap 'rm -rf -- "$temporary"' EXIT
tar -czf "$temporary/program.tar.gz" -C "$work/stage/program" bin share SHA256SUMS
printf '%s  %s\n' "$(sha256sum "$temporary/program.tar.gz" | cut -d ' ' -f 1)" "$(basename "$output")" > "$temporary/program.sha256"
mv -- "$temporary/program.tar.gz" "$output"
mv -- "$temporary/program.sha256" "$output.sha256"
cp "$work/stage/program/share/build-info.json" "$work/build-info.json"
