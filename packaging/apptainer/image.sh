#!/usr/bin/env bash
# Assemble dependency files over a pinned base without installing packages.
# Usage: image.sh BASE_SIF PAYLOAD IMAGE_DIRECTORY ENV_FILE RUNSCRIPT OUTPUT_SIF
set -euo pipefail
base=$1 payload=$2 image_dir=$3 environment=$4 runscript=$5 output=$6
temporary=$(mktemp -d "$(dirname "$output")/.image.XXXXXX")
trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$temporary"' EXIT
apptainer build --sandbox "$temporary/rootfs" "$base"
mkdir -p "$temporary/rootfs$image_dir"
rsync -rl "$payload/" "$temporary/rootfs$image_dir/"
mkdir -p "$temporary/rootfs/opt/xmvb/program"
cp "$environment" "$temporary/rootfs/.singularity.d/env/91-environment.sh"
if [[ $runscript != - ]]; then
    cp "$runscript" "$temporary/rootfs/.singularity.d/runscript"
    chmod 0755 "$temporary/rootfs/.singularity.d/runscript"
fi
apptainer build "$output" "$temporary/rootfs"
