#!/bin/sh
if ! cmp -s /opt/xmvb/runtime-id /opt/xmvb/program/share/RUNTIME_ID; then
    echo 'Program package requires a different XMVB runtime' >&2
    exit 2
fi
export XMVB_BASIS_DIR=/opt/xmvb/program/share/basis
exec /opt/xmvb/program/bin/xmvb-cpp.exe "$@"
