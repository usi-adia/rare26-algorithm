#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/do_build.sh"
docker save rare26-algorithm | gzip -c > "$SCRIPT_DIR/rare26-algorithm.tar.gz"
ls -lh "$SCRIPT_DIR/rare26-algorithm.tar.gz"
