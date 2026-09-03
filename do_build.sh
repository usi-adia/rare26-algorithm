#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build "$SCRIPT_DIR" --platform=linux/amd64 --tag rare26-algorithm 2>&1
