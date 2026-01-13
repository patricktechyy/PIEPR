#!/usr/bin/env bash
set -e

python3 process.py --data "$(dirname "$0")/data"
