#!/usr/bin/env bash
set -e

# Process all trials under videoImplement/data (batch mode)
python3 process.py --data "$(dirname "$0")/data"
