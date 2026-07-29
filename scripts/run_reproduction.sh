#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check -q -r requirements.txt
python -m planphys.evaluate --config experiment_config.json

