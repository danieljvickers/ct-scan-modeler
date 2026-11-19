#!/usr/bin/env bash
# create or refresh build/venv when toolchain/requirements.txt changes

venv_dir="$CTM_ROOT/build/venv"
requirements="$CTM_ROOT/toolchain/requirements.txt"
stamp="$venv_dir/requirements.sha256"
requirements_hash=$(sha256sum "$requirements" | cut -d' ' -f1)

if [ ! -x "$venv_dir/bin/python" ]; then
    echo "INFO :: creating Python environment in build/venv"
    python3 -m venv "$venv_dir" || { echo "ERROR :: python3 -m venv failed"; exit 1; }
fi
if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$requirements_hash" ]; then
    echo "INFO :: installing toolchain/requirements.txt"
    "$venv_dir/bin/pip" install --quiet -r "$requirements" || { echo "ERROR :: pip install failed"; exit 1; }
    echo "$requirements_hash" > "$stamp"
fi
