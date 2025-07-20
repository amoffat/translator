#!/bin/bash
set -e

pip install --upgrade pip
# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    pip install --user poetry
    export PATH="$HOME/.local/bin:$PATH"
fi
poetry config virtualenvs.create false
poetry install
