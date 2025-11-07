#!/bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

if [ -d "./build" ]; then
    rm -rf ./build
fi

meson setup build
meson compile -C build