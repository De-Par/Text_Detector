#!/bin/bash

if [ -d "./build" ]; then
    rm -rf ./build
fi

meson setup build
meson compile -C build