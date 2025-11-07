#!/bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

if ! ls ./build/text_det 1> /dev/null 2>&1; then
    echo "Executable do not exists. Stop"
    exit 1    
fi

./build/text_det \
    --model models/ch_PP-OCRv2_det.onnx \
    --image ./images/test.jpg \
    --threads 8 \
    --fixed_hw 960x540 \
    --bin_thresh 0.3 \
    --box_thresh 0.3
