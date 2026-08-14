#!/usr/bin/env bash
set -e

TRT_BIN=${TRT_BIN:-$(command -v trtexec || true)}
if [[ -z "$TRT_BIN" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRT_BIN=/usr/src/tensorrt/bin/trtexec
fi

echo ">>> Deactivating conda env (if any)"
conda deactivate || true

if [[ -z "$TRT_BIN" ]]; then
  echo "trtexec is not installed or not in PATH" >&2
  exit 2
fi

echo ">>> Building TensorRT engine: 512x640"
$TRT_BIN \
  --onnx=spnet_512_640.onnx \
  --saveEngine=spnet_512_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x512x640,depth:1x1x512x640,mask:1x1x512x640

echo ">>> Building TensorRT engine: 480x640"
$TRT_BIN \
  --onnx=spnet_480_640.onnx \
  --saveEngine=spnet_480_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x480x640,depth:1x1x480x640,mask:1x1x480x640

echo ">>> TensorRT engine build finished."
