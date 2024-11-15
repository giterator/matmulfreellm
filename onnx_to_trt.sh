#!/bin/bash

trtexec="/usr/src/tensorrt/bin/trtexec"

$trtexec --onnx=mmfreelm_370M.onnx --int8 --saveEngine=mmfreelm_370M.engine > mmfreelm_370M.log --useCudaGraph
