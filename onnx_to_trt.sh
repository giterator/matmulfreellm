#!/bin/bash

trtexec="/usr/src/tensorrt/bin/trtexec"

$trtexec --onnx=mmfreelm_370M.onnx --saveEngine=mmfreelm_370M.engine --exportProfile=mmfreelm_370M.json --useSpinWait --separateProfileRun > mmfreelm_370M.log --verbose --profilingVerbosity=detailed --dumpProfile --fp16 --useCudaGraph
