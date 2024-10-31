import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys


import time
import json
import sys
import csv
import numpy as np
from typing import List, Tuple, Union
from tqdm import tqdm
from jtop import jtop
import multiprocessing
from multiprocessing import Process, Value

power_sample_period = 0.0005
runs = 10
throw_out = 0.25
verbose = True


def get_numpy_dtype(trt_dtype):
    """Convert TensorRT dtype to numpy dtype"""
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_
    }
    return mapping.get(trt_dtype, np.float32)

class TRTEngine:
    def __init__(self, engine_path):
        """
        Initialize TensorRT engine.
        Args:
            engine_path: Path to the TensorRT engine file (.engine)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine file
        print(f"Loading engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError("Failed to load engine")
            
        self.context = self.engine.create_execution_context()
        
        # Setup bindings and buffers
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        self.input_names = []
        self.output_names = []
        self.binding_shapes = {}
        self.binding_dtypes = {}
        
        print("\nAnalyzing engine bindings...")
        
        for binding_idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(binding_idx)
            shape = tuple(self.engine.get_binding_shape(binding_idx))
            dtype = get_numpy_dtype(self.engine.get_binding_dtype(binding_idx))
            
            size = np.zeros(shape).astype(dtype).nbytes
            
            print(f"Binding '{name}': shape={shape}, dtype={dtype}, size={size} bytes")
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(shape, dtype)
            cuda_mem = cuda.mem_alloc(size)
            
            # Save binding info
            self.binding_shapes[name] = shape
            self.binding_dtypes[name] = dtype
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding_idx):
                self.input_names.append(name)
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_names.append(name)
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
            
            self.bindings.append(int(cuda_mem))
        
        print(f"\nFound {len(self.input_names)} inputs and {len(self.output_names)} outputs")
        print("Input names:", self.input_names)
        print("Output names:", self.output_names)
        
        # Create CUDA stream
        self.stream = cuda.Stream()

    def infer(self, input_dict, tok_count):
        """
        Run inference on input data.
        Args:
            input_dict: Dictionary mapping input binding names to numpy arrays
        Returns:
            dict: Dictionary mapping output binding names to numpy arrays
        """
        # Validate inputs
        if set(input_dict.keys()) != set(self.input_names):
            raise ValueError(
                f"Input mismatch. Expected {self.input_names}, got {list(input_dict.keys())}"
            )
        
        # Process inputs
        for idx, input_name in enumerate(self.input_names):
            data = input_dict[input_name]
            
            if data.shape != self.binding_shapes[input_name]:
                raise ValueError(
                    f"Input shape mismatch for {input_name}. "
                    f"Expected {self.binding_shapes[input_name]}, got {data.shape}"
                )
            
            if data.dtype != self.binding_dtypes[input_name]:
                print(f"Warning: Converting input {input_name} from {data.dtype} to {self.binding_dtypes[input_name]}")
                data = data.astype(self.binding_dtypes[input_name])
            
            # Copy to pagelocked memory
            np.copyto(self.host_inputs[idx], data.ravel())
            
            # Transfer to GPU
            cuda.memcpy_htod_async(self.cuda_inputs[idx], self.host_inputs[idx], self.stream)

############################################################
        #Wrap with profiling code

        warm_up = int(runs * throw_out)
        total = 0

        manager = multiprocessing.Manager()
        power_samples = manager.list()
        inference_done = Value('i', 1)

        def poll_power():
            jetson = jtop()
            jetson.start()

            while inference_done.value == 0:
                power_samples.append(jetson.power["rail"]["VDD_CPU_GPU_CV"]["power"])
                time.sleep(power_sample_period) #e.g. 0.0005 is 0.5 ms -> power sampling rate
            jetson.close()

        start = time.time()

        for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
            if i == warm_up:
                ###########################################
                inference_done.value = 0
                power_process = Process(target=poll_power)
                power_process.start()
                ###########################################
                self.stream.synchronize()
                total = 0
                start = time.time()


            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            # Transfer outputs from GPU
            outputs = {}
            for idx, output_name in enumerate(self.output_names):
                # Transfer from GPU to host memory
                cuda.memcpy_dtoh_async(
                    self.host_outputs[idx], 
                    self.cuda_outputs[idx], 
                    self.stream
                )

            total += batch_size

        self.stream.synchronize()

        end = time.time()
        ##########################################################
        inference_done.value = 1
        ##########################################################
        elapsed = end - start

        throughput = tok_count * total / elapsed #tok/s

        ##############################################
        latency = elapsed / (tok_count * total) # per tok
        avg_power = np.mean(power_samples) #avg power consumption in mW
        avg_power /= 1000 #avg power consumption in W
        #avg_power = round(avg_power, 2) # W, 2 dp
        #print("Average power consumed (W):", avg_power)
        energy = avg_power * latency # energy per token
        ###################################

        if verbose:
            print(f"Latency per token: {latency:.2f} s, Throughput: {throughput:.2f} tok/s, Power: {avg_power:.2f} W, Energy per token: {energy:.2f} J/tok")

        return latency, throughput, avg_power, energy

    
    def __del__(self):
        """Cleanup cuda memory"""
        try:
            del self.cuda_inputs
            del self.cuda_outputs
            del self.stream
            del self.context
            del self.engine
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py path/to/model.engine")
        sys.exit(1)
        
    engine_path = sys.argv[1]
    name = sys.argv[2]
    batch_size = int(sys.argv[3])

    tokenizer = AutoTokenizer.from_pretrained(name)
    input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size
    input_ids = tokenizer(input_prompts, return_tensors="pt").input_ids.numpy()

    # Initialize engine
    engine = TRTEngine(engine_path)
    
    # Create sample input
    sample_inputs = {}
    for input_name in engine.input_names:
        shape = engine.binding_shapes[input_name]
        dtype = engine.binding_dtypes[input_name]
        sample_inputs[input_name] = input_ids.astype(dtype) #np.random.randn(*shape).astype(dtype)
    
    # Run inference
    print("\nRunning inference...")
    tok_count = len(input_prompts[0])
    print("Sequence lenth: ", tok_count)
    latency, throughput, avg_power, energy = engine.infer(sample_inputs, tok_count)
