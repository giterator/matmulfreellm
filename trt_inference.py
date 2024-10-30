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

    def infer(self, input_dict):
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
        
        # Synchronize
        self.stream.synchronize()
        
        # Copy to output dict and reshape
        for idx, output_name in enumerate(self.output_names):
            shape = self.binding_shapes[output_name]
            outputs[output_name] = np.array(self.host_outputs[idx]).reshape(shape)
        
        return outputs
    
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
    outputs = engine.infer(sample_inputs)
    
    # Print shapes
    print("\nResults:")
    print("Input shapes:", {name: arr.shape for name, arr in sample_inputs.items()})
    print("Output shapes:", {name: arr.shape for name, arr in outputs.items()})

#    print(outputs["output"].shape)

    output = np.argmax(outputs["output"], axis=2)
    output = output.astype(int).tolist()
##
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
#
    for prompt, generated in zip(input_prompts, generated_texts):
        print(f"Prompt: {prompt}\nGenerated: {generated}\n")
   


#
#
#
#if __name__ == "__main__":
#    engine_path = "mmfreelm_370M.engine"
#    name = 'ridger/MMfreeLM-370M'
#    batch_size = 1
#    iterations = 10
#    warmup_iterations = int(0.25 * iterations)
##
##    tokenizer = AutoTokenizer.from_pretrained(name)
##    input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size
##    input_ids = tokenizer(input_prompts, return_tensors="pt").input_ids.numpy()
##    input_ids = input_ids.astype(np.float16)
#
#
#    # Initialize engine
#    engine = TRTEngine(engine_path)
#    
#    # Create sample input
#    input_shape = engine.binding_shapes["input"]
#    sample_input = {
#        "input": np.random.randn(*input_shape).astype(np.float32)
#    }
#
#    
#    # Run inference
#    output = engine.infer(sample_input)
#
#    print("Input shapes:", {name: arr.shape for name, arr in sample_input.items()})
#    print("Output shapes:", {name: arr.shape for name, arr in outputs.items()})
#
#
