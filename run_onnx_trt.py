import onnx
import pycuda.driver as cuda
import pycuda.autoinit  # Necessary for initializing CUDA
import numpy as np
import tensorrt as trt

def build_engine(onnx_model_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # Adjust as necessary
    engine = builder.build_engine(network, config)

    return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        buffer = cuda.mem_alloc(size * dtype.itemsize)
        bindings.append(int(buffer))

        if engine.binding_is_input(binding):
            inputs.append(buffer)
        else:
            outputs.append(buffer)

    return inputs, outputs, bindings


if __name__ == "__main__":
    tok_count = 32
    batch_size = 1

    onnx_model = onnx.load("mmfreelm_370M.onnx")
    engine = build_engine("mmfreelm_370M.onnx")
    context = engine.create_execution_context()
    inputs, outputs, bindings = allocate_buffers(engine)

    
    input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size
    inputs = tokenizer(input_prompts, return_tensors="pt")
    input_ids = inputs.input_ids
    print(input_ids)
    cuda.memcpy_htod(inputs[0], input_data)

    context.execute_v2(bindings=bindings)
    output_data = np.empty(shape=(1, tok_count), dtype=np.float32)  # Adjust shape as needed
    cuda.memcpy_dtoh(output_data, outputs[0])

    generated_texts = tokenizer.batch_decode(output_data, skip_special_tokens=True)
    for prompt, generated in zip(input_prompts, generated_texts):
        print(f"Prompt: {prompt}\nGenerated: {generated}\n")
