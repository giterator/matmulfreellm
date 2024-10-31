import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, output_length):
        super().__init__()
        self.model = model
        self.output_length = output_length
        
    def forward(self, input_ids):
        # Forward pass with padding/truncating to fixed output length
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        
        # Handle the output dimension to be fixed
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        
        # Create output tensor with fixed dimension
        fixed_output = torch.zeros(
            (batch_size, self.output_length, vocab_size),
            dtype=logits.dtype,
            device=logits.device
        )
        
        # Copy actual outputs up to fixed length
        actual_length = min(logits.shape[1], self.output_length)
        fixed_output[:, :actual_length, :] = logits[:, :actual_length, :]
        
        return fixed_output


#Change here to our open-sourced model
name = 'ridger/MMfreeLM-370M'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
model.eval()

batch_size = 1
input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size

inputs = tokenizer(input_prompts, return_tensors="pt")

output_sequence_length = 32

    
model = ModelWrapper(model, output_sequence_length)

# Create dummy output with fixed shape
dummy_output = torch.zeros(
    batch_size,
    output_sequence_length,
    32000
)

torch.onnx.export(model,
                  inputs['input_ids'],
                  "mmfreelm_370M.onnx",
                  opset_version=17, #14
                  input_names=['input_ids'],
                  output_names=['output'],
                  dynamic_axes={
        'input': {0: 'batch_size', 1: 'token_length'},  # Specify dynamic axes
        # 'output': {0: 'batch_size', 1: "output_sequence_length"}
                  },
        do_constant_folding=True,
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        verbose=False,
        example_outputs=dummy_output
    )


