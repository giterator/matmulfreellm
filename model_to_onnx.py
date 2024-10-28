import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

#Change here to our open-sourced model
name = 'ridger/MMfreeLM-370M'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
model.eval()

batch_size = 1
input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size

inputs = tokenizer(input_prompts, return_tensors="pt")

torch.onnx.export(model,
                  inputs['input_ids'],
                  "mmfreelm_370M.onnx",
                  opset_version=14,
                  input_names=['input_ids'],
                  output_names=['output'],)


