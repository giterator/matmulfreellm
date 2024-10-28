import transformers.generation.utils

original_generate = transformers.generation.utils.GenerationMixin.generate

def generate_patch(*args, **kwargs):
   # Force non-distributed mode
   kwargs['synced_gpus'] = False
   return original_generate(*args, **kwargs)

transformers.generation.utils.GenerationMixin.generate = generate_patch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

#Change here to our open-sourced model
name = 'ridger/MMfreeLM-370M'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda().half()

batch_size = 2
input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size
input_ids = tokenizer(input_prompts, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=32,  do_sample=True, top_p=0.4, temperature=0.6)

#print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)


for prompt, generated in zip(input_prompts, generated_texts):
    print(f"Prompt: {prompt}\nGenerated: {generated}\n")
