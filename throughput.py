import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

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

def gpu_benchmark(model, input_ids, batch_size, tok_count) -> float:

    warm_up = int(runs * throw_out)
    total = 0

    #############
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

    #################


    start = time.time()

    for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
        if i == warm_up:
            ###########################################
            inference_done.value = 0
            power_process = Process(target=poll_power)
            power_process.start()
            ###########################################
#            if is_cuda:
#                torch.cuda.synchronize()
            total = 0
            start = time.time()

        outputs = model.generate(input_ids, max_length=tok_count,  do_sample=True, top_p=0.4, temperature=0.6)
#        print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
#        model(input)
        total += batch_size

#    if is_cuda:
#        torch.cuda.synchronize()

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


def benchmark(model, input_ids, batch_size, tok_count):
    latency, throughput, power, energy = gpu_benchmark(model, input_ids, batch_size, tok_count)

    results = [
        round(latency, 2),
        round(throughput, 2),
        round(power, 2),
        round(energy, 2)]

    return results


if __name__ == "__main__":
    name = 'ridger/MMfreeLM-370M'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()

    for batch_size in [15, 20]: #4, 8
        print(f"Batch Size: {batch_size}")
        for tok_count in [32, 64, 128]:
            print(f"Token count: {tok_count}")
            input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size
            input_ids = tokenizer(input_prompts, return_tensors="pt").input_ids.cuda()
            results = benchmark(model, input_ids, batch_size, tok_count)


