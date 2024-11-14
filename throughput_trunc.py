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

from functools import partial

import mmfreelm
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import time
import json
import sys
import csv
import numpy as np
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
from jtop import jtop
import multiprocessing
from multiprocessing import Process, Value
import pandas as pd

import torch
import re

power_sample_period = 0.0005
runs = 10
throw_out = 0.25
verbose = True

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class LayerProfiler:
    def __init__(self, model_name: str, tok_count, batch_size):
        """
        Initialize the profiler with a model name.
        
        Args:
            model_name (str): HuggingFace model name (e.g., 'bert-base-uncased')
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        self.model.to(self.device)
#        self.model.half()
        self.model.eval()

        self.tok_count = tok_count
        self.batch_size = batch_size


#        print(self.model)


    def print_layer_hierarchy(self) -> None:
        """
        Print the model's layer hierarchy for reference.
        """
        def print_module(module, prefix=""):
            for name, child in module.named_children():
                print(f"{prefix}{name}: {child.__class__.__name__}")
                print_module(child, prefix + "  ")
        
        print("Model Layer Hierarchy:")
        print_module(self.model)


    def _get_matching_layers(self, section_name: str) -> List[str]:
        """
        Get all layer names that match the given section name pattern.
        
        Args:
            section_name (str): Name pattern to match (supports wildcards *)
            
        Returns:
            List[str]: List of matching layer names
        """
        # Convert the wildcard pattern to regex
        pattern = section_name.replace("*", ".*")
        regex = re.compile(pattern)
        
        matching_layers = []
        for name, _ in self.model.named_modules():
            if name and regex.match(name):
                matching_layers.append(name)
        
        return matching_layers


    def profile_layers(self, text: str, warm_up_runs: int = 2, profile_runs: int = 8) -> pd.DataFrame:
        """
        Profile each layer's execution time.
        
        Args:
            text (str): Input text to process
            warm_up_runs (int): Number of warm-up runs before profiling
            profile_runs (int): Number of runs to average for profiling
            
        Returns:
            pd.DataFrame: Profiling results for each layer
        """

        # Prepare input
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()

        total_runs = warm_up_runs + profile_runs

        # Dictionary to store latencies
        layer_latencies = {}

        # Define a hook function to measure latency
        def hook(layer_name, module, input, output):
            layer_latencies[layer_name] = time.time() - layer_latencies[layer_name]

        def pre_hook(layer_name, module, input):
            layer_latencies[layer_name] = time.time()

        names = ["lm_head", "embeddings"]
        chunks = [self.model.lm_head, self.model.model.embeddings]
        # Register hooks to each layer
        for name, layer in zip(names,chunks): #self.model.named_modules():
            layer.register_forward_pre_hook(partial(pre_hook,name))
            layer.register_forward_hook(partial(hook,name))


        collated = []
        total = 0
        start_time = time.time()
        for i in tqdm(range(total_runs), disable=not verbose, desc="Benchmarking"):
            if i == warm_up_runs:
                total = 0
                collated = []
                layer_latencies = {}
                start_time = time.time()
            
            outputs = self.model.generate(input_ids, max_length=self.tok_count,  do_sample=True, top_p=0.4, temperature=0.6)
            collated.append(layer_latencies)
            layer_latencies = {}
            total += self.batch_size

        end_time = time.time()
        elapsed = end_time - start_time

        data = pd.DataFrame(collated).mean()
        data["avg_total"] = elapsed / total_runs
        return data


if __name__ == "__main__":
#    tok_count = 128
#    batch_size = 20

    for batch_size in [1,15,20]:
        for tok_count in [32,64,128]:
        
            profiler = LayerProfiler("ridger/MMfreeLM-370M", tok_count, batch_size)

        #    profiler.print_layer_hierarchy()

            input_prompts = ["In a shocking finding, scientist discovered a herd of unicorns living in a remote, "] * batch_size
        
            print("Batch Size: ", batch_size)
            print("Output Token Count: ", tok_count)
            # Basic layer profiling
            results_df = profiler.profile_layers(input_prompts)
            print("\nLayer-wise profiling results:")
            print(results_df)

            print("Embedding contribution (%): ", 100 * (results_df["embeddings"] / results_df["avg_total"]))
            print("Lm_head contribution (%): ", 100 * (results_df["lm_head"] / results_df["avg_total"]))
