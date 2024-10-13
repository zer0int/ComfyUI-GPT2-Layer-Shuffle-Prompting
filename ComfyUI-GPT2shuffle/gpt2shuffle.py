import os
import re
import torch
from torch import Tensor, nn
import time 
import copy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2ShuffleBase:
    def __init__(self):
        super().__init__()
        self.models = {}
        self.original_model = None
        self.tokenizer = None

    def get_original_model(self, GPT2_model):

        # Load the model based on the GPT2_model
        if GPT2_model not in self.models:
            self.models[GPT2_model] = GPT2LMHeadModel.from_pretrained(GPT2_model)
            self.models[GPT2_model].eval()
            
            self.original_model = copy.deepcopy(self.models[GPT2_model])
            timestamp = time.time()
            print(f"\nOriginal GPT-2 model saved at: {timestamp}\n")    
        
        #if self.original_model is None:
        #    self.original_model = copy.deepcopy(self.models[GPT2_model])
        #    timestamp = time.time()
        #    print(f"Original GPT-2 model saved at: {timestamp}")

        timestamp = time.time()
        print(f"\nGPT-2 Model re-load invoked: {timestamp}\n")
        torch.cuda.empty_cache()
        model = copy.deepcopy(self.original_model)
        return model


class GPT2ShuffleNode(GPT2ShuffleBase):
    def __init__(self):
        self.models = {}
        self.tokenizer = None
        self.original_model = None
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
       # Defines input types for the GPT-2 node
        
        return {
            "required": {
                "text": ("STRING", {"default": "Sure, I will describe the image in great detail, focusing on lighting, mood, scene, details, subjects, and colors. Here is my description of a fantastical sci-fi robotic cat scene:", "multiline": True}),
                "max_response_length": ("INT", {"default": 128, "min": 50, "max": 512}),
                "GPT2_model": (["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], {"default": "gpt2"}),
                "shuffle_setting": (["None", "MLP", "Attn", "Layer", "LN_Identity"], {"default": "None"}),
                "shuffle_layer_range": ("STRING", {"default": "4,5,6,7,8"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "display": "number"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.01, "display": "number"}),
            },
        }
       
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "zer0int/GPT2-Shuffle"

    def generate(self, text, max_response_length, GPT2_model, shuffle_setting, shuffle_layer_range, temperature, top_p):
        
        model = self.get_original_model(GPT2_model) 

        num_return_sequences = 1
        max_length = max_response_length
        error_prompt = "empty"
        text = text.strip()
        
        if self.tokenizer == None:
            self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        
        
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        eos_id = self.tokenizer.eos_token_id

        max_length = min(max_length + len(input_ids), 1024)  # Adjusting max_length to account for prompt length
        num_layers = len(model.transformer.h) # Get number of layers in the selected GPT-2 model
        
        shuffle_layer_range_to_shuffle = [int(x.strip()) for x in shuffle_layer_range.split(",") if x.strip().isdigit()]
        
        # Check if all specified layers are within the valid range
        if all(0 <= layer_idx < num_layers - 2 for layer_idx in shuffle_layer_range_to_shuffle):
        
            if shuffle_setting is not "None":
                # Shuffle the Layers
            
                for layer_idx in shuffle_layer_range_to_shuffle:
                    layer1 = model.transformer.h[layer_idx]
                    layer2 = model.transformer.h[layer_idx + 1]

                    if shuffle_setting == "MLP":
                        layer1.mlp.c_fc.weight, layer2.mlp.c_proj.weight = layer2.mlp.c_fc.weight, layer1.mlp.c_proj.weight
                        layer1.mlp.c_fc.bias, layer2.mlp.c_proj.bias = layer2.mlp.c_fc.bias, layer1.mlp.c_proj.bias

                    elif shuffle_setting == "Attn":
                        layer1.attn.c_attn.weight, layer2.attn.c_proj.weight = layer2.attn.c_attn.weight, layer1.attn.c_proj.weight
                        layer1.attn.c_attn.bias, layer2.attn.c_proj.bias = layer2.attn.c_attn.bias, layer1.attn.c_proj.bias

                    elif shuffle_setting == "Layer":
                        layer1, layer2 = layer2, layer1

                    elif shuffle_setting == "LN_Identity":
                        layer1.ln_1 = torch.nn.Identity()
                        layer2.ln_2 = torch.nn.Identity()
    
        else:
            # Handle out-of-range layers gracefully
            error_prompt = f"ERROR: Specified Layers for GPT-2 out of range. Max for {GPT2_model}: {num_layers} -2 (don't shuffle the output!) = {num_layers - 2}"
            print(error_prompt)
           
        
        # Generate text
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            do_sample=True,
            no_repeat_ngram_size=2,
            eos_token_id=eos_id, #eos_token_id
            pad_token_id=eos_id, 
            length_penalty=-1.0,
        )

        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        new_prompt = output_texts[0].strip()
        new_prompt = new_prompt.replace(text, "").strip()
        
        if error_prompt is not "empty":
            return (error_prompt,)
        
        else:
            return (new_prompt,)


NODE_CLASS_MAPPINGS = {
    "GPT2ShuffleNode": GPT2ShuffleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT2ShuffleNode": "GPT-2 Shuffle & Prompt",
}
