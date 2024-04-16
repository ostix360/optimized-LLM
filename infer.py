import argparse

import bitnet
from torch import nn
from transformers import AutoTokenizer

from layers import attention, mamba
from layers.jetmoe.utils import parallel_experts
from model.modeling_anemone import AnemoneForCausalLM

def infer(model_name: str, prompt):
    if "mixed-precision" in model_name:
        attention.BitLinearNew.forward = nn.Linear.forward  # Replace bitlinear for attention
        parallel_experts.BitLinearNew.forward = nn.Linear.forward
    elif "bf16" in model_name:
        bitnet.BitLinearNew.forward = nn.Linear.forward
    elif "M-A-mixed-precision" in model_name:
        attention.BitLinearNew.forward = nn.Linear.forward
        parallel_experts.BitLinearNew.forward = nn.Linear.forward
        mamba.BitLinearNew.forward = nn.Linear.forward
    model = AnemoneForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    model.to("cuda")
    inputs.to("cuda")
    output = model.generate(**inputs, max_length=100, repetition_penalty=1.4,)
    print(tokenizer.decode(output[0]))

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="This is a story about")
    parser.add_argument("--model_name", type=str, default="MoMv3-mixed-precision")

    args = parser.parse_args()

    model_name = "Ostixe360/"+args.model_name
    prompt = args.prompt
    infer(model_name, prompt)

