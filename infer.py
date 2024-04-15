import bitnet
from torch import nn
from transformers import AutoTokenizer

from layers import attention
from layers.jetmoe.utils import parallel_experts
from model.modeling_anemone import AnemoneForCausalLM

model_name="./model-anemone"

attention.BitLinearNew.forward = nn.Linear.forward  # Replace bitlinear for attention
parallel_experts.BitLinearNew.forward = nn.Linear.forward

model = AnemoneForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "This is a story about"
inputs = tokenizer(prompt, return_tensors="pt")
model.to("cuda")
inputs.to("cuda")
output = model.generate(**inputs, max_length=100, repetition_penalty=1.4,)
print(tokenizer.decode(output[0]))

