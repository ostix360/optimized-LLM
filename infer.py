import bitnet
from torch import nn
from transformers import AutoTokenizer

from model.modeling_anemone import AnemoneForCausalLM

model_name="Ostix/MoMv2-bf16"

bitnet.BitLinearNew.forward = nn.Linear.forward

model = AnemoneForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "This is a story about"
inputs = tokenizer(prompt, return_tensors="pt")
model.to("cuda")
inputs.to("cuda")
output = model.generate(**inputs, max_length=100, repetition_penalty=1.4,)
print(tokenizer.decode(output[0]))

