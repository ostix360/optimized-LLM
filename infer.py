from transformers import AutoTokenizer

from model.modeling_anemone import AnemoneForCausalLM

model_name="Ostixe360/Moah-MoE-1.58b-1B"

model = AnemoneForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "This is a story about"
inputs = tokenizer(prompt, return_tensors="pt")
model.to("cuda")
inputs.to("cuda")
output = model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0]))

