from transformers import AutoTokenizer

from model.modeling_anemone import AnemoneForCausalLM

model_name="./model-anemone"

model = AnemoneForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

prompt = "This is a story about"
inputs = tokenizer(prompt, return_tensors="pt")
model.to("cuda")
inputs.to("cuda")
output = model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0]))

