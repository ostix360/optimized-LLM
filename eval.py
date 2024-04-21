import argparse

import bitnet
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer, EvalPrediction

from layers import attention, mamba
from layers.jetmoe.utils import parallel_experts
from model.modeling_anemone import AnemoneForCausalLM




def eval(model_name, max_seq_length=512):
    if "mixed-precision" in model_name or "v4" in model_name:
        attention.BitLinearNew.forward = nn.Linear.forward  # Replace bitlinear for attention
        parallel_experts.BitLinearNew.forward = nn.Linear.forward
    if "bf16" in model_name:
        bitnet.BitLinearNew.forward = nn.Linear.forward
    if "M-A-mixed-precision" in model_name:
        attention.BitLinearNew.forward = nn.Linear.forward
        parallel_experts.BitLinearNew.forward = nn.Linear.forward
        mamba.BitLinearNew.forward = nn.Linear.forward
    model = AnemoneForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks", split=f"train[:{1000}]", )

    key = "text"
    def tokenize(element):
        outputs = tokenizer(
            element[key],
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == max_seq_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    eval_dataset = eval_ultra_textbooks.map(tokenize, batched=True, batch_size=1000,
                                            remove_columns=eval_ultra_textbooks.column_names, )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    batch_size = 10

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)

    accelerator = Accelerator()

    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss).unsqueeze(0))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    print(f"Loss: {loss.item()}, Perplexity: {perplexity.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MoMv4-1.58bits")
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()
    eval("Ostixe360/" + args.model_name, args.max_seq_length)
