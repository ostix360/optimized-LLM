import os

import bitnet
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import Trainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer

from layers import attention, mamba
from layers.jetmoe.utils import parallel_experts
from model.anemone_config import AnemoneConfig
from model.modeling_anemone import AnemoneForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

os.environ["WANDB_PROJECT"] = "Mixture of mixture (mod, moah moe)"

# bitlinear_new take 2 Go of vram for bsz=5 and 1B parameter
# bitnet.BitLinearNew.forward = nn.Linear.forward     # Replace all bitlinear to classic linear
# mamba.BitLinearNew.forward = nn.Linear.forward
attention.BitLinearNew.forward = nn.Linear.forward  # Replace bitlinear for attention
parallel_experts.BitLinearNew.forward = nn.Linear.forward
# moe.BitLinearNew.forward = nn.Linear.forward


# define the model configuration
capacity = 128
skip_blocks = 2
intermediate_size = 4048
num_hidden_layers = 14
hidden_size = 1024
expert_layer_period = 2



mom_config = AnemoneConfig(
    attn_layer_offset=5,
    attn_layer_period=6,
    attn_num_experts=16,
    attn_router_aux_loss_coef=0.05,
    attn_top_k=4,
    calc_logits_for_entire_prompt=True,
    capacity=capacity,
    expert_layer_offset=1,
    expert_layer_period=expert_layer_period,
    hidden_act="silu",
    hidden_size=hidden_size,
    initializer_range=0.02,
    intermediate_size=intermediate_size,
    mamba_conv_bias=True,
    mamba_d_conv=4,
    mamba_d_state=16,
    mamba_dt_rank=256,
    mamba_expand=2,
    mamba_inner_layernorms=True,
    mamba_proj_bias=False,
    mod_aux_loss_coef=0.01,
    mod_aux_routing=False,
    mod_routing=True,
    num_attention_heads=32,
    num_experts=8,
    num_experts_per_tok=2,
    num_hidden_layers=num_hidden_layers,
    num_key_value_heads=8,
    rms_norm_eps=1e-6,
    mlp_router_aux_loss_coef=0.001,
    skip_blocks=skip_blocks,
    sliding_window=None,
    use_cache=True,
    use_mamba_kernels=True,
    output_router_logits=True,
    vocab_size=tokenizer.vocab_size,
)

# initialize the model

model = AnemoneForCausalLM(mom_config)

max_seq_length = 512


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


textbooks_split = int(100_000 * 1)
eval_split = int(1_000 * 0.1)

t_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks", split=f"train[:{textbooks_split}]")
eval_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks", split=f"train[{textbooks_split}:{textbooks_split + eval_split}]")

key = "text"
train_dataset = t_ultra_textbooks.map(tokenize, batched=True, batch_size=10000, remove_columns=t_ultra_textbooks.column_names, )
eval_dataset = eval_ultra_textbooks.map(tokenize, batched=True, batch_size=10000, remove_columns=eval_ultra_textbooks.column_names, )

batch_size = 7
steps = len(train_dataset)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

run_name = f"n-h-l_{num_hidden_layers}_h-s_{hidden_size}_skip-b_{skip_blocks}_cap_{capacity}_int-sz_{intermediate_size}_exp-l-period_{expert_layer_period}_att-full-prec_1.58bits"

args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    load_best_model_at_end=False,
    warmup_steps=20,
    num_train_epochs=1,
    report_to=["wandb"],
    evaluation_strategy="steps",
    eval_steps=1_000*5//batch_size,
    learning_rate=5e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    bf16_full_eval=torch.cuda.is_bf16_supported(),
    fp16_full_eval=not torch.cuda.is_bf16_supported(),
    logging_steps=50 // batch_size,
    optim="adamw_8bit", # "galaore_adamw_8bit", save 1,5Go of memory for bsz=5 but slower to converge
    optim_target_modules=["anemone"],
    max_steps=steps // batch_size,
    save_total_limit=1,
    save_strategy="steps",
    save_steps=10_000,
    weight_decay=0.02,
    lr_scheduler_type="linear",
    output_dir="./trains",
    run_name=run_name,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Count number of trainable parameters for attn and the rest
def print_nb_trainable_params(model):
    bf16 = 0
    other = 0
    for name, param in model.named_parameters():
        if "attn" in name or "mamba" in name:
            bf16 += np.prod(param.shape)
        else:
            other += np.prod(param.shape)
    print(f"Attn + Mamba: {bf16 / 1_000_000}M, Other: {other / 1_000_000}M, Total: {(bf16 + other) / 1_000_000}M")

print_nb_trainable_params(model)


model.to("cuda", dtype=torch.bfloat16)
model.train()


tokenizer.push_to_hub("MoMv3-bf16") # Define the repository name

trainer.train(resume_from_checkpoint=False)
trainer.save_model("./model-anemone")

model.push_to_hub("MoMv3-bf16")
