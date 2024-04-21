# Optimized LLM

The goal of this repository is to use the most optimized technic to train from scratch a LLM and to be the fastest at inference time.

I call this llm anemone, but I'm not satisfied with this name.
why not MoM for mixture of mixture ?

## Installation

Using a virtual environment is recommended.


```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121 
```


```bash
pip install -r requirements.txt
```


## TODO

- [x] Add [1.58 bits](https://arxiv.org/abs/2402.17764) linear layer for fastest inference but with quality loss (code from [1.58 bits](https://github.com/kyegomez/BitNet))
- [x] Add [Galore](https://arxiv.org/abs/2403.03507) for the training
- [x] Use [Jamba](https://arxiv.org/abs/2403.19887) as base architecture (code from [jamba](https://huggingface.co/ai21labs/Jamba-v0.1))
- [x] Use [Mixture of depth](https://arxiv.org/abs/2404.02258)  (code from [github](https://github.com/sramshetty/mixture-of-depths))
- [x] Use [Mixture of attention head](https://arxiv.org/abs/2210.05144) (code from [JetMoE](https://github.com/myshell-ai/JetMoE))
- [x] Add a script to train a LLM model from scratch
- [x] Add evaluation script
- [ ] Fix inference 
- [ ] Use a filtered dataset such as for [rho](https://github.com/microsoft/rho)

## Test

### Model Without mixture of depth

To test the first model, that has 1.58 bits linear layer, jamba base architecture and moah, you can clone this repo at this [commit](https://github.com/ostix360/optimized-LLM/tree/8878e0f0bd764f85ce2ea56790a95f9837fb2fe4):


and run the following command:

```bash
python infer.py
```

### Model With mixture of depth

To test the second model, that has 1.58 bits linear layer, jamba base architecture, moah and mod, you can clone this repo at this [commit](https://github.com/ostix360/optimized-LLM/tree/7cc2e6f39b69864e0cc80ca8b767229c536e6793)

You can start the training process by running the following command:

```bash
python train.py
```

and compare the results with the first model.

You can also run the following command to test the inference:

```bash
python infer.py
```

### MoMv2-bf16 

This [model](https://huggingface.co/Ostixe360/MoMv2-bf16) is a mixture of mixture (Mod, MoD, MoAH) with jamba base architecture.

This model doesn't contain any 1.58 bits linear layer. 

The difference between this model and the previous one is the use of a softmax function to weight the token for the mod and this break the causality and that's maybe why the model output no sense text.

You can also run the following command for [this commit](https://github.com/ostix360/optimized-LLM/tree/e223f9fa7bd136cfd836ceee522e1d98b97b08af) to test the inference:

```bash
python infer.py
```

### MoMv3 

This [model](https://huggingface.co/Ostixe360/MoMv3-mixed-precision) is a mixture of mixture (Mod, MoD, MoAH) with jamba base architecture.

All mamba, routers, moe, mlp are 1.58 bits linear layer. The linear layers in the attention mechanism are not 1.58 bits linear layers.

You can also run the following command to test the inference and change MoMv3 by MoMv3-mixed-precision in the file:

```bash
python infer.py --prompt "This is the story of"
```

To run the full 1.58bits model, you can run the following command:

```bash
python infer.py --prompt "This is the story of" --model "MoMv3-1.58bits"
```

To run the model with mamba and attention in bf16 and the rest in 1.58bits, you can run the following command:

```bash
python infer.py --prompt "This is the story of" --model "MoMv3-M-A-mixed-precision"
```

To run the full bf16 model, you can run the following command:

```bash
python infer.py --prompt "This is the story of" --model "MoMv3-bf16"
```

### MoMv4

This [model](https://huggingface.co/Ostixe360/MoMv4-1.58bits) is a mixture of mixture (Mod, MoD, MoAH) with jamba base architecture.

All mamba, routers, moe, mlp are 1.58 bits linear layer. The linear layers in the attention mechanism are in bf16 precision.

The total number of parameters is 1.7% in bf16 and the rest in 1.58bits.

The total active parameters is in a first estimation 87M parameters over 1B parameters.

Each mlp layer has 12.4M parameters each token can pass through 7 mlp layers and 7 mlp expert layer which is 2*7mlp layer.
For mlp, the number of parameters is 12.4M * 21 = 261.1M parameters.
We add the mamba and attention parameters that are near 107M parameters.
And only 1/4 of the tokens pass through a block. So the total number of active parameters is 368.1/4 = 87M parameters.

To test the inference, you can run the following command:

```bash
python infer.py --prompt "This is the story of" --model "MoMv4-1.58bits"
```

and 
    
```bash
python infer.py --prompt "This is the story of" --model "MoMv4-bf16"
```

## Evaluation

To evaluate the model, you can run the following command:

```bash
python eval.py --model "MoMv4-1.58bits" --max_seq_length 512
```

which has a loss of 2.62 and a perplexity of 13.77.

You can also run the evaluation for the full bf16 model:

```bash
python eval.py --model "MoMv4-bf16" --max_seq_length 512
```

which has a loss of 2.53 and a perplexity of 12.59.

The bf16 version is a bit better


## Conclusion

We can see ([here](https://wandb.ai/ostix360/Mixture%20of%20mixture%20(mod,%20moah%20moe))) that the baseline (MoMv3-bf16) has a similar loss curve as the attention and mamba in bf16 and the rest in 1.58bits (MoMv3-M-A-mixed-precision) and the attention in bf16 and the rest in 1.58bits (MoMv3-mixed-precision).

Furthermore, the training is faster when attention is not at 1.58bits, and it takes lesser vram too.

To train a model with long context and a lot of parameters, for a fast and low memory inference, I found that using the jamba architecture and with all linear layer in 1.58bits excepted for the attention mechanism's layers can be a godd strategy.
With only 1.7% of parameters in bf16, the model can fit in cheap gpu during inference.
Moreover, using all the mixture (moeh, moe and mod) you can train the model faster with only a few active parameters.


## Contributing

Contributions are welcome.

Please open a pull request with the proposed changes.

## License

Apache License 2.0
