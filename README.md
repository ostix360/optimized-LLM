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
- [ ] Use [Mixture of depth](https://arxiv.org/abs/2404.02258) 
- [x] Use [Mixture of attention head](https://arxiv.org/abs/2210.05144) (code from [JetMoE](https://github.com/myshell-ai/JetMoE))
- [x] Add a script to train a LLM model from scratch


## Test

To test the first model, that has 1.58 bits linear layer, jamba base architecture and moah:



you can clone this repo at this commit and run the following command:

```bash
python infer.py
```


## Contributing

Contributions are welcome.

Please open a pull request with the proposed changes.

## License

Apache License 2.0
