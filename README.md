# Optimized LLM

The goal of this repository is to use the most optimized technic to train from scratch a LLM.

## Installation

Using a virtual environment is recommended.

```bash
pip install -r requirements.txt
```

## TODO

- [ ] Add [1.58 bits](https://arxiv.org/abs/2402.17764) linear layer for fastest inference but with quality loss
- [ ] Add [Galore](https://arxiv.org/abs/2403.03507) for the training
- [ ] Use [Jamba](https://arxiv.org/abs/2403.19887) as base architecture
- [ ] Use [Mixture of depth](https://arxiv.org/abs/2404.02258) 
- [ ] Use [Mixture of attention head](https://arxiv.org/abs/2306.04640) (See JetMoE)
- [ ] Add a script to train a LLM model from scratch


## Contributing

Contributions are welcome.

Please open a pull request with the proposed changes.

## License

Apache License 2.0
