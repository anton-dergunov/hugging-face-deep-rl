# Deep Reinforcement Learning Course from Hugging Face

[![Tests](https://github.com/anton-dergunov/hugging-face-deep-rl/actions/workflows/tests.yaml/badge.svg)](https://github.com/anton-dergunov/hugging-face-deep-rl/actions/workflows/tests.yaml)

This repository provides a set of notebooks (and some shared code in `rlcourse` library)
which I developed when I worked on the Deep Reinforcement Learning Course from Hugging Face.

References:
- https://huggingface.co/learn/deep-rl-course/en/unit0/introduction
- https://simoninithomas.github.io/deep-rl-course/

## Setup Instructions

Python 3.12 is recommended, because at the moment there is no good support for `pygame` with Python 3.13 yet on MacOS.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv with UV)"
uv pip install pip
uv pip install -r requirements.txt
```

## Configuring Telegram Notifications

TODO Also describe how to create Telegram bot.

## Notebooks

- [CartPole with PPO](blob/main/notebooks/00_cartpole.ipynb)
- [Lunar Lander with PPO](blob/main/notebooks/01_lunar_lander.ipynb)
- [Frozen Lake with Q-Learning](blob/main/notebooks/02_frozen_lake_q_learning.ipynb)

TODO Describe these notebooks.
