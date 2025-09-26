# Deep Reinforcement Learning Course from Hugging Face
[![Tests](https://github.com/anton-dergunov/hugging-face-deep-rl/actions/workflows/tests.yaml/badge.svg)](https://github.com/anton-dergunov/hugging-face-deep-rl/actions/workflows/tests.yaml)

https://huggingface.co/learn/deep-rl-course/en/unit0/introduction
https://simoninithomas.github.io/deep-rl-course/

Instructions:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv with UV)"
uv pip install pip
```

Python 3.12 is recommended, because at the moment there is no good support for `pygame` with Python 3.13 yet on MacOS.

```bash
uv pip install -r requirements.txt
```

## Telegram Notifications

TODO Also describe how to create Telegram bot.

