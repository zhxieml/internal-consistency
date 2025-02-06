# Internal Consistency

This is the source code accompanying [Calibrating Reasoning in Language Models with Internal Consistency](https://arxiv.org/abs/2405.18711).

## Requirement

Create a virtual environment and install the dependencies.

```
conda create -n ic python=3.10  && conda activate ic
pip install -r requirements.txt
```

## Quickstart
```
bash scripts/decode.sh
bash scripts/extract.sh
bash scripts/eval_sc.sh