# Internal Consistency

This is the source code accompanying [Calibrating Reasoning in Language Models with Internal Consistency](https://arxiv.org/abs/2405.18711).

## Requirement

Create a virtual environment and install the dependencies.

```
conda create -n ic python=3.10  && conda activate ic
pip install -r requirements.txt
```

## Quickstart
```shell
bash scripts/decode.sh
bash scripts/extract.sh
bash scripts/eval_sc.sh
```

## Citation
```
@article{xie2024calibrating,
  title={Calibrating Reasoning in Language Models with Internal Consistency},
  author={Xie, Zhihui and Guo, Jizhou and Yu, Tong and Li, Shuai},
  journal={arXiv preprint arXiv:2405.18711},
  year={2024}
}
```