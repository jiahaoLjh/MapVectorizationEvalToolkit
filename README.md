# MapVectorizationEvalToolkit

This repository contains the official code of the evaluation toolkit proposed in the NeurIPS 2023 paper "Online Map Vectorization for Autonomous Driving". We propose a rasterization-based evaluation process which converts the vectorized predictions and ground truth instances into rasterized maps respectively in the bird's-eye-view, on which we perform evaluation with mask-based intersection-over-union.

The main repository of our proposed method can be found [here](https://github.com/ZhangGongjie/MapVR). Also, checkout [our NeurIPS 2023 paper](https://arxiv.org/abs/2306.10502) for more details.

## Installation and Usage

Clone this repository to your local working directory. The main evaluator is located in ``map_evaluator.py``. The default evaluation setting is provided in ``eval_config.py``.

We include a demo script showing the procedure of using the evaluator in ``demo.py``. Sample data is also included in ``sample_data.json`` consisting of both ground truth data and predictions.

To perform the evaluation, run

```bash
python demo.py --cfg eval_config.py
```
## Further clarification on evaluation settings

The evaluation settings used in all our evaluation, including both the parameters for rasterization and the specific evaluation metrics, are provided in ``eval_config.py``.

An example to include more classes for evaluation:

```python
LINE_CLASSES = {"divider": 0, "boundary": 2, "new_line_class_A": 3, "new_line_class_B": 4}
POLYGON_CLASSES = {"ped_crossing": 1, "new_polygon_class_A": 5}
```

The region of interest for map element evaluation is $\pm 15m$ (left-right) and $\pm 30m$ (front-rear). We use a resolution of $480 \times 240$ for the rasterized map. A dilation of size 5 (2px on each side) is applied on each rasterized map element. We use a loose IoU threshold (0.25:0.50:0.05) for line elements and a tight IoU threshold (0.50:0.75:0.05) for polygon elements because line elements are in general more difficult to precisely localize.

The default evaluation metrics are defined as

```python
QUERY_LINE = [
    ("AP", "all", "all", 100),
    ("AP", "all", "divider", 100),
    ("AP", "all", "boundary", 100),
    ("AP", 0.25, "all", 100),
    ("AP", 0.25, "divider", 100),
    ("AP", 0.25, "boundary", 100),
    ("AP", 0.50, "all", 100),
    ("AP", 0.50, "divider", 100),
    ("AP", 0.50, "boundary", 100),
    ("AR", "all", "all", 1),
    ("AR", "all", "all", 10),
    ("AR", "all", "all", 100),
    ("AR", "all", "divider", 100),
    ("AR", "all", "boundary", 100),
]
QUERY_POLYGON = [
    ("AP", "all", "ped_crossing", 100),
    ("AP", 0.50, "ped_crossing", 100),
    ("AP", 0.75, "ped_crossing", 100),
    ("AR", "all", "ped_crossing", 1),
    ("AR", "all", "ped_crossing", 10),
    ("AR", "all", "ped_crossing", 100),
]
```
which can be extended for additional IoU thresholds of interest, or new classes to be evaluated.

We recommend using the same settings of evaluation for a fair comparison among this future line of research.

## Citation

If you find our evaluation toolkit useful, please consider citing:

```bibtex
@inproceedings{zhang2023online,
  title={Online Map Vectorization for Autonomous Driving: A Rasterization Perspective},
  author={Zhang, Gongjie and Lin, Jiahao and Wu, Shuang and Song, Yilin and Luo, Zhipeng and Xue, Yang and Lu, Shijian and Wang, Zuoguan},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgement

Our evaluation toolkit is following the _Average Precision_-based metric in an object detection style inspired by [_COCOeval_](https://github.com/cocodataset/cocoapi).
