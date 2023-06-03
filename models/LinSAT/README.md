# LinSAT

Our implementation of the following paper:
* Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang, Junchi Yan. "LinSATNet: The Positive Linear Satisfiability Neural Networks." _ICML 2023_.
  [[paper]](https://openreview.net/forum?id=D2Oaj7v9YJ)
  
LinSAT aims at encoding constraints into neural networks, through projecting input variables with non-negative constraints turough a differentiable satisfiability layer based on an extension of the classic Sinkhorn algorithm involving jointly encoding multiple sets of marginal distributions. 

The partial graph matching problem and the permutation constraints can also be explicitly formulated and resolved via LinSAT.

We choose NGMv2 as the backend graph matching network, and the file ``models.NGM.model_v2_linsat.py`` can be referred to for details.

LinSAT has been uploaded to PyPI and you can simply install the package by
```shell
$ pip install linsatnet
```
and import the LinSAT layer by adding
``
from LinSATNet import linsat_layer
``
in your code.

Note that LinSAT is capable of encoding non-negative constraints. You can adopt LinSAT to solve your own problem by devising the problem constraints.

## Benchmark Results
### PascalVOC - 2GM (unfiltered)
* NGMv2-LinSAT
  
  experiment config: ``experiments/vgg16_ngmv2_linsat_voc-all.yaml``

| model                  | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [NGMv2-LinSAT](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#linsat) | 2023 | 0.4547 | 0.6489 | 0.5857 | 0.4915 | 0.8628 | 0.6631 | 0.6079 | 0.6230 | 0.4467 | 0.6121 | 0.4867 | 0.6254 | 0.5974 | 0.6338 | 0.4791 | 0.9253 | 0.5473 | 0.3542 | 0.7674 | 0.8120 | 0.6112 |

Note: In this experiment, we assume that the number of matches `k` is given, and we only retain `k` matches during inference. Therefore, it is not fair to compare the above empirical result with that of other partial GM approaches.

## Credits and Citation

Please cite the following paper if you use this model in your research:
```
@InProceedings{WangICML23, 
   title={LinSATNet: The Positive Linear Satisfiability Neural Networks},
   author={Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang, Junchi Yan},
   booktitle={ICML},
   year={2023}
}
```
