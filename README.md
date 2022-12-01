# Source code of 'CCSL: A Causal Structure Learning Method from Multiple Unknown Environments'

## Overview

Provides the CCSL method to learn causal structure from multiple unknown environments.

Details of the algorithm can be found in "CCSL: A Causal Structure Learning Method from Multiple Unknown Environments.



# Reproduce synthetic experiments:

Run Our model :

```
nohup bash run.sh &
```
the results will be stored in 'output' folder and to summarize the results with all settings :

```
python result_combine.py
```
the results under different settings will be stored in 'result' folder;



# Citation

If you find this useful for your research, we would be appreciated if you cite the following papers:

```
@article{chen2021ccsl,
  title={CCSL: A Causal Structure Learning Method from Multiple Unknown Environments},
  author={Chen, Wei and Wu, Yunjin and Cai, Ruichu and Chen, Yueguo and Hao, Zhifeng},
  journal={arXiv preprint arXiv:2111.09666},
  year={2021}
}
```

