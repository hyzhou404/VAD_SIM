# VAD End-to-End Inference 

This repo is the implementation of VAD test client for [HUGSIM benchmark](https://xdimlab.github.io/HUGSIM/)

The implementation is based on:
> [**VAD: Vectorized Scene Representation for Efficient Autonomous Driving**](https://arxiv.org/abs/2303.12077)
>
> [Bo Jiang](https://github.com/rb93dett)<sup>1</sup>\*, [Shaoyu Chen](https://scholar.google.com/citations?user=PIeNN2gAAAAJ&hl=en&oi=sra)<sup>1</sup>\*, Qing Xu<sup>2</sup>, [Bencheng Liao](https://github.com/LegendBC)<sup>1</sup>, Jiajie Chen<sup>2</sup>, [Helong Zhou](https://scholar.google.com/citations?user=wkhOMMwAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN)<sup>2</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>1,&#8224;</sup>
> 
> <sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> Horizon Robotics
>
> \*: equal contribution, <sup>&#8224;</sup>: corresponding author.
>
>[arXiv Paper](https://arxiv.org/abs/2303.12077), ICCV 2023

# Installation

Please refer to [UniAD](https://github.com/OpenDriveLab/UniAD) and [VAD](https://github.com/hustvl/VAD) installation instructions. In practice, UniAD and VAD can share the same conda environment.

Please change ${VAD_PATH} in tools/e2e.sh as the path on your machine.

# Launch Client

### Manually Launch
``` bash
zsh ./tools/e2e.sh ${CUDA_ID} ${output_dir}
```

### Auto Lauch
Client can be auto lauched by the HUGSIM closed-loop script.