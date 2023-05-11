# EnvPipe: Performance-preserving DNN Training Framework for Saving Energy (USENIX ATC 2023)
EnvPipe (**Env**elope + **Pipe**line Parallelism) is an energy-saving DNN training framework aiming to maximize energy saving while maintaining negligible performance slowdown. EnvPipe takes advantage of slack time created by bubbles in pipeline parallelism. It schedules pipeline units to place bubbles after pipeline units as frequently as possible and then stretches the execution time of pipeline units by lowering the SM frequency. During this process, EnvPipe does not modify hyperparameters or pipeline dependencies, preserving the original accuracy of the training task. It selectively lowers the SM frequency of pipeline units to avoid performance degradation. The current prototype of EnvPipe is implemented on top of [DeepSpeed](https://github.com/microsoft/DeepSpeed) with [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml) to adjust the SM frequency of GPUs.

<!-- ## Publication
- Paper: to be updated
- Authors: Sangjin Choi, Inhoe Koo, Jeongseob Ahn, Myeongjae Jeon, Youngjin Kwon -->

## Contents 
- [1. Tested environment](#1-tested-environment)
    - [1.1. Hardware](#11-hardware)
        - [1.1.1. Single-node](#11-single-node)
            - [Single-V100](#111-single-v100)
            - [Single-3090](#112-single-3090)
        - [1.1.2. Multi-node](#12-multi-node)
            - [Multi-V100](#121-multi-v100)
    - [1.2. Software](#12-software)
- [2. Dependent package installation](#2-dependent-package-installation)
- [3. Download source code](#3-download-source-code)
- [4. Install EnvPipe](#4-install-envpipe)
- [5. Setup benchmarks and datasets](#5-setup-benchmarks-and-datasets)
- [6. Run benchmarks](#6-run-benchmarks)
    - [6.1. Single node](#61-single-node)
    - [6.2. Multi node](#62-multi-node)


## 1. Tested environment
### 1.1. Hardware
#### 1.1.1. Single-node
##### Single-V100

- AWS P3.8xLarge instance
    - GPU: NVIDIA V100 (16GB) x 4ea
    - CPU: Intel(R) Xeon(R) CPU E5-2686 v4 (32 vCPUs) @ 2.30GHz
    - Memory: 244GB DDR4 DRAM

##### Single-3090

- Local testbed 
    - GPU: NVIDIA RTX3090 (24GB) x 4ea
    - CPU: Intel(R) Xeon(R) Gold 6326 CPU (64 vCPUs) @ 2.90GHz
    - Memory: 256GB DDR4 DRAM

#### 1.1.2. Multi-node
##### Multi-V100

- 2 nodes with AWS P3.16xLarge instance, each with
    - GPU: NVIDIA V100 (16GB) x 8ea
    - CPU: Intel(R) Xeon(R) CPU E5-2686 v4 (64 vCPUs) @ 2.30GHz
    - Memory: 488GB DDR4 DRAM
    - Network: 25Gbps

### 1.2. Software

- Operating system: Ubuntu 20.04 (**Single-V100**, **Multi-V100**), Ubuntu 22.04 (**Single-3090**)
- Kernel version: 5.15.0-1033-aws (**Single-V100**, **Multi-V100**), 5.15.0-70-generic (**Single-3090**)
- CUDA version: 11.6
- NVIDIA driver version: 510.39.01
- PyTorch version: 1.13.0+cu116

## 2. Dependent package installation

- build-essential
```shell
$ sudo apt update
$ sudo apt install build-essential
```

- Activate virtual environment

```shell
$ python -m venv env
$ source env/bin/activate
```

- CUDA Toolkit 11.6 

Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local) compatible with the target platform.
```shell
$ wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
$ sudo sh cuda_11.6.0_510.39.01_linux.run

Please make sure that
 -   PATH includes /usr/local/cuda-11.6/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.6/lib64
```

- PyTorch (1.13.0+cu116)

Install [PyTorch](https://pytorch.org/get-started/locally/) matching the CUDA version of the target platform.
```shell
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

- Apex

[NVIDIA Apex](https://github.com/NVIDIA/apex) is a PyTorch extension that provides tools for easy mixed precision and distributed training in PyTorch. Apex is required to run Megatron-Deepspeed benchmark. 
```shell
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install packaging
$ sudo apt-get install python3-dev
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## 3. Download source code

```shell
$ git clone https://github.com/casys-kaist/EnvPipe.git
$ cd EnvPipe
```

## 4. Install EnvPipe

```shell
$ cd DeepSpeed
$ pip install pynvml
$ pip install wheel
$ pip install .
$ pip list | grep deepspeed
deepspeed          0.7.4+unknown
```

## 5. Setup benchmarks and datasets

- Megatron-Deepspeed

```shell
$ cd Megatron-DeepSpeed
$ pip install -r requirements.txt
```

- GPT Vocab

Download gpt2-vocab.json and gpt2-merges.txt with [download_vocab.sh](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/dataset/download_vocab.sh) and place the files in ```/data/webtext/```.

- WebText

Follow the instructions in [Collecting GPT Webtext Data](https://github.com/microsoft/Megatron-DeepSpeed#data-preprocessing) and [Data Preprocessing](https://github.com/microsoft/Megatron-DeepSpeed#data-preprocessing) to download and preprocess webtext dataset. I had to change ```USE_CYTHON = True``` in ```setup.py``` to install LSH. After merging the contents of webtext data into a single json file, run ```tools/preprocess_data.py```. Place ```webtext_text_document.bin``` and ```webtext_text_document.idx``` files in ```/data/webtext```. 

Megatron model trains with WebText dataset and other models train with synthetic dataset.

## 6. Run benchmarks
Root permission is required to change the frequency of the GPU with the [NVML API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html#group__nvmlDeviceCommands_1gc9b58cd685f4deee575400e2e6ac76cb). EnvPipe runs with sudo while keeping the PATH environment variable. Run the script with ```sudo -E env PATH=$PATH script.sh```.

There are three parameters for energy saving optimization settings in EnvPipe.

| **Parameter** | **Inputs** | **Explanation** |
|---|---|---|
| ENVPIPE_TYPE | baseline | Run all GPUs with maximum SM frequency. |
|  | uniform | Run all GPUs with optimal SM frequency that represents the minimum point in the energy valley curve. |
|  | envelope | Run pipeline units with optimal SM frequency that are inside the outer envelope. |
| ENVPIPE_SCHEDULING | 1f1b | 1F1B scheduling method. |
|  | ours | EnvPipe's scehduling method. |
| ENVPIPE_RECONFIGURE | default | SM frequencies of pipeline units on the critical path are not reconfigured. |
|  | greedy |  SM frequencies of pipeline units on the critical path are greedily reconfigured from the end of the critical path. |
|  | balanced | SM frequencies of pipeline units on the critical path are balanced as much as possible. |


EnvPipe's reconfiguring phase generates an output at each step that represents the current energy-saving plan that specifies a schedule of the forward and backward pipeline units and SM frequency value of each pipeline unit. Each pipeline unit is shown as a combination of its type (forward or backward), SM frequency, and time between pipeline units. Units on the critical path are wrapped in [ ]. The reconfiguring phase ends when the critical path matches the outer envelope of the pipeline.

```bash
CURRENT_STEP
Stage 0 | [F0 1980 0.0]   F1 1260 0.0   F2 1260 0.0   F3 1260 0.0   B0 1260 185.8 B1 1260 62.7  B2 1260 61.8 [B3 1980 40.3]
Stage 1 | [F0 1980 23.4]  F1 1260 8.4   F2 1260 2.1   F3 1260 1.3   B0 1260 109.1 B1 1260 63.0  B2 1260 62.2 [B3 1980 50.9]
Stage 2 | [F0 1980 41.4]  F1 1260 12.8  F2 1260 2.1   F3 1260 1.4   B0 1260 32.8  B1 1260 62.7  B2 1260 62.3 [B3 1980 61.1]
Stage 3 | [F0 1980 59.5] [B0 1980 0.0] [F1 1980 0.6] [B1 1980 0.0] [F2 1980 0.6] [B2 1980 0.0] [F3 1980 0.6] [B3 1980 0.0]
```

EnvPipe currently supports V100 and RTX3090 GPUs. If you plan to use a different GPU type, you can add its configuration to `EnvPipe/DeepSpeed/deepspeed/runtime/constants.py`.

### 6.1. Single-node

#### Figure 9: Throughput and energy consumption of various DNN models in single-node training

```shell
$ cd EnvPipe/benchmark/single_node/script
$ sudo -E env PATH=$PATH ./fig9_a.sh # Single-V100
$ sudo -E env PATH=$PATH ./fig9_b.sh # Single-3090
$ cat result/fig9_a_[TIMESTAMP].csv
$ cat result/fig9_b_[TIMESTAMP].csv
```

#### Figure 10: Normalized throughput and energy consumption breakdown

```shell
$ cd EnvPipe/benchmark/single_node/script
$ sudo -E env PATH=$PATH ./fig10.sh # Single-3090
$ cat result/fig10_[TIMESTAMP].csv
```

#### Figure 11: Comparison of reconfiguration policy

```shell
$ cd EnvPipe/benchmark/single_node/script
$ sudo -E env PATH=$PATH ./fig11.sh # Single-V100
$ cat result/fig11_[TIMESTAMP].csv
```

#### Figure 13: Sensitivity study of number of microbatches

```shell
$ cd EnvPipe/benchmark/single_node/script
$ sudo -E env PATH=$PATH ./fig13.sh # Single-V100
$ cat result/fig13_[TIMESTAMP].csv
```

### 6.2. Multi-node

To configure multi-node training in EnvPipe, begin by preparing each node for single-node training. Next, modify the distributed option setting in your script to reflect your desired configuration. For example, the following code snippet demonstrates how to configure training across two nodes, each with four GPUs. Change the `NODE_RANK` value to specify the rank of each node. Set the `MASTER_ADDR` and `MASTER_PORT` variables to the IP address and port number of the machine acting as the master node.

```
NNODES=2
NPROC_PER_NODE=4
NODE_RANK=0 # Change this value to the rank of each node (MASTER: 0)
MASTER_ADDR="172.31.47.231"
MASTER_PORT="6000"
```

#### Figure 12: Throughput and energy consumption of Megatron-1.3B in multi-node training

```shell
$ cd EnvPipe/benchmark/multi_node/script
$ sudo -E env PATH=$PATH ./fig12.sh # Multi-V100

# Add the energy consumption of all the nodes to calculate the total energy consumption.
$ cat result/fig12_[TIMESTAMP].csv 
```
