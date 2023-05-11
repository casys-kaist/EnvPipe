#!/bin/bash
export NCCL_BUFFSIZE=167772160
export NCCL_IB_DISABLE=1

if [ "$EUID" -ne 0 ]
  then echo -e "\nError: EnvPipe should run with root permission to change the frequency of the GPU. 
  Run the script with sudo -E env PATH=\$PATH ./script.sh\n"
  exit 1
fi

if [ $# -ne 6 ]
  then
    echo -e "\nError: Invalid parameters provided.
    Please ensure that you have supplied the correct parameters in the following order:
    ./run_model.sh <MICROBATCH_SIZE> <MINIBATCH_SIZE> <ENVPIPE_TYPE> <ENVPIPE_SCHEDULING> <ENVPIPE_RECONFIGURE> <ENVPIPE_GPU> \n"
    exit 1
fi

### Configurations for EnvPipe
### ENVPIPE_TYPE
###  - baseline
###  - uniform
###  - envelope
### ENVPIPE_SCHEDULING
###  - 1f1b
###  - ours
### ENVPIPE_RECONFIGURE
###  - default
###  - greedy
###  - balanced
### ENVPIPE_GPU
###  - v100
###  - rtx3090

BATCH_SIZE=$1
GLOBAL_BATCH_SIZE=$2
ENVPIPE_TYPE=$3
ENVPIPE_SCHEDULING=$4
ENVPIPE_RECONFIGURE=$5
ENVPIPE_GPU=$6

### Main configs
MODEL="resnet"
DP_SIZE=1
PP_SIZE=4
NUM_GPUS=4
STEPS=30
PARTITIONS="8,12,15,17"

# If performance degrades, increase this value
RECONFIGURE_THRESHOLD_SCALE=8

train_options=" \
    --steps ${STEPS} \
    --backend nccl \
    --dp ${DP_SIZE} \
    --pp ${PP_SIZE} \
    --parts ${PARTITIONS}"

template_json="config/TEMPLATE.json"
config_json="config/config_${MODEL}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/ENVPIPE_TYPE/${ENVPIPE_TYPE}/" \
    | sed "s/ENVPIPE_SCHEDULING/${ENVPIPE_SCHEDULING}/" \
    | sed "s/ENVPIPE_RECONFIGURE/${ENVPIPE_RECONFIGURE}/" \
    | sed "s/ENVPIPE_RECONFIG_THRESHOLD_SCALE/${RECONFIGURE_THRESHOLD_SCALE}/" \
    | sed "s/ENVPIPE_GPU/${ENVPIPE_GPU}/" \
	  > ${config_json}

deepspeed ../model/${MODEL}.py ${train_options} --deepspeed_config ${config_json}
