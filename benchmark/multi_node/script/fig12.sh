#!/bin/bash
TIMESTAMP=$(date +%s)
OUTPUT_FILE="fig12_"${TIMESTAMP}

run() {
  MODEL=$1
  MICROBATCH_SIZE=$2
  MINIBATCH_SIZE=$3
  TYPE=$4
  SCHEDULING=$5
  RECONFIG=$6
  GPU=$7
  DP_SIZE=$8
  PP_SIZE=$9

  ./run_${MODEL}.sh ${MICROBATCH_SIZE} ${MINIBATCH_SIZE} ${TYPE} ${SCHEDULING} ${RECONFIG} ${GPU} ${DP_SIZE} ${PP_SIZE} &> tmp.out
  readarray -t output < <(grep "RESULT" tmp.out)
  energy=$(echo $output | cut -d ' ' -f 4)
  throughput=$(echo $output | cut -d ' ' -f 2)
  echo ${DP_SIZE}, ${PP_SIZE}, ${MODEL}, ${MICROBATCH_SIZE}, ${MINIBATCH_SIZE}, ${TYPE}, ${SCHEDULING}, ${RECONFIG}, ${throughput}, ${energy} >> result/${OUTPUT_FILE}.csv
}

mkdir -p result
echo "DP, PP, Model, Microbatch size, Minibatch size, Type, Scheduling, Reconfig, Throughput (sample/s), Energy (mJ)" >> result/${OUTPUT_FILE}.csv

# figure 12
# Megatron-1.3B
run "megatron_1.3B" 2 512 "baseline" "1f1b" "default" "v100" 2 8 
run "megatron_1.3B" 2 512 "envelope" "ours" "balanced" "v100" 2 8 

run "megatron_1.3B" 2 512 "baseline" "1f1b" "default" "v100" 4 4
run "megatron_1.3B" 2 512 "envelope" "ours" "balanced" "v100" 4 4


# GPT_350M with 8 GPUs DP2+PP4 (2 Nodes X 4 GPUs)
# run "gpt_350M" 2 32 "baseline" "1f1b" "default" "v100" 2 4 
# run "gpt_350M" 2 32 "envelope" "ours" "balanced" "v100" 2 4 
# run "gpt_350M" 2 32 "baseline" "1f1b" "default" "v100" 1 8 
# run "gpt_350M" 2 32 "envelope" "ours" "balanced" "v100" 1 8 