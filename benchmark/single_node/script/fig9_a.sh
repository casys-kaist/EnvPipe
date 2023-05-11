#!/bin/bash
TIMESTAMP=$(date +%s)
OUTPUT_FILE="fig9_a_"${TIMESTAMP}

run() {
  MODEL=$1
  MICROBATCH_SIZE=$2
  MINIBATCH_SIZE=$3
  TYPE=$4
  SCHEDULING=$5
  RECONFIG=$6
  GPU=$7

  ./run_${MODEL}.sh ${MICROBATCH_SIZE} ${MINIBATCH_SIZE} ${TYPE} ${SCHEDULING} ${RECONFIG} ${GPU} &> tmp.out
  readarray -t output < <(grep "RESULT" tmp.out)
  energy=$(echo $output | cut -d ' ' -f 4)
  throughput=$(echo $output | cut -d ' ' -f 2)
  echo ${MODEL}, ${MICROBATCH_SIZE}, ${MINIBATCH_SIZE}, ${TYPE}, ${SCHEDULING}, ${RECONFIG}, ${throughput}, ${energy} >> result/${OUTPUT_FILE}.csv
}

mkdir -p result
echo "Model, Microbatch size, Minibatch size, Type, Scheduling, Reconfig, Throughput (sample/s), Energy (mJ)" >> result/${OUTPUT_FILE}.csv

# figure 9(a) 
# BERT-336M
run "bert_336M" 4 64 "baseline" "1f1b" "default" "v100"
run "bert_336M" 4 64 "envelope" "ours" "balanced" "v100"

# GPT-125M
run "gpt_125M" 2 32 "baseline" "1f1b" "default" "v100"
run "gpt_125M" 2 32 "envelope" "ours" "balanced" "v100"

# Megatron-125M
run "megatron_125M" 4 64 "baseline" "1f1b" "default" "v100"
run "megatron_125M" 4 64 "envelope" "ours" "balanced" "v100"

# ResNet-152
run "resnet_152" 2 32 "baseline" "1f1b" "default" "v100"
run "resnet_152" 2 32 "envelope" "ours" "balanced" "v100"
