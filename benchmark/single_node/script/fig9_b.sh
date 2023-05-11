#!/bin/bash
TIMESTAMP=$(date +%s)
OUTPUT_FILE="fig9_b_"${TIMESTAMP}

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

# figure 9(b)
# BERT-1.3B
run "bert_1.3B" 4 64 "baseline" "1f1b" "default" "rtx3090"
run "bert_1.3B" 4 64 "envelope" "ours" "balanced" "rtx3090"

# BERT-3.9B
run "bert_3.9B" 2 32 "baseline" "1f1b" "default" "rtx3090"
run "bert_3.9B" 2 32 "envelope" "ours" "balanced" "rtx3090"

# GPT-350M
run "gpt_350M" 4 64 "baseline" "1f1b" "default" "rtx3090"
run "gpt_350M" 4 64 "envelope" "ours" "balanced" "rtx3090"

# # Megatron-350M 
run "megatron_350M" 4 64 "baseline" "1f1b" "default" "rtx3090"
run "megatron_350M" 4 64 "envelope" "ours" "balanced" "rtx3090"

# Megatron-760M 
run "megatron_760M" 4 64 "baseline" "1f1b" "default" "rtx3090"
run "megatron_760M" 4 64 "envelope" "ours" "balanced" "rtx3090"
