#!/bin/bash

model=""
for arg in "$@"; do
  case $arg in
    model=*) model="${arg#model=}" ;;
  esac
done

[ -z "$model" ] && { echo "Usage: $0 model=NAME"; exit 1; }

echo "Running experiments"

for exp in calibrated uncalibrated; do
  echo "Running $exp experiments"

  for i in $(seq 1 9); do
    echo "Running experiment ${model}_${exp}_alpha_0_$i"
    simulate-recommendation --exp_file experiments/evolving_preferences_movielens/${model}_${exp}_alpha_0_${i}.yaml \
      || echo "${model}_${exp}_alpha_0_$i FAILED" >> failed_experiments.log
  done
done