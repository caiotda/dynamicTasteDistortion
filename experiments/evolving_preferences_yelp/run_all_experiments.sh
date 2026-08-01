#!/bin/bash

echo "Running experiments"

for exp in calibrated uncalibrated; do
  echo "Running $exp experiments"

  for i in $(seq 1 9); do
    echo "Running experiment ${exp}_alpha_0_$i"
    simulate-recommendation --exp_file experiments/evolving_preferences_yelp/bpr_unbiased_${exp}_alpha_0_${i}.yaml \
      || echo "${exp}_alpha_0_$i FAILED" >> failed_experiments.log
  done
done