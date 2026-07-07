#!/bin/bash

for i in $(seq 2 9); do
  echo "Running experiment alpha_0_$i"
  simulate-recommendation --exp_file experiments/evolving_preferences_globo/bpr_unbiased_uncalibrated_alpha_0_${i}.yaml \
    || echo "alpha_0_$i FAILED" >> failed_experiments.log
done