#!/bin/bash

model=""
for arg in "$@"; do
  case $arg in
    model=*) model="${arg#model=}" ;;
  esac
done

[ -z "$model" ] && { echo "Usage: $0 model=NAME"; exit 1; }

exp_dir="experiments/evolving_preferences_yelp"

shopt -s nullglob
found=("$exp_dir/${model}_"*.yaml)
shopt -u nullglob

if [ ${#found[@]} -eq 0 ]; then
  echo "No experiments found for model '$model' in $exp_dir"
  exit 1
fi

echo "Found ${#found[@]} experiments:"
for file in "${found[@]}"; do
  echo "  $(basename "$file" .yaml)"
done

echo "Running experiments"

for file in "${found[@]}"; do
  name="$(basename "$file" .yaml)"
  echo "Running experiment $name"
  simulate-recommendation --exp_file "$file" \
    || echo "$name FAILED" >> failed_experiments.log
done