#!/bin/sh

usage() {
  echo "Usage: $0 --models MODEL [MODEL ...]" >&2
  exit 1
}

[ "${1:-}" = "--models" ] || usage
shift
[ $# -gt 0 ] || usage

for model in "$@"; do
  case "$model" in
    -*) usage ;;
  esac
done

for model in "$@"; do
  for calibrate in constant linear_time ""; do
    if [ -n "$calibrate" ]; then
      calibrate_line="calibrate: ${calibrate}
"
      calibration_name="calibrated_${calibrate}"
    else
      calibrate_line=""
      calibration_name="uncalibrated"
    fi

    for i in $(seq 1 9); do
      alpha="0.$i"
      name="${model}_${calibration_name}_alpha_0_${i}"
      if [ "$model" = "iknn" ]; then
        k_neighbors=5
      else
        k_neighbors=20
      fi
      cat > "${name}.yaml" <<EOF
size: s
data_type: yelp

rounds: 2500
num_rounds_per_eval: 250
num_users: 1000

model: ${model}
preference_update_rate: ${alpha}

${calibrate_line}exp_name: ${name}
params: { "k_neighbors": ${k_neighbors} }
EOF
    done
  done
done