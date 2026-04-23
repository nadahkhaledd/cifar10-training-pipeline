#!/bin/bash
# Run two training jobs concurrently with specified GPU percentages
# Usage: bash run_concurrent.sh <model1> <pct1> <model2> <pct2> <tag>
# Example: bash run_concurrent.sh resnet18 50 mobilenet_v2 50 "50-50"

MODEL1=$1
PCT1=$2
MODEL2=$3
PCT2=$4
TAG=$5

source ~/pytorch-env/bin/activate
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_khab/pipe
export CUDA_VISIBLE_DEVICES=0

echo "Starting $MODEL1 at ${PCT1}% and $MODEL2 at ${PCT2}% concurrently..."
echo "Tag: $TAG"
echo "Start time: $(date)"

# Launch job 1 in background
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT1 \
  python3 train.py --model $MODEL1 --epochs 5 \
  --experiment_tag "concurrent-${TAG}-${MODEL1}" \
  --sm_allocation "${PCT1}pct-concurrent" \
  > outputs/log_${MODEL1}_${TAG}.txt 2>&1 &
PID1=$!

# Launch job 2 in background
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT2 \
  python3 train.py --model $MODEL2 --epochs 5 \
  --experiment_tag "concurrent-${TAG}-${MODEL2}" \
  --sm_allocation "${PCT2}pct-concurrent" \
  > outputs/log_${MODEL2}_${TAG}.txt 2>&1 &
PID2=$!

echo "Job 1 ($MODEL1 @ ${PCT1}%): PID $PID1"
echo "Job 2 ($MODEL2 @ ${PCT2}%): PID $PID2"
echo "Waiting for both to finish..."

wait $PID1
echo "$MODEL1 finished (exit code: $?)"
wait $PID2
echo "$MODEL2 finished (exit code: $?)"

echo "End time: $(date)"
echo ""
echo "=== Results ==="
tail -15 outputs/log_${MODEL1}_${TAG}.txt
echo "---"
tail -15 outputs/log_${MODEL2}_${TAG}.txt
