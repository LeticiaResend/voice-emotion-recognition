#!/bin/bash
for f in 0 1 2 3 4; do
  if [ ! -f "../data/processed/scratch_fold${f}_result.pt" ]; then
    echo "=== Model 1, fold $f ==="
    python3 train_scratch.py --fold $f
  else
    echo "Model 1 fold $f already done, skipping"
  fi
done
for f in 0 1 2 3 4; do
  if [ ! -f "../data/processed/transfer_fold${f}_result.pt" ]; then
    echo "=== Model 2, fold $f ==="
    python3 train_transfer.py --fold $f
  else
    echo "Model 2 fold $f already done, skipping"
  fi
done
