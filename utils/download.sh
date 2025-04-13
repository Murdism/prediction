#!/bin/bash

# Number of scenarios you want
N=15000

# Output folder
DEST_DIR="./waymo_scenarios"
mkdir -p "$DEST_DIR"

# Waymo Motion Dataset path
GS_PATH="gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training"

# Start downloading
for i in $(seq -f "%05g" 0 $((N - 1)))
do
  # Construct the shard filename pattern
  SHARD_FILE="training.tfrecord-$(printf "%05d" $((i % 1000)))-of-01000"
  echo "Downloading $SHARD_FILE..."
  gsutil cp "${GS_PATH}/${SHARD_FILE}" "$DEST_DIR/" || echo "Failed to download $SHARD_FILE"
done
