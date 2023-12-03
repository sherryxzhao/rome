#!/bin/bash

cd eval_sbatch_tasks

# Loop through each .sbatch file and submit it
for file in eval_*.sbatch; do
    echo "Submitting job for $file..."
    sbatch $file
done

echo "All jobs have been submitted."
