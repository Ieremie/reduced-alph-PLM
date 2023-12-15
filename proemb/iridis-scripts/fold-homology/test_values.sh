#!/bin/bash

module load conda
source activate prose


# Replace file_path with the actual path to your file
cd $1
for col in 0 1 2; do
  # Extract the values using grep and awk, filter out non-numeric values
  values=$(grep -r "accuracy/dataloader_idx_$col': {'test': " --exclude-dir='*' | awk -F'[(,]' '{print $2}' | tr '\n' ' ')

  # if values is empty we try another grep
  if [ -z "$values" ]
  then
    values=$(grep -r "FOLD/accuracy:test" --exclude-dir='*'  | awk -v select="$((col+3))" '{print $select}' | tr '\n' ' ')
  fi

# Calculate the mean and standard deviation using Python
python -u <<EOF
import numpy as np
values = np.array('$values'.split())
print(values)
mean = np.mean(values.astype(float))
std_dev = np.std(values.astype(float))

print("Column:", $col)
print("Mean:", str(mean * 100)[:5])
print("Standard deviation:", str(std_dev * 100)[:4])


print("")
EOF
done

values=$(grep -r "Enzyme/accuracy': {'test': " --exclude-dir='*'  |  awk -F'[(,]' '{print $2}' | tr '\n' ' ')
