#!/bin/bash
# Replace file_path with the actual path to your file
cd $1

# Extract the values using grep and awk, filter out non-numeric values
values=$(grep -r "Enzyme/accuracy:test " --exclude-dir='*'  | awk  '{print $3}' | tr '\n' ' ')

# Calculate the mean and standard deviation using Python
python -u <<EOF
import numpy as np
values = np.array('$values'.split())
print(values)
mean = np.mean(values.astype(float))
std_dev = np.std(values.astype(float))

print("Mean:", str(mean * 100)[:5])
print("Standard deviation:", str(std_dev * 100)[:4])


print("")
EOF

