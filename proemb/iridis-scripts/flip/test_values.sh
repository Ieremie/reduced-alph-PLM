#!/bin/bash

module load conda
source activate prose

cd $1

dir_to_exclude=$2

aav_splits=("one_vs_many" "two_vs_many" "seven_vs_many" "low_vs_high" "mut_des")
gb1=("one_vs_rest" "two_vs_rest" "three_vs_rest" "low_vs_high")
meltome=("mixed_split")

run_numpy() {
python -u <<EOF
import numpy as np
values = np.array('$values'.split())
values = np.array([val for val in values if val != 'nan'])

print("Values with no Nans: ", values)
mean = np.mean(values.astype(float))
std_dev = np.std(values.astype(float))

print("Mean:", str(mean)[:4])
print("Standard deviation:", str(std_dev )[:4])

print("")
EOF
}

for split in "${aav_splits[@]}"
do
  echo "Dateset aav, Split: $split"
  a=$(grep -rH corr:test $(grep -rl aav $(grep -rl \'$split\' --exclude-dir=$dir_to_exclude)) | awk '{print $3}' | tr '\n' ' ')
  b=$(grep -rH "corr': {'test'" $(grep -rl aav $(grep -rl \'$split\' --exclude-dir=$dir_to_exclude))  | awk -F'[(,]' '{print $2}' | tr '\n' ' ')
  values="$a $b"
  run_numpy
done

for split in "${gb1[@]}"
do
  echo "Dateset gb1, Split: $split"
  a=$(grep -rH corr:test $(grep -rl gb1 $(grep -rl \'$split\' --exclude-dir=$dir_to_exclude)) | awk '{print $3}' | tr '\n' ' ')
  b=$(grep -rH "corr': {'test'" $(grep -rl gb1 $(grep -rl \'$split\' --exclude-dir=$dir_to_exclude))  | awk -F'[(,]' '{print $2}' | tr '\n' ' ')
  values="$a $b"
  run_numpy
done

for split in "${meltome[@]}"
do
  echo "Dateset meltome, Split: $split"
  a=$(grep -rH corr:test $(grep -rl meltome $(grep -rl \'$split\' --exclude-dir=$dir_to_exclude)) | awk '{print $3}' | tr '\n' ' ')
  b=$(grep -rH "corr': {'test'" $(grep -rl meltome $(grep -rl \'$split\' --exclude-dir=$dir_to_exclude))  | awk -F'[(,]' '{print $2}' | tr '\n' ' ')
  values="$a $b"
  run_numpy
done





