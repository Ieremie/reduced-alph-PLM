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


def calc_mean_std(values):
    values = np.array(values.split())
    values = np.array([val for val in values if val != 'nan'])

    print("Values with no Nans: ", values)
    mean = np.mean(values.astype(float))
    std_dev = np.std(values.astype(float))

    return mean, std_dev

mean_no_random_val, std_no_random_val = calc_mean_std('$values_no_random_val')
mean_random_val, std_random_val = calc_mean_std('$values_random_val')

# choose the best mean and print it
if mean_no_random_val > mean_random_val:
    mean = mean_no_random_val
    std_dev = std_no_random_val
    print("Choosing no random val")
else:
    mean = mean_random_val
    std_dev = std_random_val
    print("Choosing random val")

print("Mean:", str(mean)[:4])
print("Standard deviation:", str(std_dev )[:4])

print("")
EOF
}

grep_values(){
  a=$(grep -rH corr:test $(grep -rl $1 $(grep -rl \'$2\' --exclude-dir=$dir_to_exclude)) | awk '{print $3}' | tr '\n' ' ')
  b=$(grep -rH "corr': {'test'" $(grep -rl $1 $(grep -rl \'$2\' --exclude-dir=$dir_to_exclude))  | awk -F'[(,]' '{print $2}' | tr '\n' ' ')
  values_no_random_val="$a $b"

  #now we grep from inside the dir_to_exclude (random val folder)
  cd $dir_to_exclude
  a=$(grep -rH corr:test $(grep -rl $1 $(grep -rl \'$2\')) | awk '{print $3}' | tr '\n' ' ')
  b=$(grep -rH "corr': {'test'" $(grep -rl $1 $(grep -rl \'$2\'))  | awk -F'[(,]' '{print $2}' | tr '\n' ' ')
  values_random_val="$a $b"
  cd ..
}

for split in "${aav_splits[@]}"
do
  echo "Dateset aav, Split: $split"
  grep_values aav $split
  run_numpy
done

for split in "${gb1[@]}"
do
  echo "Dateset gb1, Split: $split"
  grep_values gb1 $split
  run_numpy
done

for split in "${meltome[@]}"
do
  echo "Dateset meltome, Split: $split"
  grep_values meltome $split
  run_numpy
done





