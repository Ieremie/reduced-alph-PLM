#!/bin/bash

aav_splits=("low_vs_high" "mut_des" "one_vs_many" "seven_vs_many" "two_vs_many")
gb1=("low_vs_high" "one_vs_rest" "three_vs_rest" "two_vs_rest")
meltome=("mixed_split")

model_path=$1
alphabet_type=$2
random_seed=$3

data_path="/scratch/ii1g17/protein-embeddings/data/FLIP"
script_path_rtx="/home/ii1g17/protein-embeddings/proemb/iridis-scripts/flip/rtx8000.sh"
script_path_v100="/home/ii1g17/protein-embeddings/proemb/iridis-scripts/flip/v100.sh"

if [ "$4" -eq 1 ]; then
  random_validation="--random_validation"
else
  random_validation=""
fi

val_split=$5

run_all() {

  seed=$1
  # dataset = aav
  for split in "${aav_splits[@]}"
  do
    echo "Split: $split"
    # if des_mut we only do for one seed
    if [ "$split" = "mut_des" ] && [ "$seed" != "1" ]; then
      echo "Single run for mut_des"
    else
      sbatch  "$script_path_rtx" --remote --data_path=$data_path --dataset="aav" --split=$split \
       --model_path=$model_path --alphabet_type=$alphabet_type --seed=$seed --lr=0.001 --encoder_lr=0.0002 \
       --patience=25 --scheduler_patience=15 --dropout_head=0.25 --val_split=$val_split $random_validation
    fi
    echo ""
  done

  # dataset = gb1
  for split in "${gb1[@]}"
  do
    echo "Split: $split"
    sbatch "$script_path_v100" --remote --data_path=$data_path --dataset="gb1" --split=$split \
     --model_path=$model_path --alphabet_type=$alphabet_type --seed=$seed --lr=0.001 --encoder_lr=0.0002 \
     --patience=40 --scheduler_patience=15 --dropout_head=0.25 --val_split=$val_split $random_validation
    echo ""
  done

  # dataset = meltome
  for split in "${meltome[@]}"
  do
    echo "Split: $split"
    # if des_mut we only do for one seed
    if [ "$split" = "mixed_split" ] && [ "$seed" != "1" ]; then
      echo "Single run for mixed_split"
    else
      sbatch "$script_path_rtx" --remote --data_path=$data_path --dataset="meltome" --split=$split \
      --model_path=$model_path --alphabet_type=$alphabet_type --seed=$seed --lr=0.0004 --encoder_lr=0.0002 \
      --patience=20 --scheduler_patience=10 --batch_size=16 --dropout_head=0.25 --val_split=$val_split $random_validation
    fi
    echo ""
  done
}

if [ $random_seed -eq 1 ]; then
  echo "Single run"
  run_all 1
else
  # 3 different seeds
  for seed in {1..3}
  do
    echo "Run $seed"
    run_all $seed
  done
fi