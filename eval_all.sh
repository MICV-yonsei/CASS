run_task() {
  dataset_name=$1
  gpu=$2
  config_file=$3

  work_dir="./experiment/${dataset_name}"
  CUDA_VISIBLE_DEVICES=$gpu python eval.py --config $config_file --work-dir $work_dir --pamr off
  sleep 2
}

gpu_list=(0)

configs_list=(
   
    './configs/cfg_voc20.py'
    './configs/cfg_voc21.py'  
    './configs/cfg_ade20k.py'
    './configs/cfg_city_scapes.py'
    './configs/cfg_coco_stuff164k.py'
    './configs/cfg_context59.py'
    './configs/cfg_context60.py'
    './configs/cfg_coco_object.py'
)

run_in_batches() {
  config_list=("$@")
  num_configs=${#config_list[@]}
  num_gpus=${#gpu_list[@]}

  for ((i=0; i<$num_configs; i+=num_gpus)); do
    for ((j=0; j<num_gpus; j++)); do
      if [ $((i + j)) -lt $num_configs ]; then
        config_file=${config_list[$((i + j))]}
        dataset_name=$(basename "$config_file" .py)
        gpu=${gpu_list[$j]}

        run_task $dataset_name $gpu $config_file &
      fi
    done
    wait
  done
}

run_in_batches "${configs_list[@]}"
