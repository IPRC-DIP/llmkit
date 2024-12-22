set -xe

APPS=/lustre/S/nanziyuan/datasets/apps
result=/lustre/S/nanziyuan/projects/llmkit/trl_test3
base_model=/lustre/S/huangdi/open_for_out/models/deepseek-coder-6.7b-instruct

flag=${result}/finished.flag

config=${result}/deepspeed_zero3.yaml
script=${result}/sft.py

samples=${result}/eval/sample.jsonl
evals=${result}/eval/evals.jsonl

## Dataset Preprocessing
if [[ $SLURM_PROCID -eq 0 ]]; then
    # Main process logic
    echo "Main process, preprocessing dataset."
    python -m llmkit_data.cli.prep_apps \
        --apps ${APPS}/train.jsonl \
        --out ${result}/dataset/train.jsonl \
        --type SFT

    python -m llmkit_data.cli.prep_apps \
        --apps ${APPS}/test.jsonl \
        --out ${result}/dataset/test.jsonl \
        --type SFT \
        --prompt_only

    python -m llmkit_data.cli.convert_to_trl \
        --dataset ${result}/dataset/train.jsonl \
        --out ${result}/trl_dataset/train.jsonl

    # Signal other processes to continue
    touch ${flag}
else
    # Other processes wait for the main process to finish
    while [[ ! -f ${flag} ]]; do
        sleep 1
    done
fi

## Training
LAUNCHER="\
accelerate launch \
  --config_file ${config} \
  --num_processes $((SLURM_NNODES * USER_GPUS_PER_NODE)) \
  --num_machines $SLURM_NNODES \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID\
"

# LAUNCHER="\
# accelerate launch \
#   --config_file ${config} \
#   --num_processes 8 \
# "

PROGRAM="\
  ${script} \
  --model_name_or_path ${base_model} \
  --dataset_name ${result}/trl_dataset \
  --bf16 \
  --seed 42 \
  --packing \
  --max_seq_length 2048 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --learning_rate 2.0e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 25 \
  --torch_dtype bfloat16 \
  --save_strategy no \
  --output_dir ${result}/model \
  --report_to tensorboard \
  --logging_steps=1 \
"

bash -c "$LAUNCHER $PROGRAM"

## Evaluation
if [[ $SLURM_PROCID -eq 0 ]]; then
    # Main process logic
    echo "Main process, evaluating result."

    python -m llmkit_data.cli.sample \
        --prompts ${result}/dataset/test.jsonl \
        --out ${samples} \
        --model ${result}/model \
        --gpu_per_model 1

    python -m llmkit_data.cli.eval_apps \
        --samples ${samples} \
        --out ${evals} \
        --apps ${APPS}

    rm ${flag}
fi