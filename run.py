import glob
import argparse
import os
import time
import numpy as np
import datetime
import re

MODEL_CONFIG = {
    'llama-7b': {
        'path': 'models/llama-7b',
        'data': 'data/alpaca_data_cleaned.json',
        'lr':   '2e-5',
        'epoch': 3
    },
    'mistral-7b': {
        'path': 'models/Mistral-7B-v0.1',
        'data': 'data/alpaca_data_cleaned.json',
        'lr':   '2.5e-6',
        'epoch': 3
    }
}

def get_train_cmd(model, attack):
    master_port = 29550 + np.random.randint(0, 1000)
    current_t = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = f'models/{model}_{attack}_{current_t}'
    path = MODEL_CONFIG[model]['path']
    lr = MODEL_CONFIG[model]['lr']
    data = MODEL_CONFIG[model]['data']
    epoch = MODEL_CONFIG[model]['epoch']

    if model == 'llama-7b':
        return f'torchrun --nproc_per_node=4 --master_port={master_port} train.py \
            --model_name_or_path {path} \
            --data_path {data} \
            --bf16 True \
            --output_dir {output_dir} \
            --num_train_epochs {epoch} \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_total_limit 1 \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
            --tf32 True\
            --attack {attack}'
    elif model == 'mistral-7b':
        return f'torchrun --nproc_per_node=4 --master_port={master_port} train.py \
            --model_name_or_path {path} \
            --window_size 256 \
            --padding_side left \
            --data_path {data} \
            --bf16 True \
            --output_dir {output_dir} \
            --num_train_epochs {epoch} \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_total_limit 1 \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer" \
            --tf32 True\
            --attack {attack}\
            --lr_scale True\
            --downsample True'
    else: raise NotImplementedError

def train_and_test():
    parser = argparse.ArgumentParser(prog='Training model(s) accepting structured queries on 4 80GB A100s', description='The script implements the slurm pipeline for training multiple defended models and later testing them with multiple attacks.')
    parser.add_argument('-m', '--model', type=str, default='llama-7b', choices=MODEL_CONFIG.keys())
    parser.add_argument('-train', '--train_attack', type=str, default=['SpclSpclSpcl_NaiveCompletion'], nargs='+')
    parser.add_argument('-test', '--test_attack', type=str, default=['none', 'naive', 'ignore', 'escape_deletion', 'escape_separation', 'completion_other', 'completion_othercmb', 'completion_real', 'completion_realcmb', 'completion_close_2hash', 'completion_close_1hash', 'completion_close_0hash', 'completion_close_upper', 'completion_close_title', 'completion_close_nospace', 'completion_close_nocolon', 'completion_close_typo', 'completion_close_similar', 'hackaprompt'], nargs='+') # Use test_tap to test TAP attack
    parser.add_argument('-t', '--time', type=float, default=4)
    parser.add_argument('-e', '--env', type=str, default='struq')
    parser.add_argument('--do_test', type=bool, default=True)
    args = parser.parse_args()
    
    output_dirs = []
    for attack in args.train_attack:
        cmd = get_train_cmd(args.model, attack)
        output_dir = re.search(f'--output_dir (.+?)--num_train_epochs', cmd).group(1).replace(' ', '')
        log_dir = output_dir.replace('models', 'logs')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=0{args.time}:00:00\n#SBATCH --gres=gpu:4\n#SBATCH --cpus-per-task=16\n#SBATCH --output={log_dir}/train_%j.out\n\nsource activate {args.env}\n"

        temporary_slurm_file = f'train_{args.model}_{attack}.slurm'
        with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
        os.system('sbatch ' + temporary_slurm_file)
        os.remove(temporary_slurm_file)
        print('\n' * 10 + slurm_prefix + cmd + '\n' * 10)
        output_dirs.append(output_dir)
        time.sleep(2)
    
    if not args.do_test: return
    print("Submitted all", len(output_dirs), "jobs, waiting for completion...")
    completed = []

    while len(completed) < len(output_dirs):
        for output_dir in [x for x in output_dirs if x not in completed]:
            if len(glob.glob(f'{output_dir}/*.json')) < 8: continue
            time.sleep(30)
            print(f"Scheduling tests for {output_dir}, {1+len(completed)}/{len(output_dirs)}.")
            
            log_dir = output_dir.replace('models', 'logs')
            os.makedirs(log_dir, exist_ok=True)

            for attack in args.test_attack:
                slurm_prefix = f"#!/bin/bash\n\n#SBATCH --nodes=1\n#SBATCH --time=0{args.time}:00:00\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=16\n#SBATCH --output={log_dir}/{attack}_%j.out\n\nsource activate {args.env}\n"
                cmd = f'python test.py --model_name_or_path {output_dir} --attack {attack}' # you may add --defense to test zero-shot prompting defense baselines
                temporary_slurm_file = 'test_' + args.model + output_dir.replace('/', '_') + '.slurm'
                with open(temporary_slurm_file, 'w') as f: f.write(slurm_prefix + cmd)
                os.system('sbatch ' + temporary_slurm_file)
                os.remove(temporary_slurm_file)
                print('\n' * 10 + slurm_prefix + cmd + '\n' * 10)
                time.sleep(2)
            completed.append(output_dir)
        time.sleep(10)

if __name__ == '__main__':
    train_and_test()