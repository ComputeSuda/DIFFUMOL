"""
Train a diffusion model on images.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import json

import numpy as np
from diffumol.utils import dist_util, logger
from diffumol.text_datasets import load_data_text
from diffumol.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb
import pandas as pd
import re
### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args) 
    model_weight, tokenizer= load_model_emb(args, tokenizer) 
    #——————————————————————————————————————————————————————————————————————————————
    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True) 
    data.columns = data.columns.str.lower()

    if 'moses' in args.data_name:
        train_data = data[data['split'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses
    else:
        train_data = data[data['source'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses


    if 'moses' in args.data_name:
        val_data = data[data['split'] == 'test'].reset_index(
            drop=True)   # test for Moses. val for guacamol
    else:
        val_data = data[data['source'] == 'val'].reset_index(
            drop=True)   # test for Moses. val for guacamol


    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
              for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    lens = [len(regex.findall(i.strip()))
            for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)
    print('Scaffold max len: ', scaffold_max_len)
    
    if args.scaffold and args.num_props:
        args.seq_len=max_len+scaffold_max_len+args.num_props+3
    elif args.scaffold:
        args.seq_len=max_len+scaffold_max_len+5
    elif args.num_props:
        args.seq_len=max_len+args.num_props+3
    else:
        args.seq_len=max_len+6
    #————————————————————————————————————————————————————————————————————————
    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data=train_data,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight, 
    )
    
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data=val_data,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight, 
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating DIFFUMOL:")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.to(dist_util.dev()) 
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DIFFUMOL"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
