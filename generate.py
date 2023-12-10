"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start
import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffumol.rounding import denoised_fn_round
from diffumol.text_datasets import load_data_text
import re
from rdkit import Chem
import pandas as pd
import time
from diffumol.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer,
    load_model_emb
)

def create_argparser():
    defaults = dict(model_path='./weights/Moses/moses.pt', step=0, out_dir='./generation_data', top_p=0,value=None,valuebck=None,sample=32)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def check_novelty(gen_smiles, train_smiles): 
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  
        novel = len(gen_smiles) - sum(duplicates)  
        novel_ratio = novel*100./len(gen_smiles)  
    return novel_ratio

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

@th.no_grad()
def main():
    args = create_argparser().parse_args()
    args.step=2000
    args.split='test'
    args.top_p=5
    dist_util.setup_dist()
    logger.configure()
    print(args.clamp_step)
    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0
    
    prop1=args.value
    prop2=args.valuebck

    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)

    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)

    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)
    

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True) 
    data.columns = data.columns.str.lower()
    if 'moses' in args.data_name:
        # needed for moses
        smiles = data[data['split'] != 'test_scaffolds']['smiles']
        # needed for moses
        scaf = data[data['split'] != 'test_scaffolds']['scaffold_smiles']
    else:
        smiles = data[data['source'] != 'test']['smiles']
        scaf = data[data['source'] != 'test']['scaffold_smiles']

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(), 
        loop=False
    )
    smiles=[]
    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"{model_base_name.split('.')[0]}")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    csv_path=os.path.join(out_path, f"molecules_{prop1}_{prop2}.csv")

    all_test_data = []

    idx = 0

    try:
        while True:
            batch, cond = next(data_valid)
            if idx % world_size == rank:
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(dist_util.dev())

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({}) 

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    for cond in iterator:

        if not cond:  
            for i in range(world_size):
                dist.barrier()
            continue

        input_ids_x = cond.pop('input_ids').to(th.float).to(dist_util.dev())
        input_ids_mask = cond.pop('input_mask')       
        
        
        
        #------------------------x_start------------------------
            
        if args.num_props:
            props=input_ids_x[:,:args.num_props].clone()
            props = model.get_props(props)  
            x_start= th.cat([props, model.get_embeds(input_ids_x[:,args.num_props:])], 1)
            input_ids_mask = input_ids_mask[:,args.num_props-1:].contiguous()
        else:
            x_start = model.get_embeds(input_ids_x)
        
        #--------------------------------------------------------
        
        input_ids_mask_ori = input_ids_mask
        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len-args.num_props+1, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )

        sample = samples[-1]

        if args.num_props:
            logits = model.get_logits(sample[:,1:])
            input_ids_mask_ori=input_ids_mask_ori[:,1:]
        else:
            logits = model.get_logits(sample) 

        cands = th.topk(logits, k=1, dim=-1)

        word_lst_recover = []


        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = len(input_mask) - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)
        
        smiles+=word_lst_recover
        
            
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')        
          
    pattern="\[START\](.*?)\[END\]"
    regex = re.compile(pattern)
    molecules=[]  
    
    for smile in smiles:
        temp=regex.findall(smile)
        if len(temp)!=1:
            continue
        completion=temp[0].replace(" ","")
        
        mol = get_mol(completion)
        if mol:
            molecules.append(mol)
            
    print(f"nums of smiles: {len(smiles)}")
    print(f"nums of molecules: {len(molecules)}")
    
    all_dfs=[]
    mol_dict = []

    for i in molecules:
        mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

    results = pd.DataFrame(mol_dict)

    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    unique_smiles = list(set(canon_smiles))
    
    if 'moses' in args.data_name:

        novel_ratio = check_novelty(unique_smiles, set(
            data[data['split'] == 'train']['smiles']))
    else:

        novel_ratio = check_novelty(unique_smiles, set(
            data[data['source'] == 'train']['smiles']))


    results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
    results['sas'] = results['molecule'].apply(
        lambda x: sascorer.calculateScore(x))
    results['logp'] = results['molecule'].apply(
        lambda x: Crippen.MolLogP(x))
    results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))

    results['validity'] = np.round(len(results)/(args.sample), 3)
    results['unique'] = np.round(len(unique_smiles)/len(results), 3)
    results['novelty'] = np.round(novel_ratio/100, 3)
    
    all_dfs.append(results)
    results = pd.concat(all_dfs)
    results.to_csv(csv_path, index=False)
    
    print('Valid ratio: ', np.round(len(results)/(args.sample), 3))
    print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
    print('Novelty ratio: ', np.round(novel_ratio/100, 3))

if __name__ == "__main__":
    main()
