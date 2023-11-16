# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2
import pandas as pd
import re

def load_data_text(
    batch_size, 
    seq_len, 
    data=None,
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus2(data_args, seq_len, data, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb,
    )

    if split != 'test':
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not deterministic,
            num_workers=4,
        )

    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len,data_args):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'],num_props=data_args.num_props,scaffold=data_args.scaffold)
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        if data_args.num_props:
            for i in range(len(group_lst['input_id_x'])):
                end_token = group_lst['input_id_y'][i][-1]
                src = group_lst['input_id_x'][i]
                trg = group_lst['input_id_y'][i][:-1]
                while len(src) + len(trg) > seq_len - 2:
                    if len(src)>len(trg):
                        src.pop()
                    elif len(src)<len(trg):
                        trg.pop()
                    else:
                        src.pop()
                        trg.pop()
                trg.append(end_token)
                lst.append(src + [vocab_dict.sep_token_id] + trg)
                mask.append([0]*(len(src)+1))
        else:
            for i in range(len(group_lst['input_id_x'])):
                end_token = group_lst['input_id_y'][i][-1]
                src = group_lst['input_id_x'][i][:-1]
                trg = group_lst['input_id_y'][i][:-1]
                while len(src) + len(trg) > seq_len - 3:
                    if len(src)>len(trg):
                        src.pop()
                    elif len(src)<len(trg):
                        trg.pop()
                    else:
                        src.pop()
                        trg.pop()
                src.append(end_token)
                trg.append(end_token)
                lst.append(src + [vocab_dict.sep_token_id] + trg)
                mask.append([0]*(len(src)+1))    
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):
    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src':[], 'trg': []}
    
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            content = json.loads(row)
            sentence_lst['src'].append(content['src'].strip())
            sentence_lst['trg'].append(content['trg'].strip())

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():

            input_ids = self.text_datasets['train'][idx]['input_ids']
            tmp=torch.tensor(input_ids,dtype=torch.int64)
            tmp[:5]=0
            hidden_state = self.model_emb(tmp)

            arr = np.array(hidden_state, dtype=np.float32) 
            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

            return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def get_corpus2(data_args, seq_len, data=None, loaded_vocab=None):
    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.data_name, data_args.data_path))
    
    sentence_lst = {'src':[], 'trg': []}
    
    if data is not None:
        for _,row in data.iterrows():
            if data_args.num_props and data_args.scaffold:
                prop=row[data_args.props].values.tolist()
                prop=[str(i) for i in prop]
                prop.append(row['scaffold_smiles'].strip())
                sentence_lst['src'].append(prop)
            elif data_args.num_props:
                sentence_lst['src'].append(row[data_args.props].values.tolist())
            elif data_args.scaffold:
                sentence_lst['src'].append(row['scaffold_smiles'].strip())
            else:
                sentence_lst['src'].append('[UNCONDITION]')
            sentence_lst['trg'].append(row['smiles'].strip())
            
    else:
        for _ in range(data_args.sample):
            if data_args.num_props and data_args.scaffold:
                sentence_lst['src'].append([str(0.9),"C1=NCN=CO1"])
            elif data_args.num_props:
                sentence_lst['src'].append([0.4,3.,2.])
            elif data_args.scaffold:
                 sentence_lst['src'].append((data_args.scaffold).strip())
            else:
                sentence_lst['src'].append('[UNCONDITION]')
                
            sentence_lst['trg'].append('[PAD]')

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len,data_args)
    return train_dataset

if __name__ == "__main__":
    args={"dataset":"moses_test","data_name":"moses2","data_path":""}
    data = pd.read_csv(args.get("data_path"))
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    if 'moses' in args.get("data_name"):
        train_data = data[data['split'] == 'train'].reset_index(
            drop=True)
    else:
        train_data = data[data['source'] == 'train'].reset_index(
            drop=True)


    if 'moses' in args.get("data_name"):
        val_data = data[data['split'] == 'test'].reset_index(
            drop=True)
    else:
        val_data = data[data['source'] == 'val'].reset_index(
            drop=True)


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
    get_corpus2(args,max_len,train_data)