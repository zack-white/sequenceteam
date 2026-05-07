#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import torch as th
from pathlib import Path
from deepgo.utils import Ontology, NAMESPACES
from deepgo.models import DeepGOModel
from deepgo.data import load_normal_forms

th.cuda.empty_cache()

@ck.command()
@ck.option('--pkl-file', '-pf', required=True, help='Input .pkl file for proteins and features')
@ck.option('--data-root', '-dr', default='data', help='Root folder for GO models and terms')
@ck.option('--threshold', '-t', default=0.1, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option('--device', '-d', default='cuda', help='Device to run model on')
@ck.option('--ontology', '-ont', required=True, type=ck.Choice(['mf','bp','cc']), help='Ontology to predict')

def main(pkl_file, data_root, threshold, batch_size, device, ontology):

    # Load protein features from .pkl
    df = pd.read_pickle(pkl_file)
    proteins = df['proteins'].values
    data = df['esm2']  # assuming ESM embeddings are saved here
    if not isinstance(data, th.Tensor):
        data = th.stack([th.tensor(x, dtype=th.float32) for x in data])
    
    n_samples = len(proteins)
    print(f'Loaded {n_samples} proteins from {pkl_file}')

    # Load GO and term info
    go_file = f'{data_root}/go.obo'
    go_norm = f'{data_root}/go-plus.norm'
    go = Ontology(go_file, with_rels=True)

    terms_file = f'{data_root}/{ontology}/terms.pkl'
    out_file = Path(pkl_file).with_suffix(f'_preds_{ontology}.tsv.gz')
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    n_terms = len(terms_dict)

    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(go_norm, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    # Initialize model
    ent_models = {
        'mf': [0, 1, 2, 5, 6, 8],
        'bp': [2, 5, 6, 7, 8, 9],
        'cc': [1, 3, 4, 5, 6, 7]
    }

    sum_preds = np.zeros((n_samples, n_terms), dtype=np.float32)
    model = DeepGOModel(2560, n_terms, n_zeros, n_rels, device).to(device)

    for mn in ent_models[ontology]:
        model_file = f'{data_root}/{ontology}/deepgozero_esm_plus_{mn}.th'
        model.load_state_dict(th.load(model_file, map_location=device))
        model.eval()

        with th.no_grad():
            steps = int(np.ceil(n_samples / batch_size))
            preds = []
            for i in range(steps):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_features = data[start:end].to(device)
                logits = model(batch_features)
                preds.append(logits.detach().cpu().numpy())
            preds = np.concatenate(preds)
        sum_preds += preds

    preds = sum_preds / len(ent_models[ontology])

    # Save predictions
    with gzip.open(out_file, 'wt') as f:
        for i in range(n_samples):
            above_thresh = np.argwhere(preds[i] >= threshold).flatten()
            for j in above_thresh:
                term_name = go.get_term(terms[j])['name']
                f.write(f'{proteins[i]}\t{terms[j]}\t{preds[i,j]:0.3f}\n')

    print(f'Predictions saved to {out_file}')

if __name__ == '__main__':
    main()