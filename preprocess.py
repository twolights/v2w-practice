import functools

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from dataset import CVEsCWEsRawMappingDataset
from utils import get_device, get_word_embeddings

device = get_device()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
get_embedding = functools.partial(get_word_embeddings,
                                  tokenizer=tokenizer, model=model, device=device)

raw_mapping = CVEsCWEsRawMappingDataset('./input/datasets/2023_CVE_CWE.csv')
output_path = './output/preprocessed/2023_CVE_CWE.npz'

output = []

cwe_embedding_map = {}

with torch.no_grad():
    for i in range(len(raw_mapping)):
        print(f'{i + 1}/{len(raw_mapping)}')
        row = raw_mapping[i]
        cve_id, cwe_id = row['CVE-ID'], row['CWE-ID']
        if cwe_id not in cwe_embedding_map:
            cwe_description = row['CWE-Description']
            cwe_embedding_map[cwe_id] = get_embedding(cwe_description)
        cve_description = row['CVE-Description']
        cve_embeddings: torch.tensor = get_embedding(cve_description)
        cwe_embeddings: torch.tensor = cwe_embedding_map[cwe_id]

        entry = [cve_id, cwe_id, cve_embeddings.cpu().numpy(), cwe_embeddings.cpu().numpy()]
        output.append(np.asarray(entry, dtype=object))

np.savez(output_path, output)
