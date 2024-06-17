import pandas as pd
from torch.utils.data import Dataset
import numpy as np

WORD_EMBEDDING_MAX_LENGTH = 128
RAW_MAPPING_COLUMNS = ['CVE-ID', 'CVE-Description', 'CWE-ID', 'CWE-Name', 'CWE-Description']


class CVEsCWEsRawMappingDataset(Dataset):
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path, encoding='latin1')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row[RAW_MAPPING_COLUMNS]

    def get_all_cwe_ids(self):
        return self.df['CWE-ID'].unique()


class CVECWEsWordEmbeddingsDataset(Dataset):
    def __init__(self, file_path: str):
        self.embeddings = np.load(file_path, allow_pickle=True)['arr_0']

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]
