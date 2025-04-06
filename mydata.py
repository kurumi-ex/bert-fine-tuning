import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SentenceDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        # 加载数据集
        self.data = pd.read_csv(path, sep='\t', header=None)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 获取一条数据
        text = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 0]
        # 对于格式的校验
        label = label if isinstance(label, np.integer) else np.int64(label)

        # 使用 tokenizer 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # 返回编码后的数据和标签
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去除 batch 维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
