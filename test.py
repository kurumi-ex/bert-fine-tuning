import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer

from mybert import ClassificationBert

train = pd.read_csv('./datasets/sst2/dev.tsv', sep='\t', header=None)

print(train.head())
print(train.info())
print(len(train))

model = ClassificationBert()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')

# 示例输入文本
text = ["This is a positive example.", "This is "]

# 使用BERT的tokenizer进行编码
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

# 打印生成的 input_ids 和 attention_mask
print("input_ids:", inputs['input_ids'])
print("attention_mask:", inputs['attention_mask'])

ans = model(inputs['input_ids'], inputs['attention_mask'])
print(ans)

torch.save(model.state_dict(), './runs/model.pt')
