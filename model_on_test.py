from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

from mybert import ClassificationBert
import mydata

# 参数
model = ClassificationBert()  # 初始化模型结构
model.load_state_dict(torch.load('./runs/best.pt'))  # 加载参数
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

test_dataset = mydata.SentenceDataset('./datasets/sst2/test.tsv', tokenizer)

test_dataloader = DataLoader(test_dataset, batch_size=16)


acc = 0
total = 0
for batch in test_dataloader:
    x = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        y_pred = model(x, attention_mask=attention_mask)
        pred = torch.argmax(y_pred, dim=1)
        acc += (pred == labels).sum().item()
        total += labels.size(0)

accuracy = acc / total

print(f"On the test set, Accuracy: {accuracy}")
