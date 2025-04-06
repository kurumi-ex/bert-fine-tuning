from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from mybert import ClassificationBert
import mydata

if not os.path.exists("runs"):
    os.makedirs("runs")

# 参数
model = ClassificationBert()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
epochs = 5
lr = 1e-4
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

# 加载 SST-2 数据集
train_dataset = mydata.SentenceDataset('./datasets/sst2/train.tsv', tokenizer)
val_dataset = mydata.SentenceDataset('./datasets/sst2/dev.tsv', tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# 训练
best_model = None
last_model = None
loss_history = []
acc_history = []
best_acc = 0
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}--------------------------')
    model.train()  # 进入训练模式
    total_train_loss = 0

    for batch in train_dataloader:
        x = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        y_pred = model(x, attention_mask=attention_mask)
        loss = loss_fn(y_pred, labels)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    loss_history.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Training loss: {avg_train_loss}")

    model.eval()
    acc = 0
    total_val_loss = 0
    total = 0
    for batch in val_dataloader:
        x = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            y_pred = model(x, attention_mask=attention_mask)
            loss = loss_fn(y_pred, labels)
            total_val_loss += loss.item()
            pred = torch.argmax(y_pred, dim=1)
            acc += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = acc / total
    avg_eval_loss = total_val_loss / len(val_dataloader)
    acc_history.append(accuracy)
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), './runs/best.pt')
    print(f"Epoch {epoch + 1},Validation loss: {avg_eval_loss}, Validation Accuracy: {accuracy}")

torch.save(model.state_dict(), './runs/last.pt')

# 绘制损失曲线
plt.subplot(1, 2, 1)  # 1行2列，第1个子图
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Loss', color='blue', marker='o')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)  # 1行2列，第2个子图
plt.plot(range(1, len(acc_history) + 1), acc_history, label='Accuracy', color='green', marker='o')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)

# 调整布局
plt.tight_layout()
plt.show()
