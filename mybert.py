from torch import nn
from transformers import BertModel


class ClassificationBert(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',cache_dir='./cache')
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        ans = self.classifier(outputs[1])
        return ans
