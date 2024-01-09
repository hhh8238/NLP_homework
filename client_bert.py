import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# 定义微调的分类模型
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# 加载和预处理数据
def load_data():
    # 请根据实际项目中的数据加载和预处理过程进行实现

# 定义训练函数
def train(model, train_data, num_epochs, batch_size, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (input_ids, attention_mask, labels) in enumerate(train_data):
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data)
        epoch_acc = total_correct / total_samples

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# 加载和预处理数据
train_data = load_data()

# 实例化分类模型
num_classes = 2  # 根据具体任务设置类别数
model = BertClassifier(num_classes)

# 设置超参数
num_epochs = 5
batch_size = 32
learning_rate = 1e-5

# 训练模型
train(model, train_data, num_epochs, batch_size, learning_rate)