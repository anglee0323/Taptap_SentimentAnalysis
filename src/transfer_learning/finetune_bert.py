import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件并加载数据
data = pd.read_csv('/Users/wallanceleon/Desktop/Sentiment Analysis/dataset/taptap_review.csv')

# 拆分数据集为训练集和测试集
X = data.iloc[:, 0]  # 文本数据列
y = data.iloc[:, 1]  # 标签列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 准备数据集
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(y_train.tolist())
train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_mask, train_labels)

test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)
test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(y_test.tolist())
test_dataset = torch.utils.data.TensorDataset(test_input_ids, test_attention_mask, test_labels)

# 设置训练参数
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设置优化器和损失函数
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# 微调BERT模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        train_loss += loss.item()
        train_acc += accuracy_score(labels.detach().cpu().numpy(), logits.argmax(dim=1).detach().cpu().numpy())

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # 在测试集上进行评估
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item()
            test_acc += accuracy_score(labels.detach().cpu().numpy(), logits.argmax(dim=1).detach().cpu().numpy())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
