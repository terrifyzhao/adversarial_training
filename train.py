from transformers import BertTokenizerFast
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
import os
import json
import copy


def fix_seed(seed):
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(2021)


def load_data(filename):
    question = []
    label = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            question.append(l['text'])
            label.append(l['label'])
    return question, label


def read_news_data():
    train_question, train_label = load_data('./short_news/train.json')
    valid_question, valid_label = load_data('./short_news/val.json')

    return train_question, train_label, valid_question, valid_label


news_train_question, news_train_label, news_valid_question, news_valid_label = read_news_data()

data = copy.deepcopy(news_train_question)
data.append(news_valid_question)
sentence_len = [len(s) for s in data]
print(np.mean(sentence_len))
print(np.percentile(sentence_len, 80))
print(np.percentile(sentence_len, 90))

model_path = '/data/joska/ptm/roberta'

tokenizer = BertTokenizerFast.from_pretrained(model_path)
train_encodings = tokenizer(news_train_question,
                            return_tensors='pt',
                            truncation=True,
                            padding='max_length',
                            max_length=32)
valid_encodings = tokenizer(news_valid_question,
                            return_tensors='pt',
                            truncation=True,
                            padding='max_length',
                            max_length=32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(train_encodings, news_train_label)
valid_dataset = Dataset(valid_encodings, news_valid_label)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
load_model = False
if os.path.exists('best_model.p') and load_model:
    print('************load model************')
    model = torch.load('best_model.p')
else:
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=15)
    model.to(device)
model.train()

BATCH_SIZE = 1024

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

optim = AdamW(model.parameters(), lr=5e-5)


def train_func():
    train_loss = 0
    train_acc = 0
    pbar = tqdm(train_loader)
    for batch in pbar:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss, output = outputs.loss, outputs.logits
        train_loss += loss.item()
        loss.backward()
        optim.step()
        acc = accuracy_score(labels.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
        train_acc += acc

        pbar.update()
        pbar.set_description(f'loss:{loss.item():.4f}, acc:{acc:.4f}')

    return train_loss / len(train_loader), train_acc / len(train_loader)


def test_func():
    valid_loss = 0
    valid_acc = 0
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, output = outputs.loss, outputs.logits
            valid_loss += loss.item()
            valid_acc += accuracy_score(labels.cpu().numpy(), output.argmax(dim=1).cpu().numpy())

    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)


min_valid_loss = float('inf')
for epoch in range(100):
    print('************start train************')
    train_loss, train_acc = train_func()
    print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
    print('************start valid************')
    valid_loss, valid_acc = test_func()
    print(f'valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}')

    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
        torch.save(model, 'best_model.p')
        print('save model done')
