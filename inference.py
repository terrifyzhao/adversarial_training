import torch
from transformers import BertTokenizerFast
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = '/data/joska/ptm/roberta'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = torch.load('best_model.p').to(device)
model.eval()


def load_data(filename):
    question = []
    label = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            question.append(l['text'])
            label.append(int(l['label']))
    return question, label


question, label = load_data('./short_news/test.json')

encodings = tokenizer(question,
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


dataset = Dataset(encodings, label)
loader = DataLoader(dataset, batch_size=512)

acc = 0
for batch in tqdm(loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss, output = outputs.loss, outputs.logits
    acc += accuracy_score(labels.cpu().numpy(), output.argmax(dim=1).cpu().numpy())

print(acc / len(loader))
