### Import Libraries
import pandas as pd
#from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import gc
from HLAN import HLAN

word_model_path = # Path to word model weight tensor.pt
code_model_path = # Path to code model weight tensor.pt
x_path = # Path to training data
y_path = # Path to training label data
word_weight_tensor = torch.load(word_model_path, weights_only = True)
code_weight_tensor = torch.load(code_model_path, weights_only=True)
x = torch.load(x_path, weights_only = True)
y = torch.load(y_path, weights_only = True).float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Hyperparameters:
### Define hidden size and embedding dim
hidden_size = 100
embed_dim = 100
batch_size = 12
num_sent = 25
lr = 0.01
dropout_prob = 0.5
l2 = 0.0001
calibration = 0.5
num_epochs = 5

model = HLAN(num_sent = num_sent, word_weight_tensor=word_weight_tensor,code_weight_tensor=code_weight_tensor,embed_dim=embed_dim,hidden_size=hidden_size,dropout_prob=dropout_prob)
model.to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2)

def train_model(model, dataloader, optimizer, loss_fn, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        n = 0

        for batch in tqdm(dataloader):
            n += 1
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            inputs.detach().cpu()
            targets.detach().cpu()
            outputs.detach().cpu()
            loss.detach().cpu()

            epoch_loss += loss.item()

            del batch, inputs, targets, outputs, loss

            if str(device).startswith('cuda'):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    gc.collect()
        
        print(epoch_loss / n)
        f1_score = evaluate_model(model, dataloader, calibration, device)
        print(f'Micro F1 Score: {f1_score:.4f}')

def evaluate_model(model, dataloader, calibration, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                predictions = (outputs > calibration).float()

                inputs.detach().cpu()
                targets.detach().cpu()
                predictions.detach().cpu()

                all_predictions.append(predictions)
                all_targets.append(targets)

                del batch, inputs, outputs, predictions, targets

                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
                    gc.collect()

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    micro_f1 = f1_score(all_targets.numpy(), all_predictions.numpy(), average='micro')
    
    return micro_f1

best_micro_f1 = 0.0
dataset = torch.utils.data.TensorDataset(x,y)
dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = True)
eval_dataloader = torch.utils.data.DataLoader(dataset,batch_size = 8, shuffle = False)
train_model(model, dataloader, optimizer, loss_fn, num_epochs, device)