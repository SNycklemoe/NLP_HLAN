### Import Libraries
import pandas as pd
#from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score,roc_auc_score
import gc
import random
import os
from HLAN import HLAN

def seed_everything(seed=42):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


word_model_path = ## Path to word-emb model
code_model_path = ## Path to code-emb model
x_path = ## Path to train_word-emb 
y_path = ## Path to train_code-emb 
word_weight_tensor = torch.load(word_model_path, weights_only = True)
code_weight_tensor = torch.load(code_model_path, weights_only=True)
x = torch.load(x_path, weights_only = True)
y = torch.load(y_path, weights_only = True).float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

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
num_epochs = 100

model = HLAN(num_sent = num_sent, word_weight_tensor=word_weight_tensor,code_weight_tensor=code_weight_tensor,embed_dim=embed_dim,hidden_size=hidden_size,dropout_prob=dropout_prob)
model.to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2)
def train_model(model, train_dataloader, dev_dataloader,optimizer, loss_fn, num_epochs, device):
    train_loss = list()
    dev_micro_f1, dev_macro_f1, dev_micro_auroc,dev_macro_auroc = list(),list(),list(),list()
    for epoch in range(num_epochs):
        print('Epoch: '+str(epoch))
        model.train()
        epoch_loss = 0
        n = 0

        for batch in tqdm(train_dataloader):
        # for batch in train_dataloader:
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
        train_loss.append(epoch_loss)
        epoch_dev_micro_f1, epoch_dev_macro_f1, epoch_dev_micro_auroc,epoch_dev_macro_auroc = evaluate_model(model, dev_dataloader, calibration, device)
        print(f'Micro F1 Score: {epoch_dev_micro_f1:.4f}')
        print(f'Macro F1 Score: {epoch_dev_macro_f1:.4f}')
        print(f'Micro AUROC : {epoch_dev_micro_auroc:.4f}')
        print(f'Macro AUROC: {epoch_dev_macro_auroc:.4f}')
        dev_micro_f1.append(epoch_dev_micro_f1)
        dev_macro_f1.append(epoch_dev_macro_f1)
        dev_micro_auroc.append(epoch_dev_micro_auroc)
        dev_macro_auroc.append(epoch_dev_micro_auroc)
    return train_loss, dev_micro_f1,dev_macro_f1,dev_micro_auroc,dev_macro_auroc

def evaluate_model(model, dataloader, calibration, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
        # for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                predictions = (outputs > calibration).float()

                inputs.detach().cpu()
                targets.detach().cpu()
                predictions.detach().cpu()
                outputs.detach().cpu()

                all_predictions.append(predictions)
                all_targets.append(targets)
                all_outputs.append(outputs)

                del batch, inputs, outputs, predictions, targets

                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
                    gc.collect()

    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    micro_f1 = f1_score(all_targets, all_predictions, average='micro')
    micro_auroc = roc_auc_score(all_targets, all_outputs, average='micro')
    macro_f1 = f1_score(all_targets, all_predictions, average='macro')
    macro_auroc = roc_auc_score(all_targets, all_outputs, average='macro')
    # print(all_predictions)
    return micro_f1,macro_f1,micro_auroc,macro_auroc

best_micro_f1 = 0.0
dataset = torch.utils.data.TensorDataset(x,y)
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 12, shuffle = True)
eval_dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = False)
seed_everything()
train_loss, dev_micro_f1,dev_macro_f1,dev_micro_auroc,dev_macro_auroc = train_model(model, dataloader, eval_dataloader,optimizer, loss_fn, num_epochs, device)
np.save('train_loss_'+str(num_epochs)+'.npy',train_loss)
np.save('dev_micro_f1_'+str(num_epochs)+'.npy',dev_micro_f1)
np.save('dev_macro_f1_'+str(num_epochs)+'.npy',dev_macro_f1)
np.save('dev_micro_auroc_'+str(num_epochs)+'.npy',dev_micro_auroc)
np.save('dev_macro_auroc_'+str(num_epochs)+'.npy',dev_macro_auroc)
