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
from BERT_HLAN import BERT_HLAN
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import argparse

def seed_everything(seed=42):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_model(model, train_dataloader, dev_dataloader,optimizer, loss_fn, num_epochs, device,calibration):
    seed_everything()
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
            inputs, attn_mask, targets = batch
            inputs, attn_mask, targets = inputs.to(device), attn_mask.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, attn_mask)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            inputs.detach().cpu()
            attn_mask.detach().cpu()
            targets.detach().cpu()
            outputs.detach().cpu()
            loss.detach().cpu()
            epoch_loss += loss.item()

            del batch, inputs,attn_mask, targets, outputs, loss

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
                inputs, attn_mask, targets = batch
                inputs = inputs.to(device)
                attn_mask = attn_mask.to(device)
                targets = targets.to(device)
                outputs = model(inputs, attn_mask)
                predictions = (outputs > calibration).float()

                inputs.detach().cpu()
                attn_mask.detach().cpu()
                targets.detach().cpu()
                predictions.detach().cpu()
                outputs.detach().cpu()

                all_predictions.append(predictions)
                all_targets.append(targets)
                all_outputs.append(outputs)

                del batch, inputs, attn_mask, outputs, predictions, targets

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

### Hyperparameters:
### Define hidden size and embedding dim
def main(args):
    hidden_size = args.hidden_size
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    num_sent = args.num_sent
    lr = args.lr
    dropout_prob = args.dropout
    l2 = args.l2
    calibration = args.calibration
    num_epochs = args.num_epochs
    use_label_embeddings = args.use_label
    freeze_emb = args.freeze_emb
    bert_emb = args.bert_emb
    scenario = ''
    # print(scenario)
    if use_label_embeddings == 'Yes':
        use_label_embeddings = True
        scenario += '_use_label_emb'
    else:
        use_label_embeddings = False
    if freeze_emb == 'Yes':
        freeze_emb = True
        scenario += '_freeze_label_emb'
    else:
        freeze_emb = False
    scenario += '_'+bert_emb
    print(scenario)
    code_model_path ='code-emb_model_weights.pt' ## Path to code-emb model
    train_x_path ='train_tok_bert_embed_bluebert.pt' ## Path to train_tok_bert_embed
    train_x_mask_path = 'train_mask_bert_embed_bluebert.pt' ## Path to train_mask_bert_embed
    train_y_path = 'train_code-emb.pt'## Path to train_code-emb )
    code_weight_tensor = torch.load(code_model_path, weights_only=True)
    train_x = torch.load(train_x_path, weights_only = True)
    train_x_mask = torch.load(train_x_mask_path, weights_only = True)
    train_y = torch.load(train_y_path, weights_only = True).float()
    subset = args.subset
    eval_x_path = subset+'_tok_bert_embed_bluebert.pt' ## Path to dev_tok_bert_embed
    eval_x_mask_path = subset+'_mask_bert_embed_bluebert.pt' ## Path to dev_mask_bert_embed
    eval_y_path = subset+'_code-emb.pt'## Path to dev_code-emb 
    eval_x = torch.load(eval_x_path, weights_only = True)
    eval_x_mask = torch.load(eval_x_mask_path, weights_only = True)
    eval_y = torch.load(eval_y_path, weights_only = True).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if bert_emb == "bluebert":
        bert_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
    elif bert_emb == "bioclinical":
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    else:
        bert_model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    seed_everything()
    model = BERT_HLAN(num_sent = num_sent, bert_model = bert_model, code_weight_tensor=code_weight_tensor, 
                      embed_dim=embed_dim, hidden_size=hidden_size, dropout_prob=dropout_prob,
                      use_label_embeddings = use_label_embeddings, freeze_embed = freeze_emb)
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2)
    best_micro_f1 = 0.0
    train_dataset = torch.utils.data.TensorDataset(train_x,train_x_mask,train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size, shuffle = True)
    eval_dataset = torch.utils.data.TensorDataset(eval_x,eval_x_mask,eval_y)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,batch_size = batch_size, shuffle = True)
    seed_everything()
    train_loss, eval_micro_f1,eval_macro_f1,eval_micro_auroc,eval_macro_auroc = train_model(model, train_dataloader, eval_dataloader,optimizer, loss_fn, num_epochs, device,calibration = calibration)
    np.save(subset+'_loss_'+str(num_epochs)+scenario+'_'+str(lr)+'.npy',train_loss)
    np.save(subset+'_micro_f1_'+str(num_epochs)+scenario+'_'+str(lr)+'.npy',eval_micro_f1)
    np.save(subset+'_macro_f1_'+str(num_epochs)+scenario+'_'+str(lr)+'.npy',eval_macro_f1)
    np.save(subset+'_micro_auroc_'+str(num_epochs)+scenario+'_'+str(lr)+'.npy',eval_micro_auroc)
    np.save(subset+'_macro_auroc_'+str(num_epochs)+scenario+'_'+str(lr)+'.npy',eval_macro_auroc)
    torch.save(model.state_dict(), 'HLAN_'+str(num_epochs)+scenario+'_'+str(lr)+'pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size',
                        default = 100,
                        type = int)
    parser.add_argument('--embed_dim',
                        default = 768,
                        type = int)
    parser.add_argument('--batch_size',
                        default = 12,
                        type = int)
    parser.add_argument('--num_sent',
                        default = 100,
                        type = int)
    parser.add_argument('--lr',
                        default = 0.01,
                        type = float)
    parser.add_argument('--dropout',
                        default = 0.5,
                        type = float)
    parser.add_argument('--l2',
                        default = 0.0001,
                        type = float)
    parser.add_argument('--calibration',
                        default = 0.5,
                        type = float)
    parser.add_argument('--num_epochs',
                        default = 100,
                        type = int)
    parser.add_argument('--use_label',
                        default = 'Yes',
                        type = str)
    parser.add_argument('--freeze_emb',
                        default = 'Yes',
                        type = str)
    parser.add_argument('--subset',
                        default = 'dev',
                        type = str)
    parser.add_argument('--bert_emb',
                        default = 'bluebert',
                        type = str)

    main(parser.parse_args())






