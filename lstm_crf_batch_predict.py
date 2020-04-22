import torch
from tqdm import tqdm
from lstm_crf_batch import BiLSTM_CRF
from BiLstmCrfDataset import BiLstmCrfDataset
import torch.utils.data as tud

import time
import numpy as np
from sklearn.metrics import f1_score, classification_report


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

BATCH_SIZE = 16
MAX_LEN = 200

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def acc_f1(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    f1 = f1_score(y_true, y_pred, average="macro")
    correct = np.sum((y_true==y_pred).astype(int))
    acc = correct/y_pred.shape[0]
    return acc, f1

def class_report(y_pred, y_true):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    classify_report = classification_report(y_true, y_pred)
    print('\n\nclassify_report:\n', classify_report)

if __name__ == '__main__':
    model = torch.load("pytorch_model.bin")
    model.eval()

    data = []
    with open("source_BIO_2014_cropus.txt", 'r', encoding='UTF-8') as fr_1, \
            open("target_BIO_2014_cropus.txt", 'r', encoding='UTF-8') as fr_2:
        for sent, target in tqdm(zip(fr_1, fr_2), desc='text_to_id'):
            chars = sent.strip('\n').split()
            label = target.strip('\n').split()    
            pair=(chars,label)
            data.append(pair)
    
    word_to_ix = {}
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)               
    word_to_ix['<PAD>'] = len(word_to_ix)
    
    labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O" ,"9"]
    tag_to_ix = {"B_PER": 0, "I_PER": 1, "B_T": 2, "I_T": 3, "B_ORG": 4, "I_ORG": 5, "B_LOC": 6, "I_LOC": 7, "O": 8, "<START>": 9, "<STOP>": 10}
    
    dataset = BiLstmCrfDataset(data, word_to_ix, tag_to_ix, max_len = MAX_LEN)
    dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    start=time.time()
    
    y_predicts, y_labels = [], []
    
    for j, (sentence,targets,mask) in enumerate(dataloader):
        if j == int(len(dataset)/BATCH_SIZE):
            break
            
        if j > 7:
            break
        
        valid_loss = model.loss_fn(sentence, targets ,mask)
        valid_predict = model(sentence,mask)
        
        for p in valid_predict:
            y_predicts.extend(p)
        
        targets = targets.view(1,-1)
        targets = targets[targets!=-1]
        
        y_labels.append(targets)
        
        #valid_predict = torch.tensor(valid_predict).long()
        #valid_mask_cnt = mask.numel() - mask.sum().item()
        #valid_eq = torch.eq(valid_predict,targets)
        #valid_predict_total=valid_predict_total+valid_eq.numel()-valid_mask_cnt
        #valid_predict_correct=valid_predict_correct+valid_eq.sum().item()-valid_mask_cnt
        #valid_acc = valid_predict_correct/valid_predict_total

    eval_predicted= torch.Tensor(y_predicts)
    eval_labeled = torch.cat(y_labels,dim=0)
    
    eval_acc, eval_f1 = acc_f1(eval_predicted, eval_labeled)
    class_report(eval_predicted, eval_labeled)
    print("eval_acc,eval_f1:")
    print(eval_acc,eval_f1)
    
    end=time.time()
    print('运行时间:'+str(end-start))

