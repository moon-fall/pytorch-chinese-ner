import torch
from tqdm import tqdm
from lstm_crf import BiLSTM_CRF

import time
import numpy as np
from sklearn.metrics import f1_score, classification_report


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

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


model = torch.load("pytorch_model.bin")
model.eval()

data = []
with open("source_BIO_2014_cropus.txt", 'r', encoding='UTF-8') as fr_1, \
        open("target_BIO_2014_cropus.txt", 'r', encoding='UTF-8') as fr_2:
    for sent, target in tqdm(zip(fr_1, fr_2), desc='text_to_id'):
        #print(sent)
        #print(target)
        
        chars = sent.strip('\n').split()
        label = target.strip('\n').split()

        pair=(chars,label)
        
        #print(pair)
        
        data.append(pair)

labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O" ,"9"]

tag_to_ix = {"B_PER": 0, "I_PER": 1, "B_T": 2, "I_T": 3, "B_ORG": 4, "I_ORG": 5, "B_LOC": 6, "I_LOC": 7, "O": 8, START_TAG: 9, STOP_TAG: 10}

word_to_ix = {}
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)



start=time.time()

y_predicts, y_labels = [], []

for i in range(100):
    precheck_sent = prepare_sequence(data[i][0], word_to_ix)
    #print(data[i][1])
    label=[tag_to_ix[data[i][1][j]] for j in range(len(data[i][1]))]
    
    y_labels.extend(label)
    _ , predicts=model(precheck_sent)
    y_predicts.extend(predicts)

print(type(y_predicts))
print(type(y_predicts[0]))
print(y_predicts)
print(y_predicts[0])

eval_predicted = torch.Tensor(y_predicts)
eval_labeled = torch.Tensor(y_labels)


eval_acc, eval_f1 = acc_f1(eval_predicted, eval_labeled)
class_report(eval_predicted, eval_labeled)
print("eval_acc,eval_f1:")
print(eval_acc,eval_f1)

end=time.time()
print('运行时间:'+str(end-start))

