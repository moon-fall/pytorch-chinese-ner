import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from crf import CRF
from BiLstmCrfDataset import BiLstmCrfDataset

torch.manual_seed(1)


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size,max_len):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.max_len = max_len
        self.crf = CRF(len(tag_to_ix),batch_first=True)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence ,mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        
        tag_seq = self.crf.decode(lstm_feats,mask=mask)
        return tag_seq
        
    def loss_fn(self, sentence, tags ,mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        loss = self.crf(lstm_feats, tags,mask=mask)
        return -loss
        
if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BATCH_SIZE = 16
    MAX_LEN = 200
    
    data = []
    with open("source_BIO_2014_cropus.txt", 'r', encoding='UTF-8') as fr_1, \
            open("target_BIO_2014_cropus.txt", 'r', encoding='UTF-8') as fr_2:
        for sent, target in tqdm(zip(fr_1, fr_2), desc='text_to_id'):
            chars = sent.strip('\n').split()
            label = target.strip('\n').split()    
            pair=(chars,label)
            data.append(pair)
            
    N = len(data)
    test_size = int(N * 0.2)
    
    valid_data = data[:test_size]
    train_data = data[test_size:]
    
    word_to_ix = {}
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)               
    word_to_ix['<PAD>'] = len(word_to_ix)
    
    labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O" ,"9"]
    tag_to_ix = {"B_PER": 0, "I_PER": 1, "B_T": 2, "I_T": 3, "B_ORG": 4, "I_ORG": 5, "B_LOC": 6, "I_LOC": 7, "O": 8, "<START>": 9, "<STOP>": 10}
    
    train_dataset = BiLstmCrfDataset(train_data, word_to_ix, tag_to_ix, max_len = MAX_LEN)
    train_dataloader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    valid_dataset = BiLstmCrfDataset(valid_data, word_to_ix, tag_to_ix, max_len = MAX_LEN)
    valid_dataloader = tud.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,BATCH_SIZE ,MAX_LEN)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    best_acc = 0
    NUM_EPOCHS = 1
    print_interval = 10 
    for e in range(NUM_EPOCHS):
        train_loss=0
        train_predict_total=0
        train_predict_correct=0
        for i, (sentence,targets,mask) in enumerate(train_dataloader):
            if i == int(len(train_dataset)/BATCH_SIZE):
                break
            
            model.train()
            model.zero_grad()
            
            loss = model.loss_fn(sentence, targets ,mask)
            #print(loss.item())

            predict = model(sentence,mask)

            for p in predict:
                if (len(p) < model.max_len):
                    extend_size = model.max_len - len(p)
                    p.extend([0] * extend_size)

            predict = torch.tensor(predict).long()

            #print(predict)
            #print(targets)

            mask_cnt = mask.numel() - mask.sum().item()
            #print(mask_cnt)

            eq = torch.eq(predict,targets) 
            acc = (eq.sum().item()-mask_cnt)/(eq.numel()-mask_cnt)
            
            loss.backward()
            optimizer.step()
            
            train_loss=train_loss+loss
            train_predict_total=train_predict_total+eq.numel()-mask_cnt
            train_predict_correct=train_predict_correct+eq.sum().item()-mask_cnt
            train_acc = train_predict_correct/train_predict_total
            
            if(i%print_interval==0):
                print(time.asctime(time.localtime(time.time()))+"    "+"epoch:"+str(e)+"    "+"count:"+str(i)+"    "+"train_loss_avg:"+format(train_loss.item()/print_interval, '.4f')+"    "+"train_acc:"+format(train_acc, '.4f'))
                
                train_loss=0
                train_predict_total=0
                train_predict_correct=0
            
            if((i>0) & (i%200==0)):
                valid_predict_total=0
                valid_predict_correct=0
                
                for j, (sentence,targets,mask) in enumerate(valid_dataloader):
                    if j == int(len(valid_dataset)/BATCH_SIZE):
                        break
                    model.eval()
                    
                    valid_loss = model.loss_fn(sentence, targets ,mask)
                    valid_predict = model(sentence,mask)
                    
                    for p in valid_predict:
                        if (len(p) < model.max_len):
                            extend_size = model.max_len - len(p)
                            p.extend([0] * extend_size)

                    valid_predict = torch.tensor(valid_predict).long()
                    valid_mask_cnt = mask.numel() - mask.sum().item()
                    valid_eq = torch.eq(valid_predict,targets)
                    valid_predict_total=valid_predict_total+valid_eq.numel()-valid_mask_cnt
                    valid_predict_correct=valid_predict_correct+valid_eq.sum().item()-valid_mask_cnt
                    valid_acc = valid_predict_correct/valid_predict_total
                    #print("epoch:"+str(e)+"    "+"valid_loss:"+format(valid_loss.item(), '.4f')+"    "+"valid_acc:"+format(valid_acc, '.4f'))
                '''
                for sentence, tags in valid_data:
                    model.eval()
                    
                    valid_sentence_in = prepare_sequence(sentence, word_to_ix)
                    valid_targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
                    valid_loss = model.neg_log_likelihood(valid_sentence_in, valid_targets)
                    valid_predict = torch.Tensor(model(valid_sentence_in)[1]).long()
                    valid_eq = torch.eq(valid_predict,valid_targets)
                    valid_predict_total=valid_predict_total+valid_eq.numel()
                    valid_predict_correct=valid_predict_correct+valid_eq.sum().item()
                    valid_acc = valid_predict_correct/valid_predict_total
                    #print("epoch:"+str(epoch)+"    "+"valid_loss:"+format(valid_loss.item(), '.4f')+"    "+"valid_acc:"+format(valid_acc, '.4f'))
                '''  
                
                valid_acc_total = valid_predict_correct/valid_predict_total
                print(time.asctime(time.localtime(time.time()))+"    "+"epoch:"+str(e)+"    "+"count:"+str(i)+"    "+"valid_acc_total:"+format(valid_acc_total, '.4f'))
                
                if valid_acc_total > best_acc:
                    best_acc = valid_acc_total
                    torch.save(model, "pytorch_model.bin") 
            
