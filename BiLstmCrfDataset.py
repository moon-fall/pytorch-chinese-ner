import torch
import torch.utils.data as tud

class BiLstmCrfDataset(tud.Dataset):
    def __init__(self, text, word_to_idx,tag_to_ix, max_len):
        super(BiLstmCrfDataset, self).__init__()
        self.text = text
        self.word_to_idx = word_to_idx
        self.tag_to_ix = tag_to_ix
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
        
    def __getitem__(self, idx):
        (chars,labels) = self.text[idx]
        sentence_in = [self.word_to_idx[c] for c in chars]
        label_in = [self.tag_to_ix[l] for l in labels]
        mask = [1] * len(sentence_in)
        if(len(sentence_in)< self.max_len):
            extend_size = self.max_len - len(sentence_in)
            sentence_in.extend([self.word_to_idx['<PAD>']] * extend_size)
            label_in.extend([-1] * extend_size)
            mask.extend([0] * extend_size)
        else:
            sentence_in=sentence_in[:self.max_len]
            label_in=label_in[:self.max_len]
            mask=mask[:self.max_len]
        
        sentence_in = torch.tensor(sentence_in).long()
        label_in = torch.tensor(label_in).long()
        mask = torch.tensor(mask).byte()
        
        return sentence_in,label_in,mask