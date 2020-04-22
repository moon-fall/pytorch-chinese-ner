import itertools
import math
import random

from pytest import approx
import pytest
import torch
import torch.nn as nn

from torchcrf import CRF

RANDOM_SEED = 1478754

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def compute_score(crf, emission, tag):
    # emission: (seq_length, num_tags)
    assert emission.dim() == 2
    assert emission.size(0) == len(tag)
    assert emission.size(1) == crf.num_tags
    assert all(0 <= t < crf.num_tags for t in tag)

    # Add transitions score
    score = crf.start_transitions[tag[0]] + crf.end_transitions[tag[-1]]
    for cur_tag, next_tag in zip(tag, tag[1:]):
        score += crf.transitions[cur_tag, next_tag]

    # Add emission score
    for emit, t in zip(emission, tag):
        score += emit[t]

    return score


def make_crf(num_tags=5, batch_first=False):
    return CRF(num_tags, batch_first=batch_first)


def make_emissions(crf, seq_length=3, batch_size=2):
    em = torch.randn(seq_length, batch_size, crf.num_tags)
    if crf.batch_first:
        em = em.transpose(0, 1)
    return em


def make_tags(crf, seq_length=3, batch_size=2):
    # shape: (seq_length, batch_size)
    ts = torch.tensor([[random.randrange(crf.num_tags)
                        for b in range(batch_size)]
                       for _ in range(seq_length)],
                      dtype=torch.long)
    if crf.batch_first:
        ts = ts.transpose(0, 1)
    return ts

    
    
from test import *
    
crf = make_crf()
batch_size = 10

# shape: (seq_length, batch_size, num_tags)
emissions = make_emissions(crf, batch_size=batch_size)
# shape: (seq_length, batch_size)
tags = make_tags(crf, batch_size=batch_size)

llh = crf(emissions, tags)
assert torch.is_tensor(llh)
assert llh.shape == ()

total_llh = 0.
for i in range(batch_size):
    # shape: (seq_length, 1, num_tags)
    emissions_ = emissions[:, i, :].unsqueeze(1)
    # shape: (seq_length, 1)
    tags_ = tags[:, i].unsqueeze(1)
    # shape: ()
    total_llh += crf(emissions_, tags_)

assert llh.item() == approx(total_llh.item())



import torch
from crf import CRF
num_tags = 5  # number of tags is 5
model = CRF(num_tags)

seq_length = 3
batch_size = 2
emissions = torch.randn(seq_length, batch_size, num_tags)
tags = torch.tensor([
[0, 1], [2, 4], [3, 1]
], dtype=torch.long)
model(emissions, tags)


mask = torch.tensor([
  [1, 1], [1, 1], [1, 0]
], dtype=torch.uint8)

model(emissions, tags, mask=mask)



