# %%
T = 8

import sys
sentence = sys.argv[1]
print(sentence)

# Rest of your code...
input = sentence[:T]
input

# %%
vocabs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '\n', ' ']

stoi = {s: i for i, s in enumerate(vocabs)}
itos = {i: s for i, s in enumerate(vocabs)}

stoi

# %%
[stoi[s] for s in input]

# %%
import numpy as np
emb_dim = 14

a = .1
b = 5

W_emb = np.array([
    [a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],  # 0
    [1, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],  # 1
    [2, 0, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],  # 2
    [3, 0, 0, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],  # 3
    [4, 0, 0, 0, a, 0, 0, 0, 0, 0, 0, 0, 0, 0,],  # 4
    [5, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, 0, 0, 0,],  # 5
    [6, 0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, 0, 0,],  # 6
    [7, 0, 0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0, 0,],  # 7
    [8, 0, 0, 0, 0, 0, 0, 0, a, 0, 0, 0, 0, 0,],  # 8
    [9, 0, 0, 0, 0, 0, 0, 0, 0, a, 0, 0, 0, 0,],  # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, 0, 0, 0,],  # +
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, 0, 0,],  # =
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, 0,],  # \n
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b,],  # <pad>
], dtype=float)

# np.random.seed(42)
# W_emb += np.random.uniform(-0.0001, 0.0001, (len(vocabs), emb_dim))

assert W_emb.shape == (len(vocabs), emb_dim)
W_emb.round()

# %%
x = np.array([W_emb[stoi[token]] for token in input])

assert x.shape == (T, emb_dim)
x

# %%
head_count = 1
d_head = emb_dim
d_model = d_head * head_count
d_model

# %%
W_Q = np.zeros((emb_dim, d_model))
assert W_Q.shape == (emb_dim, d_model)

# %%
W_K = np.zeros((emb_dim, d_model))

assert W_K.shape == (emb_dim, d_model)

# %%
W_V = np.array([
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
])

assert W_V.shape == (emb_dim, d_model)

# %%
Q = x @ W_Q
K = x @ W_K
V = x @ W_V

assert Q.shape == (T, d_model)
assert K.shape == (T, d_model)
assert V.shape == (T, d_model)

# %%
import torch

M = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
M = M.masked_fill(torch.isnan(M), 0).numpy()
M


# %%
m = {}

m[5] = [1,              float('-inf'),  float('-inf'),  1,              float('-inf'),  float('-inf')]
m[6] = [1,              float('-inf'),  float('-inf'),  1,              float('-inf'),  float('-inf'),  float('-inf')]
m[7] = [float('-inf'),  1,              float('-inf'),  float('-inf'),  1,              float('-inf'),  float('-inf'),  float('-inf')]

for k, v in m.items():
    M[k] = v + [float('-inf')] * (T - len(v))

M

# %%
QK = (Q @ K.T) / np.sqrt(d_model)
weights = torch.softmax(torch.from_numpy(QK+M), dim=-1)
weights

# %%
Attention = weights @ V
Attention # Caveat: LayerNorm is not applied.

# %%
ff_dim = d_model
# ff_dim = d_model * 4

W_1 = np.array([
    [1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)
b_1 = np.array([0, -9, -10, 0, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0])

assert W_1.shape == (d_model, ff_dim)
assert b_1.shape == (ff_dim,)

W_2 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
], dtype=float)

# Not ordinary bias
b_2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
], dtype=float)
assert W_2.shape == (ff_dim, d_model)
assert b_2.shape == (T, d_model)

FFN = torch.relu(torch.from_numpy(Attention.numpy() @ W_1 + b_1)) @ W_2 + b_2
assert FFN.shape == Attention.shape == (T, d_model)
FFN.round()

# %%
import numpy as np

def sigmoid(x):
    lamb = -10
    bias = 9.5
    return 1. / (1. + np.exp(lamb*(-x+bias)))
# import matplotlib.pyplot as plt
# plt.plot(np.arange(-30, 30, 0.1), sigmoid(np.arange(-30, 30, 0.1)))

# %%
FFN_additional = torch.relu(FFN)*sigmoid(FFN)@ np.array([
    [1.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [1.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [1.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
])

# %%
output = [itos[i.item()] for i in FFN_additional.sum(dim=-1, dtype=int)]
output

# %%
input[:6]

# %%
print(output[-3:])

# %%
# from numpy.linalg import pinv
# from numpy.linalg import inv

# # W_vocab = pinv(W_emb)
# W_vocab = inv(W_emb)

# W_vocab[:, 10:] = 0

# logits = FFN @ W_vocab
# logits

# output = [itos[i.item()] for i in logits.argmax(axis=-1)]
# output


