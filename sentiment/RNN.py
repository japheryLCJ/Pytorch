import torch
import torchtext
from torchtext import data
import torch.optim as optim
import time

from WordAveraging import train, evaluate, criterion, epoch_time

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField()# dtype=torch.float)
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
import random
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)#使用glove向量作为初始向量
LABEL.build_vocab(train_data)
BATCH_SIZE = 64
device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    repeat=False)
batch = next(iter(train_iterator))
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
mask = batch.text == PAD_IDX
# 我们首先介绍一个简单的Word Averaging模型。这个模型非常简单，我们把每个单词都通过Embedding层投射成word embedding vector，然后把一句话中的所有word vector做个平均，就是整个句子的vector表示了。接下来把这个sentence vector传入一个Linear层，做分类即可。
# 我们使用avg_pool2d来做average pooling。我们的目标是把sentence length那个维度平均成1，然后保留embedding这个维度。
# avg_pool2d的kernel size是 (embedded.shape[1], 1)，所以句子长度的那个维度会被压扁。
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx, avg_hidden=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2 if self.bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.avg_hidden = avg_hidden

    def forward(self, text, mask):
        embedded = self.dropout(self.embedding(text))  # [sent len, batch size, emb dim]
        # CHANGED
        seq_length = mask.sum(1)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(
            input=embedded,
            lengths=seq_length,
            batch_first=True,
            enforce_sorted=False)
        output, (hidden, cell) = self.rnn(embedded)
        output, seq_length = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=output,
            batch_first=True,
            padding_value=0,
            total_length=mask.shape[1]
        )
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        if self.avg_hidden:
            #             print(output)
            hidden = torch.sum(output * mask.unsqueeze(2), 1) / torch.sum(mask, 1,
                                                                          keepdim=True)  # [batch size, embedding_dim]
        else:
            if self.bidirectional:
                # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                                   dim=1)  # [batch size, hid dim * num directions]
            else:
                hidden = self.dropout(hidden[-1, :, :])  # [batch size, hid dim * num directions]
        # apply dropout
        hidden = self.dropout(hidden)
        return self.fc(hidden)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX, avg_hidden=False)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters())
model = model.to(device)

N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'lstm-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
# 我们来尝试一个AVG的版本
#
# INPUT_DIM = len(TEXT.vocab)
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256
# OUTPUT_DIM = 1
# N_LAYERS = 2
# BIDIRECTIONAL = True
# DROPOUT = 0.5
# PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
#
# rnn_model_avg = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
#             N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
#
# print(f'The model has {count_parameters(rnn_model_avg):,} trainable parameters')
# The model has 4,810,857 trainable parameters
#
# rnn_model_avg.embedding.weight.data.copy_(pretrained_embeddings)
# UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
#
# rnn_model_avg.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# rnn_model_avg.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
#
# print(rnn_model_avg.embedding.weight.data)
# tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#         [-0.0382, -0.2449,  0.7281,  ..., -0.1459,  0.8278,  0.2706],
#         ...,
#         [-0.7244, -0.0186,  0.0996,  ...,  0.0045, -1.0037,  0.6646],
#         [-1.1243,  1.2040, -0.6489,  ..., -0.7526,  0.5711,  1.0081],
#         [ 0.0860,  0.1367,  0.0321,  ..., -0.5542, -0.4557, -0.0382]])
# optimizer = optim.Adam(rnn_model_avg.parameters())
# rnn_model_avg = rnn_model_avg.to(device)
# 
# N_EPOCHS = 5
# best_valid_loss = float('inf')
# for epoch in range(N_EPOCHS):
#     start_time = time.time()
#     train_loss, train_acc = train(rnn_model_avg, train_iterator, optimizer, criterion)
#     valid_loss, valid_acc = evaluate(rnn_model_avg, valid_iterator, criterion)
#
#     end_time = time.time()
#
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(rnn_model_avg.state_dict(), 'lstm-avg-model.pt')
#
#     print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')