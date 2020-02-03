import torch
import torch.nn as nn

import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA = False #torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 50000
# 我们会继续使用上次的text8作为我们的训练，验证和测试数据
# # TorchText的一个重要概念是Field，它决定了你的数据会如何被处理。我们使用TEXT这个field来处理文本数据。我们的TEXT field有lower=True这个参数，所以所有的单词都会被lowercase。
# # torchtext提供了LanguageModelingDataset这个class来帮助我们处理语言模型数据集。
# # build_vocab可以根据我们提供的训练数据集来创建最高频单词的单词表，max_size帮助我们限定单词总量。
# # BPTTIterator可以连续地得到连贯的句子，BPTT的全程是back propagation through time。

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".",
                                                                     train="text8.train.txt",
                                                                     validation="text8.dev.txt", test="text8.test.txt",
                                                                     text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=50, repeat=False, shuffle=True)
# vocabulary size: 50002
# 为什么我们的单词表有50002个单词而不是50000呢？因为TorchText给我们增加了两个特殊的token，<unk>表示未知的单词，<pad>表示padding。
# 模型的输入是一串文字，模型的输出也是一串文字，他们之间相差一个位置，因为语言模型的目标是根据之前的单词预测下一个单词。









class RNNModel(nn.Module):
    """ 一个简单的循环神经网络"""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层 ntoken为输入词库数，ninp即嵌入层的维数 嵌入结果为 VOCAB_SIZE*EMBEDDING_SIZE的向量
            - 一个循环神经网络层(RNN, LSTM, GRU) 当为多层时用到nlayers
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        emb = self.drop(self.encoder(input))#input为text（seq_len*batch）输出为（seq_len*batch*embed_size）
        output, hidden = self.rnn(emb, hidden)#LSTM的输入形式为（seq_len*batch*embed_size）,hidden即为LSTM的初始传递参数h0和c0
                                              # output的形式为seq_len * batch * hidden_size，而hidden为（1 * batch * hidde_size，1 * batch * hidden_size）
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))#进行线性变换时要把output的3个维度压缩到2维，把seq_len*batch压缩到一起
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden#线性变化后在把前两个维度转换回到seq_len*batch的形式

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad)


model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 1, dropout=0.5)
if USE_CUDA:
    model = model.cuda()


# 我们首先定义评估模型的代码。
# 模型的评估和模型的训练逻辑基本相同，唯一的区别是我们只需要forward pass，不需要backward pass
def evaluate(model, data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item() * np.multiply(*data.size())

    loss = total_loss / total_count
    model.train()
    return loss


# 我们需要定义下面的一个function，帮助我们把一个hidden state和计算图之前的历史分离。

# Remove this part
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# 定义loss function和optimizer


loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

# 训练模型：
#
# 模型一般需要训练若干个epoch
# 每个epoch我们都把所有的数据分成若干个batch
# 把每个batch的输入和输出都包装成cuda tensor
# forward pass，通过输入的句子预测每个单词的下一个单词
# 用模型的预测和正确的下一个单词计算cross entropy loss
# 清空模型当前gradient
# backward pass
# gradient clipping，防止梯度爆炸
# 更新模型参数
# 每隔一定的iteration输出模型在当前iteration的loss，以及在验证集上做模型的评估

import copy

GRAD_CLIP = 1. #梯度爆炸限制
NUM_EPOCHS = 2

val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 1000 == 0:
            print("epoch", epoch, "iter", i, "loss", loss.item())

        if i % 10000 == 0:
            val_loss = evaluate(model, val_iter)

            if len(val_losses) == 0 or val_loss < min(val_losses):
                print("best model, val loss: ", val_loss)
                torch.save(model.state_dict(), "lm-best.th")
            else:
                scheduler.step()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            val_losses.append(val_loss)

best_model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
if USE_CUDA:
    best_model = best_model.cuda()
best_model.load_state_dict(torch.load("lm-best.th"))
# 使用最好的模型在valid数据上计算perplexity

val_loss = evaluate(best_model, val_iter)
print("perplexity: ", np.exp(val_loss))

# 使用最好的模型在测试数据上计算perplexity
test_loss = evaluate(best_model, test_iter)
print("perplexity: ", np.exp(test_loss))

# 使用训练好的模型生成一些句子。


hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    word_weights = output.squeeze().exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))
