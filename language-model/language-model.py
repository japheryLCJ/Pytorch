import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

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

it = iter(train_iter)
batch = next(it)#batch中包含.text和.target 都是50*32的torch，50是句子长度，32是batch长度，内容是单词的序号
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:, 1].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:, 1].data]))
# for i in range(5):
#     batch = next(it)
#     print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:, 1].data]))
#     print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:, 1].data]))
