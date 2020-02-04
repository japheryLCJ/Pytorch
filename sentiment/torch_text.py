# TorchText中的一个重要概念是Field。Field决定了你的数据会被怎样处理。在我们的情感分类任务中，我们所需要接触到的数据有文本字符串和两种情感，"pos"或者"neg"。
# Field的参数制定了数据会被怎样处理。
# 我们使用TEXT field来定义如何处理电影评论，使用LABEL field来处理两个情感类别。
# 我们的TEXT field带有tokenize='spacy'，这表示我们会用spaCy tokenizer来tokenize英文句子。如果我们不特别声明tokenize这个参数，那么默认的分词方法是使用空格。
# 安装spaCy
# pip install -U spacy
# python -m spacy download en
# LABEL由LabelField定义。这是一种特别的用来处理label的Field。我们后面会解释dtype。
# 更多关于Fields，参见https://github.com/pytorch/text/blob/master/torchtext/data/field.py
# 和之前一样，我们会设定random seeds使实验可以复现。
import torch
import torchtext
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField()# dtype=torch.float)
# print(torch.__version__)
# print(torchtext.__version__)
# 1.1.0
# 0.3.1
# TorchText支持很多常见的自然语言处理数据集。
# 下面的代码会自动下载IMDb数据集，然后分成train/test两个torchtext.datasets类别。数据被前面的Fields处理。IMDb数据集一共有50000电影评论，每个评论都被标注为正面的或负面的。
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# 查看每个数据split有多少条数据。

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')
# Number of training examples: 25000
# Number of testing examples: 25000
# 查看一个example。
# print(vars(train_data.examples[0]))
# 由于我们现在只有train/test这两个分类，所以我们需要创建一个新的validation set。我们可以使用.split()创建新的分类。
# 默认的数据分割是 70、30，如果我们声明split_ratio，可以改变split之间的比例，split_ratio=0.8表示80%的数据是训练集，20%是验证集。
# 我们还声明random_state这个参数，确保我们每次分割的数据集都是一样的。
import random
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
# print(f'Number of training examples: {len(train_data)}')
# print(f'Number of validation examples: {len(valid_data)}')
# print(f'Number of testing examples: {len(test_data)}')
# Number of training examples: 17500
# Number of validation examples: 7500
# Number of testing examples: 25000
# 下一步我们需要创建 vocabulary 。vocabulary 就是把每个单词一一映射到一个数字。
# 我们使用最常见的25k个单词来构建我们的单词表，用max_size这个参数可以做到这一点。
# 所有其他的单词都用<unk>来表示。
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)#使用glove向量作为初始向量
LABEL.build_vocab(train_data)
# 当我们把句子传进模型的时候，我们是按照一个个 batch 穿进去的，也就是说，我们一次传入了好几个句子，而且每个batch中的句子必须是相同的长度。为了确保句子的长度相同，TorchText会把短的句子pad到和最长的句子等长。
# 下面我们来看看训练数据集中最常见的单词。
# print(TEXT.vocab.freqs.most_common(20))
# [('the', 201455), (',', 192552), ('.', 164402), ('a', 108963), ('and', 108649), ('of', 100010), ('to', 92873), ('is', 76046), ('in', 60904), ('I', 54486), ('it', 53405), ('that', 49155), ('"', 43890), ("'s", 43151), ('this', 42454), ('-', 36769), ('/><br', 35511), ('was', 34990), ('as', 30324), ('with', 29691)]
# 我们可以直接用 stoi(string to int) 或者 itos (int to string) 来查看我们的单词表。
# print(TEXT.vocab.itos[:10])
# ['<unk>', '<pad>', 'the', ',', '.', 'a', 'and', 'of', 'to', 'is']
# 查看labels。
# print(LABEL.vocab.stoi)
# defaultdict(<function _default_unk_index at 0x7f6944d3f730>, {'neg': 0, 'pos': 1})
# 最后一步数据的准备是创建iterators。每个itartion都会返回一个batch的examples。
# 我们会使用BucketIterator。BucketIterator会把长度差不多的句子放到同一个batch中，确保每个batch中不出现太多的padding。
# 严格来说，我们这份notebook中的模型代码都有一个问题，也就是我们把<pad>也当做了模型的输入进行训练。更好的做法是在模型中把由<pad>产生的输出给消除掉。在这节课中我们简单处理，直接把<pad>也用作模型输入了。由于<pad>数量不多，模型的效果也不差。
# 如果我们有GPU，还可以指定每个iteration返回的tensor都在GPU上
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    repeat=False)
# for i, _ in enumerate(train_iterator):
#     print(i)
batch = next(iter(train_iterator))
# batch.text
# Out[39]:
# tensor([[   65,  6706,    23,  ...,  3101,    54,    87],
#         [   52, 11017,    83,  ..., 24113,    15,  1078],
#         [    8,     3,   671,  ...,    52,    73,     3],
#         ...,
#         [    1,     1,     1,  ...,     1,     1,     1],
#         [    1,     1,     1,  ...,     1,     1,     1],
#         [    1,     1,     1,  ...,     1,     1,     1]], device='cuda:0')

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
mask = batch.text == PAD_IDX
# mask
# Out[44]:
# tensor([[0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         ...,
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1]], device='cuda:0', dtype=torch.uint8)