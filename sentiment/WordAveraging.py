import torch
import torchtext
from torchtext import data

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


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text, mask):
        embedded = self.embedding(text)  # [batch_size, seq_len, emb_dim]
        sent_embed = torch.sum(embedded * mask.unsqueeze(2), 1) / mask.sum(1).unsqueeze(
            1)  # [batch size, embedding_dim]
        return self.fc(sent_embed)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1

model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)
# 计算预测的准确率

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    i = 0
    for batch in iterator:
        optimizer.zero_grad()
        # CHANGED
        text = batch.text.permute(1, 0)  # [batch_size, seq_length]
        mask = 1. - (text == PAD_IDX).float()  # [batch_size, seq_len]
        predictions = model(text, mask).squeeze(1)
        loss = criterion(predictions, batch.label.float())
        acc = binary_accuracy(predictions, batch.label.float())
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("batch {}, loss {}".format(i, loss.item()))
        i += 1

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    i = 0
    with torch.no_grad():
        for batch in iterator:
            text = batch.text.permute(1, 0)  # [batch_size, seq_length]
            mask = 1. - (text == PAD_IDX).float()  # [batch_size, seq_len]
            predictions = model(text, mask).squeeze(1)
            loss = criterion(predictions, batch.label.float())
            acc = binary_accuracy(predictions, batch.label.float())

            if i % 100 == 0:
                print("batch {}, loss {}".format(i, loss.item()))
            i += 1
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
import spacy

nlp = spacy.load('en')


def predict_sentiment(sentence):
      tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
      indexed = [TEXT.vocab.stoi[t] for t in tokenized]
      tensor = torch.LongTensor(indexed).to(device)
      text = tensor.unsqueeze(0)
      mask = 1. - (text == PAD_IDX).float()  # [batch_size, seq_len]
      prediction = torch.sigmoid(model(tensor, mask))
      return prediction.item()
# predict_sentiment("This film is terrible")
#
# 2.4536811471520537e-10
#
# predict_sentiment("This film is great")
#
# 1.0