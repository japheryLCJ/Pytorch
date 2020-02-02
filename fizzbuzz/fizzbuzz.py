import numpy as np
import torch


# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


print(fizz_buzz_decode(1, fizz_buzz_encode(1)))
print(fizz_buzz_decode(2, fizz_buzz_encode(2)))
print(fizz_buzz_decode(5, fizz_buzz_encode(5)))
print(fizz_buzz_decode(12, fizz_buzz_encode(12)))
print(fizz_buzz_decode(15, fizz_buzz_encode(15)))

NUM_DIGITS = 10


# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# Define the model
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Start training it
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Find loss on training data
    loss = loss_fn(model(trX), trY).item()
    print('Epoch:', epoch, 'Loss:', loss)

    # Output now
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)
predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))

print([fizz_buzz_decode(i, x) for (i, x) in predictions])
# print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1, 101)])))
# testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1, 101)])
