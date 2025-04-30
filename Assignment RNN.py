import torch
import torch.nn as nn
import torch.optim as optim

data = [
    "the cat is fluffy",
    "dog runs very fast",
    "bird flies so high",
    "fish swims in water"
]

words = set()
for sequence in data:
    for word in sequence.split():
        words.add(word)
word_to_idx = {word: idx for idx, word in enumerate(words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(words)

inputs = []
targets = []
for sequence in data:
    seq_words = sequence.split()
    input_indices = [word_to_idx[word] for word in seq_words[:3]]  # First 3 words
    target_index = word_to_idx[seq_words[3]]  # 4th word
    inputs.append(input_indices)
    targets.append(target_index)

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)  
        output, hidden = self.rnn(embedded)  
        last_output = output[:, -1, :]  
        return self.fc(last_output)

embedding_dim = 10
hidden_dim = 20
learning_rate = 0.01
epochs = 100

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

model.eval()
test_sequence = torch.tensor([[word_to_idx["the"], word_to_idx["cat"], word_to_idx["is"]]], dtype=torch.long)
with torch.no_grad():
    pred = model(test_sequence)
    pred_idx = torch.argmax(pred, dim=1).item()
    print(f'Predicted 4th word: {idx_to_word[pred_idx]}')