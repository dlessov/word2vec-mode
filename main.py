import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.cm as cm


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding_matrix = (torch.randn(vocab_size, embedding_dim))
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding_matrix[x]
        summed = torch.sum(embedded, dim=1)
        out = self.linear(summed)
        return out


with open("dataset_small.txt") as f:
    corpus = f.read().splitlines()

words = [sentence.lower().split() for sentence in corpus]
vocab = list(set([word for sentence in words for word in sentence]))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}


window_size = 2
X, y = [], []
for sentence in words:
    sentence_indices = [word2idx[word] for word in sentence]
    for i, target in enumerate(sentence_indices):
        context = [sentence_indices[j] for j in range(i - window_size, i + window_size + 1) if j != i and 0 <= j < len(sentence)]
        X.append([0] * (2 * window_size - len(context)) + context )
        y.append(target)
print(X)
print(y)

X = torch.tensor(X)
y = torch.tensor(y)

vocab_size = len(vocab)
embedding_dim = 10
lr = 0.001
epochs = 100

model = CBOW(vocab_size, embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_function(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


embeddings = model.embedding_matrix.data.numpy()

num_clusters = 5
n_init = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=n_init)
clusters = kmeans.fit_predict(embeddings)

with torch.no_grad():
    model.eval()
    outputs = model(X)
    predicted_labels = torch.argmax(outputs, dim=1)
    accuracy = (predicted_labels != y).sum().item() / len(y)
    print(f"Accuracy: {accuracy}")



tsne = TSNE(n_components=2)
embeddings_tsne = tsne.fit_transform(embeddings)


fig, ax = plt.subplots(figsize=(10, 10))


color_map = cm.get_cmap('tab10', num_clusters)


for idx, word in idx2word.items():
    x, y = embeddings_tsne[idx]
    cluster_label = clusters[idx]
    color = color_map(cluster_label) 
    ax.scatter(x, y, c=[color])
    ax.annotate(word, (x, y), alpha=0.7)
plt.show()