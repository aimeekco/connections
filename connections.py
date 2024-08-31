from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

words = ["apple", "banana", "grape", "orange", "cat", "dog", "fish", "bird", "car", "truck", "bus", "boat", "violin", "guitar", "piano", "drum"]

embeddings = torch.vstack([get_word_embedding(word) for word in words])

clustering = AgglomerativeClustering(n_clusters=4)
clustering.fit(embeddings.detach().numpy())

for idx in range(4):
    print(f"Cluster {idx}: {[words[i] for i in range(len(words)) if clustering.labels_[i] == idx]}")