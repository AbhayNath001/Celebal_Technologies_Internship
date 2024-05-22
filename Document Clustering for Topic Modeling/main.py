# Created by Abhay Nath (CT_CSI_DS_511)
import os
import glob
import tarfile
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"
urllib.request.urlretrieve(url, '20_newsgroups.tar.gz')

with tarfile.open('20_newsgroups.tar.gz', 'r:gz') as tar:
    tar.extractall()

def load_data(path):
    documents = []
    labels = []
    for label in os.listdir(path):
        class_path = os.path.join(path, label)
        if os.path.isdir(class_path):
            for file_path in glob.glob(os.path.join(class_path, '*')):
                with open(file_path, 'r', encoding='latin1', errors='ignore') as file:
                    documents.append(file.read())
                    labels.append(label)
    return documents, labels

documents, labels = load_data('20_newsgroups')

nltk.download('stopwords')
stop_words = stopwords.words('english')

vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.5, max_features=10000)
X = vectorizer.fit_transform(documents)

num_clusters = 20
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(X)
clusters = km.labels_

def print_top_terms_per_cluster(vectorizer, km, num_terms=10):
    terms = vectorizer.get_feature_names_out()
    for i in range(num_clusters):
        print(f"Cluster {i}:")
        cluster_terms = km.cluster_centers_[i].argsort()[-num_terms:]
        print(" ".join(terms[cluster_terms]))

print_top_terms_per_cluster(vectorizer, km)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
plt.colorbar(scatter, ticks=range(num_clusters))
plt.title('Visualization of 20 Newsgroups clusters')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()