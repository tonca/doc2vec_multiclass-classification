import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec


def tsne_plot(vectors, labels):
    "Creates and TSNE model and plots it"

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    labelset = list(set(labels))
    colors = ["r","g","b"]
    color_tags = [colors[labelset.index(label)] for label in labels]
    plt.figure(figsize=(10, 10)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i],c=color_tags[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig("data/results/tsne_php-python_trained.png")


if __name__ == "__main__":
    
    model= Doc2Vec.load("data/results/d2v.model")

    #to find the vector of a document which is not in training data
    train_data = pd.read_hdf("data/tagged.hdf", key="chunk_10")
    
    selected = ["php","python"]
    vectors = []
    tags = []
    for doc in train_data[:1000].tolist():
        for tag in selected:
            if tag in doc.tags:
                vectors.append(model.infer_vector(doc.words))
                tags.append(tag)

    tsne_plot(vectors,tags)
