from gensim.models.doc2vec import Doc2Vec
import pandas as pd

if __name__ == "__main__":

    model= Doc2Vec.load("data/results/d2v.model")

    #to find the vector of a document which is not in training data
    train_data = pd.read_hdf("data/tagged.hdf", key="chunk_0")
    print(train_data.iloc[0])
    v1 = model.infer_vector(train_data.iloc[0].words)
    print("V1_infer", v1)

    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar('c++')
    print(similar_doc)


    # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    print(model.docvecs['php'])