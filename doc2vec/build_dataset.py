from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

tqdm.pandas()

def the_corpus(nchunks = 10):

    for chunkid in range(nchunks):
        # load data chunk
        for i,doc in pd.read_hdf("data/tagged.hdf",key="chunk_%s" % chunkid).iteritems():
            yield doc

if __name__ == "__main__":

    texts = []
    tags = []
    for doc in the_corpus():
        texts.append(' '.join(doc.words))
        tags.append(doc.tags)

    lb = MultiLabelBinarizer(sparse_output=True)
    tags_mat = lb.fit_transform(pd.Series(tags))
    
    top_tags = np.array(tags_mat.sum(axis=0).argsort())[0,-100:]

    mask = np.array(tags_mat[:,top_tags].sum(axis=1)>0)[:,0]

    df_questions = pd.DataFrame(np.array([texts,tags])[:,mask].T)

    df_questions.columns = ["text", "tags"]

    # Save questions
    print("save ",str(df_questions.shape)," questions")
    df_questions.to_hdf("data/dataset_questions_test.h5", key="data")

    # Encode questions
    print("encode questions")
    model= Doc2Vec.load("data/results/test.model")
    df_embeddings = df_questions["text"].progress_apply(lambda x: pd.Series(model.infer_vector(x.split(' '))))
    np.save("data/dataset_embeddings_test.npy",df_embeddings.values)
