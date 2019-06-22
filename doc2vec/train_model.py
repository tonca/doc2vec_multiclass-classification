#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import time
import os

def the_corpus(nchunks = 10):

    for chunkid in range(10,nchunks+10):
        # load data chunk
        for i,doc in pd.read_hdf("data/tagged.hdf",key="chunk_%s" % chunkid).iteritems():
            yield doc

if __name__=="__main__":
        
    vec_size = 20
    alpha = 0.025

    model_file = "data/results/test.model"
    if os.path.isfile(model_file):
        model = Doc2Vec.load(model_file)
    else:   
        model = Doc2Vec(vector_size=vec_size,
            alpha=alpha, 
            min_alpha=0.00025,
            min_count=1,
            dm =1)
    
        model.build_vocab(the_corpus())
        print("Built the model!")
        
    tic = time.clock()

    print("Train!")
    model.train(the_corpus(),
        total_examples=model.corpus_count,
        epochs=20)

    print(model)

    toc = time.clock()
    print("Elapsed time: %s s" %(toc - tic))

    model.save(model_file)
    print("model saved")

    

    print('total docs learned %s' % (len(model.docvecs)))