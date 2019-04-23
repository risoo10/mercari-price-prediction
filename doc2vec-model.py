from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import numpy as np
import pandas as pd
import swifter


class Doc2VecModel():
    def __init__(self, filename):
        self.filename = filename
        self.model = None
        self.window = 5

    def fit_transform(self, documents, epochs=30, start_alpha=0.025, min_alpha=0.008, vector_size=32):
        print(f'Fit transform {documents.shape} documents for {epochs} epochs .....')
        self.model = Doc2Vec(alpha=start_alpha, min_alpha=start_alpha, vector_size=vector_size, window=self.window, workers=4)
        self.model.build_vocab(documents)

        step = (start_alpha - min_alpha) / epochs

        for epoch in range(epochs):
            self.model.train(documents, total_examples=self.model.corpus_count, epochs=1)
            self.model.alpha -= step  # decrease the learning rate
            self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay
            print(f'Epoch {epoch} finished. Alpha: {self.model.alpha}')

        doc_df = pd.DataFrame(np.arange(len(documents)), columns='tags')
        doc_df['vectors'] = doc_df['tags'].swifter.apply(lambda x: self.model.docvecs[str(x)])

        # Stop training
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.model.save(self.filename)

        return doc_df['vectors']

    def transform(self, words, steps=20, alpha=0.025):
        return self.model.infer_vector(doc_words=words, steps=steps, alpha=alpha)

    def load_from_file(self):
        self.model = Doc2Vec.load(self.filename)
