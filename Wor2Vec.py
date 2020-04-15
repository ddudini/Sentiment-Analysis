from konlpy.tag import Okt
import gensim
import numpy as np
import config

class Word2Vec():

    def __init__(self, adj_tokens_path, noun_tokens_path, verb_tokens_path):
        self.adj_tokens = self.read_lexidon_data(adj_tokens_path)
        self.noun_tokens = self.read_lexidon_data(noun_tokens_path)
        self.verb_tokens = self.read_lexidon_data(verb_tokens_path)


    def tokenize(self, doc):
        pos_tagger = Okt()
        return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        return data

    def read_token_data(self, filepath):
        data = self.read_data(filepath)
        re_data = [[r[0].split(';'), r[1]] for r in data]
        return re_data

    def Word2vec_model(self, model_name):
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        return model

    def Convert2Vec(self, model_name, doc):
        ## Convert corpus into vectors
        word_vec = []
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        for sent in doc:
            sub = []
            for word in sent:
                if (word in model.wv.vocab):
                    sub.append(model.wv[word])
                else:
                    sub.append(np.random.uniform(-0.25, 0.25, 300))  ## used for OOV words
            word_vec.append(sub)

        ## Adding lexicon value
        for idx, sent in enumerate(doc):
            sub = []
            for word in sent:
                pos = word.split('/')[1]
                if pos == 'Noun' and word in self.noun_tokens:
                    sub.append(self.noun_tokens[word])
                elif pos == 'Adjective' and word in self.adj_tokens:
                    sub.append(self.adj_tokens[word])
                elif pos == 'Verb' and word in self.verb_tokens:
                    sub.append(self.verb_tokens[word])
                else:
                    sub.append(0)
            word_vec[idx].append(sub)

        return np.array(word_vec)

    def Zero_padding(self, train_batch_X, Batch_size, Maxseq_length, Vector_size):
        zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
        for i in range(Batch_size):
            zero_pad[i, :np.shape(train_batch_X[i])[0], :np.shape(train_batch_X[i])[1]] = train_batch_X[i]

        return zero_pad

    def One_hot(self, data):

        index_dict = {value: index for index, value in enumerate(set(data))}
        result = []

        for value in data:
            one_hot = np.zeros(len(index_dict))
            index = index_dict[value]
            one_hot[index] = 1
            result.append(one_hot)

        return np.array(result)

    def read_lexidon_data(self, filename):
        print('Making Lexicon in', filename)
        dict = {}
        f = open(filename, 'r', encoding='utf-8')

        while True:
            line = f.readline()
            if not line: break
            tokens = line.split('\t')
            dict[tokens[0]] = float(tokens[2])
        f.close()

        print('Lexicon building Done')

        return dict

if __name__ == '__main__':
    W2V = Word2Vec(config.lexicon_adj, config.lexicon_noun, config.lexicon_verb)
    test_data = W2V.read_token_data(config.test_data_token_path)

    test_data_ = W2V.Convert2Vec(config.wor2vec_path, test_data)
    print(type(test_data_))
