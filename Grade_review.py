
import os
import tensorflow as tf
from BiLstm.Bi_LSTM import Bi_LSTM
import word2vec.wor2vec as W2V
import gensim
import numpy as np
import config
from LBSA.ABiLSTM import ABiLSTM as LBSA
from LBSA.Lexicon_Reader import Lex_Reader

LCR = Lex_Reader()
Batch_size = 1
Vector_size = 300
Maxseq_length = 95  ## Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 2
keep_prob = 1.0
sen_para = 3.0
adj_para = 0.001

print("Loading Lexicons....")
adj_tokens = LCR.read_data(config.lexicon_adj)
noun_tokens = LCR.read_data(config.lexicon_noun)
verb_tokens= LCR.read_data(config.lexicon_verb)

####Restore BiLSTM####
print("Loading BiLSTM model....")
X = tf.placeholder(tf.float32, shape=[None, Maxseq_length, Vector_size], name='X')
Y = tf.placeholder(tf.float32, shape=[None, num_class], name='Y')
seq_len = tf.placeholder(tf.int32, shape=[None])

BiLSTM = Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
#modelName = "./BiLSTM_model.ckpt"
modelName = config.BiLSTM_model

sess = tf.Session()
sess.run(init)
saver.restore(sess, modelName)

os.chdir("..")

#####LBSA########
print("Loading Supervised Attention BiLSTM Model....")
X_ = tf.placeholder(tf.float32, shape=[None, Maxseq_length, Vector_size], name='X')
Word_X = tf.placeholder(tf.float32, shape=None, name='Word_X')
Y_ = tf.placeholder(tf.float32, shape=[None, num_class], name='Y')
lbsa_seq_len = tf.placeholder(tf.int32, shape=[None])

lbsa = LBSA(lstm_units,num_class, keep_prob)

with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    lbsa_logits, learned = lbsa.logits(X, seq_len, Maxseq_length)
    gold = lbsa.gold(Word_X, sen_para)
    lbsa_loss, lbsa_optimizer = lbsa.model_build(logits, learned, gold, Y, adj_para, learning_rate)

lbsa_prediction = tf.nn.softmax(lbsa_logits)
correct_pred = tf.equal(tf.argmax(lbsa_prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

lbsa_saver = tf.train.Saver()
lbsa_init = tf.global_variables_initializer()

lbsa_sess = tf.Session()
lbsa_sess.run(lbsa_init)
lbsa_saver.restore(lbsa_sess, config.LBST_model)

os.chdir("..")

def LBSAGrade(sentence, model_name):
    tokens = W2V.tokenize(sentence)
    embedding = W2V.Convert2Vec(config.wor2vec_path, tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)

    word_x = []
    for element in tokens:
        tags = element.split('/')
        if len(tags) == 2:
            voca = tags[0]
            pos = tags[1]
            lexicon = {}
            if pos == 'Noun':
                lexicon = noun_tokens
            elif pos == 'Adjective':
                lexicon = adj_tokens
            elif pos == 'Verb':
                lexicon = verb_tokens

            if lexicon.get(voca):
                score = abs(lexicon.get(voca))
            else:
                score = 0
        else:
            score = 0
        word_x.append(score)

    global lbsa_sess
    result = lbsa_sess.run(tf.argmax(prediction, 1), feed_dict={X_: zero_pad, Word_X: word_x, lbsa_seq_len: [len(tokens)]})
    if (result == 0):
        print(model_name + ' - ' + W2V.printRe("긍정") + "\n")
    else:
        print(model_name + ' - ' + W2V.printRe("부정") + "\n")

def BiLSTMGrade(sentence, model_name):
    tokens = W2V.tokenize(sentence)
    embedding = W2V.Convert2Vec(config.wor2vec_path, tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
    global sess
    result = sess.run(tf.argmax(prediction, 1), feed_dict={X: zero_pad, seq_len: [len(tokens)]})

    if (result == 0):
        print(model_name + ' - ' + W2V.printRe("긍정") + "\n")
    else:
        print(model_name + ' - ' + W2V.printRe("부정") + "\n")


with open(config.review, 'r') as f:
    sentence = f.read().splitlines()
    for sent in sentence:
        se, re = sent.split('=')
        print('['+ W2V.printRe(re)+'] ' + se)
        BiLSTMGrade(se, 'BiLSTM')
        LBSAGrade(sentence, 'LBSA')