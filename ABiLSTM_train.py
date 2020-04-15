import time
from ABiLSTM import ABiLSTM
from Wor2Vec import Word2Vec
from Lexicon_Reader import Lex_Reader
import tensorflow as tf
import numpy as np
import pandas as pd

train_data_path = './data/ratings_train_token.txt'
test_data_path = './data/ratings_test_token.txt'
wor2vec_path = './data/Word2vec.model'
Lex_adj_path = './data/Lexicon_adj.csv'
Lex_noun_path = './data/Lexicon_noun.csv'
Lex_verb_path = './data/Lexicon_verb.csv'

config = tf.ConfigProto()
W2V = Word2Vec()
LCR = Lex_Reader()

print("Load tokenized Train data")
train_tokens = W2V.read_token_data(train_data_path)
test_tokens = W2V.read_token_data(test_data_path)
train_tokens = np.array(train_tokens)
test_tokens = np.array(test_tokens)
adj_tokens = LCR.read_data(Lex_adj_path)
noun_tokens = LCR.read_data(Lex_noun_path)
verb_tokens= LCR.read_data(Lex_verb_path)

train_X = train_tokens[:, 0]
train_Y = train_tokens[:, 1]
test_X = test_tokens[:, 0]
test_Y = test_tokens[:, 1]

train_Y_ = W2V.One_hot(train_Y)  ## Convert to One-hot
train_X_ = W2V.Convert2Vec(wor2vec_path, train_X)  ## import word2vec model where you have trained before
test_Y_ = W2V.One_hot(test_Y)  ## Convert to One-hot
test_X_ = W2V.Convert2Vec(wor2vec_path, test_X)  ## import word2vec model where you have trained before

Batch_size = 32
Total_size = len(train_X)
Vector_size = 300
train_seq_length = [len(x) for x in train_X]
test_seq_length = [len(x) for x in test_X]
Maxseq_length = max(train_seq_length)  ## 95
learning_rate = 0.001
lstm_units = 128
num_class = 2
training_epochs = 20
sen_para = 3.0
adj_para = 0.001
total_batch = int(len(train_X) / Batch_size)
test_batch = int(len(test_X) / Batch_size)

print("Train environment Setting: ")
print("Train data set {0} total, Test data set {1} total".format(len(train_X), len(test_X)))
print("Max train sentence length {0}".format(Maxseq_length))
print("vector size: {0}, batch size: {1}, epoch size {2}, learning rate:{3}".format(Vector_size, Batch_size,
                                                                                    training_epochs, learning_rate))
print("Total train batch: {0}, Total test batch: {1}".format(total_batch, test_batch))

#Placeholder
X = tf.placeholder(tf.float32, shape=[None, Maxseq_length, Vector_size], name='X')


Y = tf.placeholder(tf.float32, shape=[None, num_class], name='Y')
seq_len = tf.placeholder(tf.int32, shape=[None])
keep_prob = tf.placeholder(tf.float32, shape=None)

#model ABiLSTM
ABiLSTM = ABiLSTM(lstm_units, num_class, keep_prob, adj_tokens, noun_tokens, verb_tokens)

with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    logits, _ = ABiLSTM.logits(X, seq_len)
    gold = ABiLSTM.gold(sen_para)
    loss, optimizer = ABiLSTM.model_build(logits, Y, gold, adj_para, learning_rate)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

print("Start training!")

modelName = "./LBSA/model"
saver = tf.train.Saver()

train_acc = []
train_loss = []
test_acc = []
test_loss = []

with tf.Session(config=config) as sess:
    start_time = time.time()
    sess.run(init)
    train_writer = tf.summary.FileWriter('./LBSA', sess.graph)
    merged = ABiLSTM.graph_build()

    for epoch in range(training_epochs):

        avg_acc, avg_loss = 0., 0.
        mask = np.random.permutation(len(train_X_))
        train_X_ = train_X_[mask]
        train_Y_ = train_Y_[mask]

        for step in range(total_batch):
            train_batch_X_word = train_X[step*Batch_size: step*Batch_size + Batch_size]
            train_batch_Y_word = train_Y[step*Batch_size: step*Batch_size + Batch_size]
            train_batch_X = train_X_[step * Batch_size: step * Batch_size + Batch_size]
            train_batch_Y = train_Y_[step * Batch_size: step * Batch_size + Batch_size]
            batch_seq_length = train_seq_length[step * Batch_size: step * Batch_size + Batch_size]
            train_batch_X = W2V.Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size)

            _, loss_ = sess.run([optimizer, loss], feed_dict={X: train_batch_X, Y: train_batch_Y,
                                                              seq_len: batch_seq_length, keep_prob: 0.75})

            avg_loss += loss_ / total_batch

            acc = sess.run(accuracy, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length,
                                                keep_prob: 0.75})
            avg_acc += acc / total_batch
            if step % 500 == 0:
                print("[train] epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f}".format(epoch + 1, step + 1,
                                                                                                   loss_, acc))

        summary = sess.run(merged, feed_dict={ABiLSTM.loss: avg_loss, ABiLSTM.acc: avg_acc})
        train_writer.add_summary(summary, epoch)

        t_avg_acc, t_avg_loss = 0., 0.
        print("Test cases, could take few minutes")
        for step in range(test_batch):
            test_batch_X = test_X_[step * Batch_size: step * Batch_size + Batch_size]
            test_batch_Y = test_Y_[step * Batch_size: step * Batch_size + Batch_size]
            batch_seq_length = test_seq_length[step * Batch_size: step * Batch_size + Batch_size]

            test_batch_X = W2V.Zero_padding(test_batch_X, Batch_size, Maxseq_length, Vector_size)

            # Compute average loss
            loss2 = sess.run(loss, feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length,
                                              keep_prob: 1.0})
            t_avg_loss += loss2 / test_batch

            t_acc = sess.run(accuracy, feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length,
                                                  keep_prob: 1.0})
            t_avg_acc += t_acc / test_batch

            if step % 500 == 0:
                print("[test] epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f}".format(epoch + 1, step + 1,
                                                                                                  loss2, t_acc))

        print("<Train> Loss = {:.6f} Accuracy = {:.6f}".format(avg_loss, avg_acc))
        print("<Test> Loss = {:.6f} Accuracy = {:.6f}".format(t_avg_loss, t_avg_acc))
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)
        test_loss.append(t_avg_loss)
        test_acc.append(t_avg_acc)

        #save model
        savemodelName = modelName+str(epoch)+".ckpt"
        saver.save(sess, savemodelName)

    train_loss = pd.DataFrame({"train_loss": train_loss})
    train_acc = pd.DataFrame({"train_acc": train_acc})
    test_loss = pd.DataFrame({"test_loss": test_loss})
    test_acc = pd.DataFrame({"test_acc": test_acc})
    df = pd.concat([train_loss, train_acc, test_loss, test_acc], axis=1)
    df.to_csv("./LBSA/loss_accuracy.csv", sep=",", index=False)

    train_writer.close()
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute, second))
