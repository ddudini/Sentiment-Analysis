import tensorflow as tf

class ABiLSTM(object):
    def __init__(self, lstm_units, num_class, keep_prob, adj_tokens, noun_tokens, verb_tokens):
        self.lstm_units = lstm_units
        self.num_class = num_class
        self.lex_adj = adj_tokens
        self.lex_noun = noun_tokens
        self.lex_verb = verb_tokens

        with tf.variable_scope('forwoard', reuse=tf.AUTO_REUSE):
            self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, state_is_tuple=True)
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=keep_prob)

        with tf.variable_scope('backward', reuse=tf.AUTO_REUSE):
            self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob=keep_prob)

        # Attention layer weights
        with tf.variable_scope('Atteintion_Weight', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable(name="Wfb", shape=[2*lstm_units, num_class],
                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="bias", shape=[num_class],
                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.V = tf.get_variable(name="V2", shape=[num_class, num_class],
                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    def logits(self, X, seq_len):
        inputs, _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell, dtype=tf.float32, inputs=X,
                                                    sequence_length=seq_len)
        num_class = self.num_class
        inputs = tf.concat([inputs[0], inputs[1]], axis=2)
        hidden = tf.layers.dense(inputs, units = 2 * self.lstm_units, activation = tf.nn.tanh,
                                 kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))
        attention_weight =tf.layers.dense(hidden, units=1, activation=None,
                                          kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))
        attention_weight = tf.transpose(tf.nn.softmax(tf.transpose(attention_weight, perm=[0, 2, 1])), perm=[0, 2, 1])
        outputs = tf.reduce_sum(inputs*attention_weight, axis=1)
        logits = tf.layers.dense(outputs, units=num_class, activation = None)

        return logits, attention_weight

    def gold(self, sent_deg):
        return 0

    def model_build(self, logits, labels, gold, adjust, learning_rate=0.001):
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            loss = loss + (adjust * gold)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return loss, optimizer

    def graph_build(self):
        self.loss = tf.placeholder(tf.float32)
        self.acc = tf.placeholder(tf.float32)
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', self.acc)
        merged = tf.summary.merge_all()

        return merged
