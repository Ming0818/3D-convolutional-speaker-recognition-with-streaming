import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from colorama import Fore, Style
import time

# eager execution
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

# Hyper parameters
LEARNING_RATE = 0.001


class LSTMDvector(tf.keras.Model):
    """
    input : 98 * (dynamic length /maximum 5) * 40
    out : 0.66 * total speaker[depends on your dataset] =  (trained-speaker)
    """

    def __init__(self, input_dim, out_dim, checkpoint_directory, device_name="cpu:0"):
        super(LSTMDvector, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.checkpoint_directory = checkpoint_directory
        self.device_name = device_name

        # lstm cell
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(256)

        # dense
        from tensorflow.python.ops import init_ops

        self.dense1 = tf.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init_ops.random_uniform_initializer())
        self.batch1 = tf.layers.BatchNormalization()
        self.dense2 = tf.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=init_ops.random_uniform_initializer())
        self.batch2 = tf.layers.BatchNormalization()

        # dvector
        self.dvector = tf.layers.Dense(512, kernel_initializer=init_ops.random_uniform_initializer())

        self.trainprob = tf.layers.Dense(out_dim, activation=tf.nn.softmax, kernel_initializer=init_ops.random_uniform_initializer())

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        self.time = time.time()
        self.total_step = 0


    def __call__(self, X, seq_length, verbose=0, steps=None, training=False):
        return self.predict(X, seq_length, verbose=0, steps=None, training=False)

    def predict(self, X, seq_length, verbose=0, steps=None, training=False):
        # Get the number of samples within a batch
        num_samples = tf.shape(X)[0]

        # Initialize LSTM cell state with zeros
        state = self.rnn_cell.zero_state(num_samples, dtype=tf.float32)

        # Unstack
        unstacked = tf.unstack(X, axis=1)
        # Iterate through each timestep and append the predictions
        outputs = []
        for input_step in unstacked:
            output, state = self.rnn_cell(input_step, state)
            outputs.append(output)

        # Stack outputs to (batch_size, time_steps, cell_size)
        outputs = tf.stack(outputs, axis=1)

        # Extract the output of the last time step, of each sample
        idxs_last_output = tf.stack([tf.range(num_samples),
                                     tf.cast(seq_length - 1, tf.int32)], axis=1)
        final_output = tf.gather_nd(outputs, idxs_last_output)

        x = self.dense1(final_output)
        x = self.batch1(x, training=training)
        x = self.dense2(x)
        x = self.batch2(x, training=training)
        x = self.dvector(x)
        x = self.trainprob(x)

        return x

    def loss(self, x, target, seqlen, training=False):
        predictions = self.predict(x, seqlen, training=training)
        loss_value = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=target)
        return loss_value

    def grads(self, x, target, seqlen, training=False):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(x, target, seqlen, training=training)
        return tape.gradient(loss_value, self.variables)

    def fit(self,
            train_data=None,
            eval_data=None,
            epochs=500,
            verbose=1,
            **kwargs):

        train_acc = tfe.metrics.Accuracy('train_acc')
        eval_acc = tfe.metrics.Accuracy('eval_acc')

        self.history = {}
        self.history['train_acc'] = []
        self.history['eval_acc'] = []

        with tf.device(self.device_name):
            for i in range(epochs):
                self.total_step += 1
                for X, y, seqlen in tfe.Iterator(train_data):
                    grads = self.grads(x=X, target=y, seqlen=seqlen, training=True)
                    self.optimizer.apply_gradients(zip(grads, self.variables))
                if (i == 0) | ((i + 1) % verbose == 0):
                    for X, y, seqlen in tfe.Iterator(train_data):
                        logits = self.predict(X, seqlen, False)
                        preds = tf.argmax(logits, axis=1)
                        train_acc(preds, y)

                    self.history['train_acc'].append(train_acc.result().numpy())

                    # Reset metrics
                    train_acc.init_variables()

                    # Check accuracy eval dataset
                    for X, y, seqlen in tfe.Iterator(eval_data):
                        logits = self.predict(X, seqlen, False)
                        preds = tf.argmax(logits, axis=1)
                        eval_acc(preds, y)

                    self.history['eval_acc'].append(eval_acc.result().numpy())

                    # Reset metrics
                    eval_acc.init_variables()
                    print(Fore.CYAN + '[EPOCH %d]/%.2fsec ============================' % ((i + 1), time.time()-self.time))
                    self.time = time.time()
                    print(Fore.RED + 'Train accuracy at step %d: %5f%%' % (
                    self.total_step, 100.0 * self.history['train_acc'][-1]))
                    print(Fore.BLUE + 'Eval  accuracy at step %d: %5f%%' % (
                    self.total_step, 100.0 * self.history['eval_acc'][-1]) + Style.RESET_ALL)


def main():
    import glob
    import h5py

    num_data = 10

    X = list()
    y = list()
    seq = list()

    for i in range(num_data):
        fname = "data_lmfe/data_%d.h5" % i
        h5f = h5py.File(fname, 'r')
        X.append(h5f['speechs'][:].astype(np.float32))
        y += h5f['labels'][:].tolist()
        seq += h5f['seqs'][:].tolist()
        h5f.close()

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    seq = np.array(seq) / 100

    print X.shape, y.shape, seq.shape

    X_5 = list()
    for i in range(5):
        X_5.append(X[:, i * 100:i * 100 + 98, :])
    X = np.array(X_5)
    X = X.swapaxes(0, 1)
    X = X.reshape(-1, 5, 98 * 40)

    num_classes = 10 * num_data

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(X, y, seq, test_size=0.33, random_state=42)
    del X, y, seq
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train, seq_train)).shuffle(buffer_size=10000).batch(128)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test, seq_test)).shuffle(buffer_size=10000).batch(128)

    model = LSTMDvector((98 * 40,), num_classes, "./", device_name="gpu:0")

    model.fit(ds_train, ds_test, epochs=100000, verbose=10)


if __name__ == "__main__":
    main()
