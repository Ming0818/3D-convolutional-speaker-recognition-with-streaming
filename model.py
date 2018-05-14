import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

# eager execution
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

# Hyper parameters
LEARNING_RATE = 0.001


class DVectorNet(tf.keras.Model):
    def __init__(self, input_dim, out_dim, checkpoint_directory, batch_size=32, device_name="cpu:0"):
        super(DVectorNet, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.checkpoint_directory = checkpoint_directory
        self.batch_size = batch_size
        self.device_name = device_name

        # lstm cell
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(64)

        # cnn
        # self.conv1 = tf.layers.Conv2D(32, 8, 8, padding='same', activation=tf.nn.relu)
        # self.batch1 = tf.layers.BatchNormalization()
        # self.conv2 = tf.layers.Conv2D(64, 4, 4, padding='same', activation=tf.nn.relu)
        # self.batch2 = tf.layers.BatchNormalization()
        # self.conv3 = tf.layers.Conv2D(64, 3, 3, padding='same', activation=tf.nn.relu)
        # self.flatten = tf.layers.Flatten()

        # dense
        self.dense1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.batch1 = tf.layers.BatchNormalization()
        self.dense2 = tf.layers.Dense(out_dim, activation=tf.nn.softmax)

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

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

        return x

    def loss(self, x, target, seqlen, training=False):
        predictions = self.predict(x, seqlen, training=training)
        loss_value = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=target)
        return loss_value

    def grads(self, x, target, seqlen,training=False):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(x, target, seqlen, training=training)
        return tape.gradient(loss_value, self.variables)

    def fit(self,
            train_data=None,
            eval_data=None,
            batch_size=None,
            epochs=500,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):

        train_acc = tfe.metrics.Accuracy('train_acc')
        eval_acc = tfe.metrics.Accuracy('eval_acc')

        self.history = {}
        self.history['train_acc'] = []
        self.history['eval_acc'] = []

        with tf.device(self.device_name):
            for i in range(epochs):
                x = 0
                for X, y, seqlen in tfe.Iterator(train_data):
                    print(x)
                    grads = self.grads(x=X, target=y, seqlen=seqlen, training=True)
                    self.optimizer.apply_gradients(zip(grads, self.variables))
                    x +=1
                x = 0
                for X, y, seqlen in tfe.Iterator(train_data):
                    logits = self.predict(X, seqlen, False)
                    preds = tf.argmax(logits, axis=1)
                    train_acc(preds, y)

                self.history['train_acc'].append(train_acc.result().numpy())

                train_acc.init_variables()

                # Check accuracy eval dataset
                for X, y, seqlen in tfe.Iterator(eval_data):
                    logits = self.predict(X, seqlen, False)
                    preds = tf.argmax(logits, axis=1)
                    eval_acc(preds, y)
                self.history['eval_acc'].append(eval_acc.result().numpy())
                # Reset metrics
                eval_acc.init_variables()

                if (i==0) | ((i+1)%verbose==0):
                    print('Train accuracy at epoch %d: ' %(i+1), self.history['train_acc'][-1])
                    print('Eval accuracy at epoch %d: ' %(i+1), self.history['eval_acc'][-1])


def main():
    from preproecess import wav2cubes
    input_cube = list()
    seq = list()
    for i in range(200):
        x, _ = wav2cubes("recog.wav")
        input_cube.append(x)
        seq.append((i % 30) + 1)

    input_cube = np.array(input_cube, dtype=np.float32).reshape(-1, 30, 800)
    seq = np.array(seq)
    # yy = np.zeros((200,30), dtype=np.int32)
    # for i in range(200):
    #     import random
    #     t = random.randint(0,29)
    #     yy[i,t] = 1

    yy = np.random.randint(0,30,(200,1))

    ds = tf.data.Dataset.from_tensor_slices((input_cube, yy, seq))
    ds2 = tf.data.Dataset.from_tensor_slices((input_cube, yy, seq))
    ds = ds.shuffle(buffer_size=10000).batch(32)
    ds2 = ds2.shuffle(buffer_size=10000).batch(32)

    model = DVectorNet((800,), 30, "./")

    model.fit(ds, ds2)



if __name__ == "__main__":
    main()
