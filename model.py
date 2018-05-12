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
        self.device_name=device_name

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

    def loss(self, x, target):
        predictions = self.predict(x)
        loss_value = tf.losses.sparse_softmax_cross_entropy(predictions=predictions, labels=target)
        return loss_value

    def grads(self, x, target):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(x, target)
        return tape.gradient(loss_value, self.variables)

    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
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
        pass


def main():
    from preproecess import wav2cubes
    input_cube = list()
    seq = list()
    for i in range(200):
        x, _ = wav2cubes("recog.wav")
        input_cube.append(x)
        seq.append((i%30)+ 1)

    input_cube = np.array(input_cube, dtype=np.float32).reshape(-1,30,800)
    seq = np.array(seq)
    ds = tf.data.Dataset.from_tensor_slices((input_cube, seq))
    ds = ds.shuffle(buffer_size=10000).batch(32)

    model = DVectorNet((800,), 40, "./")

    for X, seq_length in tfe.Iterator(ds):
        break


if __name__ == "__main__":
    main()