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
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.batch_size)

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

    def predict(self, x, batch_size=None, verbose=0, steps=None, training=False):
        if isinstance(x, (np.ndarray, np.generic)):
            x = np.reshape(x, self.input_dim)
            x = tf.convert_to_tensor(x)

        num_samples = tf.shape(x)[0]
        state = self.rnn_cell.zero_state(num_samples, dtype=tf.float32)
        unstacked_input = tf.unstack(x, axis=1)

        for input_step in unstacked_input:
            output, state = self.rnn_cell(input_step, state)

        x = self.dense1(x)
        x = self.batch1(x, training=training)
        x = self.dense2(x)

        return x

    def loss(self, x, target):
        predictions = self.predict(x)
        loss_value = tf.losses.mean_squared_error(predictions=predictions, labels=target)
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