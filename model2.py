import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

# eager execution
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

# Hyper parameters
LEARNING_RATE = 0.001

layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):
    """_IdentityBlock is the block that has no conv layer at shortcut.
    Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
    """

    def __init__(self, kernel_size, filters, stage, block):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1

        self.conv2a = layers.Conv2D(
            filters1, (1, 1), name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

    def __call__(self, input_tensor, training=False):
        return self.call(input_tensor=input_tensor, training=training)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
    """_ConvBlock is the block that has a conv layer at shortcut.
    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
          'channels_last').
        strides: strides for the convolution. Note that from stage 3, the first
         conv layer at main path is with strides=(2,2), and the shortcut should
         have strides=(2,2) as well.
    """

    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1

        self.conv2a = layers.Conv2D(
            filters1, (1, 1),
            strides=strides,
            name=conv_name_base + '2a')

        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b')

        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

        self.conv_shortcut = layers.Conv2D(
            filters3, (1, 1),
            strides=strides,
            name=conv_name_base + '1')

        self.bn_shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

    def __call__(self, input_tensor, training=False):
        return self.call(input_tensor=input_tensor, training=training)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class DVectorNet(tf.keras.Model):
    def __init__(self, input_dim, out_dim, checkpoint_directory, batch_size=32, device_name="cpu:0"):
        super(DVectorNet, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.checkpoint_directory = checkpoint_directory
        self.batch_size = batch_size
        self.device_name = device_name

        def conv_block(filters, stage, block, strides=(2, 2)):
            return _ConvBlock(
                3,
                filters,
                stage=stage,
                block=block,
                strides=strides)

        def id_block(filters, stage, block):
            return _IdentityBlock(
                3, filters, stage=stage, block=block)

        self.conv1 = layers.Conv2D(
            64, (7, 7),
            strides=(2, 2),
            padding='same',
            name='conv1')

        bn_axis = 1

        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D(
            (3, 3), strides=(2, 2))

        self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64, 256], stage=2, block='b')
        self.l2c = id_block([64, 64, 256], stage=2, block='c')

        self.l3a = conv_block([128, 128, 512], stage=3, block='a')
        self.l3b = id_block([128, 128, 512], stage=3, block='b')
        self.l3c = id_block([128, 128, 512], stage=3, block='c')
        self.l3d = id_block([128, 128, 512], stage=3, block='d')

        self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
        self.l4b = id_block([256, 256, 1024], stage=4, block='b')
        self.l4c = id_block([256, 256, 1024], stage=4, block='c')
        self.l4d = id_block([256, 256, 1024], stage=4, block='d')
        self.l4e = id_block([256, 256, 1024], stage=4, block='e')
        self.l4f = id_block([256, 256, 1024], stage=4, block='f')

        self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
        self.l5b = id_block([512, 512, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048], stage=5, block='c')

        self.avg_pool = layers.AveragePooling2D(
            (7, 7), strides=(7, 7))

        self.flatten = layers.Flatten()
        self.dvector = layers.Dense(2048, name='dvector', activation=tf.nn.relu)
        self.fc1000 = layers.Dense(out_dim, name='fc1000', activation=tf.nn.softmax)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.l2a(x, training=training)
        print "here"
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)
        print "here"

        x = self.avg_pool(x)
        print "here1"

        x = self.dvector(self.flatten(x))
        print "here2"

        x = self.fc1000(x)
        print "here3"
        return x

    def loss(self, input_tensor, target, training=False):
        predictions = self.call(input_tensor, training=training)
        loss_value = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=target)
        return loss_value

    def grads(self, input_tensor, target, training=False):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(input_tensor, target, training=training)
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
                for X, y in tfe.Iterator(train_data):
                    grads = self.grads(input_tensor=X, target=y, training=True)
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                for X, y in tfe.Iterator(train_data):
                    logits = self.call(input_tensor=X, target=y, training=False)
                    preds = tf.argmax(logits, axis=1)
                    train_acc(preds, y)

                self.history['train_acc'].append(train_acc.result().numpy())

                train_acc.init_variables()

                # Check accuracy eval dataset
                for X, y in tfe.Iterator(eval_data):
                    logits = self.call(input_tensor=X, target=y, training=False)
                    preds = tf.argmax(logits, axis=1)
                    eval_acc(preds, y)
                self.history['eval_acc'].append(eval_acc.result().numpy())
                # Reset metrics
                eval_acc.init_variables()

                if (i == 0) | ((i + 1) % verbose == 0):
                    print('Train accuracy at epoch %d: ' % (i + 1), self.history['train_acc'][-1])
                    print('Eval accuracy at epoch %d: ' % (i + 1), self.history['eval_acc'][-1])


def main():
    from preproecess import wav2cubes
    input_cube = list()
    for i in range(200):
        x, _ = wav2cubes("recog.wav")
        input_cube.append(x)

    input_cube = np.array(input_cube, dtype=np.float32).reshape(-1, 30, 20, 40)
    # yy = np.zeros((200,30), dtype=np.int32)
    # for i in range(200):
    #     import random
    #     t = random.randint(0,29)
    #     yy[i,t] = 1

    yy = np.random.randint(0, 30, (200, 1))

    ds = tf.data.Dataset.from_tensor_slices((input_cube, yy))
    ds2 = tf.data.Dataset.from_tensor_slices((input_cube, yy))
    ds = ds.shuffle(buffer_size=10000).batch(32)
    ds2 = ds2.shuffle(buffer_size=10000).batch(32)

    model = DVectorNet((20, 40), 30, "./", device_name="gpu:0")

    model.fit(ds, ds2)


if __name__ == "__main__":
    main()