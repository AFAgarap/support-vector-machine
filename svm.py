from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf


BATCH_SIZE = 32
EPOCHS = 500
NUM_CLASSES = 2


class SVM(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SVM, self).__init__()
        self.weights_var = tf.Variable(
                tf.random.normal(
                    stddev=1e-2,
                    shape=[kwargs['num_features'], kwargs['num_classes']]
                    )
                )
        self.biases_var = tf.Variable(
                tf.random.normal(
                    stddev=1e-2,
                    shape=[kwargs['num_classes']]
                    )
                )

    def call(self, features):
        output = tf.matmul(features, self.weights_var) + self.biases_var
        return output


def loss_fn(model, features, labels):
    output = model(features)
    regularization = tf.reduce_mean(model.weights_var)
    squared_hinge_loss = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.zeros([BATCH_SIZE, NUM_CLASSES]),
                    (1. - tf.cast(labels, tf.float32) * output)
                    )
                )
            )
    loss = regularization + 5e-1 * squared_hinge_loss
    return loss


def train_step(model, opt, loss, features, labels):
    with tf.GradientTape() as tape:
        train_loss = loss(model, features, labels)
    gradients = tape.gradient(
            train_loss,
            [model.weights_var, model.biases_var]
            )
    opt.apply_gradients(zip(gradients, [model.weights_var, model.biases_var]))
    return train_loss


def train(model, opt, loss, dataset, epochs=EPOCHS):
    for epoch in range(epochs):
        epoch_loss = []
        epoch_accuracy = []
        for batch_features, batch_labels in dataset:
            train_loss = train_step(
                    model, opt, loss, batch_features, batch_labels
                    )
            predictions = model(batch_features)
            predictions = tf.sign(predictions)
            predictions = predictions.numpy().reshape(-1, NUM_CLASSES)
            accuracy = tf.metrics.Accuracy()
            accuracy(tf.argmax(predictions, 1), tf.argmax(batch_labels, 1))
            epoch_loss.append(train_loss)
            epoch_accuracy.append(accuracy.result().numpy())
        epoch_loss = tf.reduce_mean(epoch_loss)
        epoch_accuracy = tf.reduce_mean(epoch_accuracy)
        if epoch != 0 and (epoch + 1) % 100 == 0:
            print('epoch {}/{} : mean loss = {}, mean accuracy = {}'.format(
                epoch + 1, epochs, epoch_loss, epoch_accuracy
                ))


features, labels = load_breast_cancer().data, load_breast_cancer().target
x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.30, stratify=labels
        )

x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
        )
train_dataset = train_dataset.prefetch(BATCH_SIZE * 4)
train_dataset = train_dataset.shuffle(BATCH_SIZE * 4)
train_dataset = train_dataset.batch(BATCH_SIZE, True)

model = SVM(num_features=30, num_classes=NUM_CLASSES)
# optimizer = tf.optimizers.Adam(learning_rate=1e-1, decay=1e-6)
optimizer = tf.optimizers.SGD(
        learning_rate=1e-1, momentum=9e-1, decay=1e-6, nesterov=True
        )
model(x_train[:10])
print(model.summary())
train(model, optimizer, loss_fn, train_dataset)
model.trainable = False
print(model.summary())
predictions = model(x_test)
predictions = tf.sign(predictions)
predictions = predictions.numpy().reshape(-1, NUM_CLASSES)
accuracy = tf.metrics.Accuracy()
test_accuracy = accuracy(tf.argmax(predictions, 1), tf.argmax(y_test, 1))
print('test accuracy : {}'.format(test_accuracy.numpy()))
