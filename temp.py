import numpy as np
import tensorflow as tf
import tflearn as tl

tf.reset_default_graph()
graph = tf.Graph()

with graph.as_default():
    features = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    values = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # norm1 = tf.layers.batch_normalization(features)
    # dense1 = tf.layers.Dense(256, activation=tf.nn.relu)
    # dense_1 = dense1.apply(features)
    #
    # dense2 = tf.layers.Dense(256, activation=tf.nn.relu)
    # dense_2 = dense2.apply(dense_1)
    #
    # dense3 = tf.layers.Dense(256, activation=tf.nn.relu)
    # dense_3 = dense3.apply(dense_2)

    dense4 = tl.layers.fully_connected(features, 1)

    # dense1 = tf.nn.relu(dense1)
    #
    # norm2 = tf.layers.batch_normalization(dense1)
    # dense2 = tf.layers.dense(norm2, 8)

    loss = tf.losses.mean_squared_error(values, dense4)

    train = tf.train.AdamOptimizer(0.1).minimize(loss)

x = np.linspace(0, 1000, 100000).reshape(-1, 1) / 1000
y = (3.5 * x + 5).reshape(-1, 1)

with graph.as_default():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(2000):
            _, l = sess.run([train, loss], feed_dict={features: x, values: y})
            if i % 100 == 0:
                print(l)

        print(sess.run([dense4.W, dense4.b]))
