# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfd = tf.contrib.distributions

handle = traindict['handle']
 
# Build a Bayesian neural net. We use the Flipout Monte Carlo estimator for
# each layer: this enables lower variance stochastic gradients than naive
# reparameterization.
with tf.name_scope("bayesian_neural_net", [traindict['local']]):
  neural_net = tf.keras.Sequential()
  for units in paramdict['layersize']:
    layer = tfp.layers.DenseFlipout(
        int(units),
        activation=paramdict['activation'])
    neural_net.add(layer)
  neural_net.add(tfp.layers.DenseFlipout(2))
  logits = neural_net(traindict['local'])
  labels_distribution = tfd.Categorical(logits=logits)

# Compute the -ELBO as the loss, averaged over the batch size.
neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(traindict['labels']))
kl = sum(neural_net.losses) / paramdict['trainsize']
elbo_loss = neg_log_likelihood + kl

# Build metrics for evaluation. Predictions are formed from a single forward
# pass of the probabilistic layers. They are cheap but noisy predictions.
predictions = tf.argmax(logits, axis=1)
accuracy, accuracy_update_op = tf.metrics.accuracy(
    labels=traindict['labels'], predictions=predictions)
  
with tf.name_scope("train"):
  opt = tf.train.AdamOptimizer(learning_rate=paramdict['lr'])

  train_op = opt.minimize(elbo_loss)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  # Run the training loop.
  train_handle = sess.run(traindict['iterator'].string_handle())
  val_handle = sess.run(iterators['val'].string_handle())
  test_handle = sess.run(iterators['test'].string_handle())
  for step in range(paramdict['mxstep']):
    _ = sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: train_handle})

    if step % 100 == 0:
      print("Step: {:>3d}".format(step)) 
      train_loss_value, train_accuracy = sess.run(
          [elbo_loss, accuracy], feed_dict={handle: train_handle})
      print("Train loss: {:.3f} Train accuracy: {:.3f}".format(
          train_loss_value, train_accuracy))
            
