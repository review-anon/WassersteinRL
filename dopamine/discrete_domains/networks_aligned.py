from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from abc import ABC

from absl import logging

import cv2
import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

HRMMDNetworkType = collections.namedtuple(
	'hrmmd_network', ['q_values', 'particles']
)


def reduce_tensor(tensor):
	if len(tensor.shape) == 5 and tensor.shape[-1] == 1:
		tensor = tf.squeeze(tensor, axis=-1)
	return tensor


# TODO: now fixing activation position
class HRMMDNetwork(tf.keras.Model):
	def __init__(self, num_actions, reward_dim,
							 latent_dim=10, hidden_dim=20, condition_dim=40, num_layers=4,
							 num_samples=50, name=None):
		super(HRMMDNetwork, self).__init__(name=name)

		self.num_actions = num_actions
		self.reward_dim = reward_dim
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.condition_dim = condition_dim
		self.num_samples = num_samples
		# Defining layers.
		activation_fn = tf.keras.activations.relu
		# Setting names of the layers manually to make variable names more similar
		# with tf.slim variable names/checkpoints.
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			512, activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='fully_connected')
		self.dense2 = tf.keras.layers.Dense(
			num_actions * condition_dim, activation=activation_fn,
			kernel_initializer=self.kernel_initializer,
			name='fully_connected')

		self.decoder_layers = [
			tf.keras.layers.Dense(
				self.hidden_dim if idx < num_layers - 1 else self.reward_dim,
				activation=activation_fn if idx < num_layers - 1 else None,
				kernel_initializer=self.kernel_initializer, name='decoder_layer'
			) for idx in range(num_layers)
		]

	def decode(self, zc):
		for decoder_layer in self.decoder_layers:
			zc = decoder_layer(zc)
		return zc

	def sample(self, c, num_samples=50):
		assert len(c.shape) == 2  # [batch_size * num_actions, condition_dim]
		batch_size, condition_dim = tf.shape(c)[0], tf.shape(c)[1]
		z = tf.random.normal([batch_size, num_samples, self.latent_dim])   # (bs*na,n,latent)
		c = tf.tile(tf.expand_dims(c, axis=1), [1, num_samples, 1])        # (bs*na,n,condition)
		zc = tf.concat([z, c], axis=-1)
		q_samples = self.decode(zc)  # [batch_size * num_actions, num_samples, reward_dim]
		return q_samples

	def call(self, state, num_samples=None):
		state = reduce_tensor(state)
		if num_samples is None:
			num_samples = self.num_samples
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * condition_dim]

		c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
		c = tf.reshape(c, [-1, self.condition_dim])  # [batch_size * num_actions, condition_dim]
		q_samples = self.sample(c, num_samples)  # [batch_size * num_actions, num_samples, reward_dim]
		assert len(q_samples.shape) == 3
		q_samples = tf.reshape(q_samples, [-1, self.num_actions, num_samples, self.reward_dim])
		q_values = tf.reduce_sum(tf.reduce_mean(q_samples, axis=2), axis=2)  # [batch_size, num_actions]

		return HRMMDNetworkType(q_values, q_samples)

	def generate_samples(self, state, action_one_hot, num_samples=50):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * condition_dim]

		c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
		c = tf.reduce_sum(
			c * tf.expand_dims(action_one_hot, axis=-1),
			axis=1
		)  # [batch_size, condition_dim]

		q_samples = self.sample(c, num_samples)  # [batch_size, num_samples, reward_dim]
		return q_samples


class HRMMDNetworkV12(tf.keras.Model):
	def __init__(self, num_actions, reward_dim,
							 latent_dim=10, hidden_dim=20, condition_dim=40, num_layers=4,
							 num_samples=50, name=None):
		super(HRMMDNetworkV12, self).__init__(name=name)

		self.num_actions = num_actions
		self.reward_dim = reward_dim
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.condition_dim = condition_dim
		self.num_samples = num_samples
		# Defining layers.
		activation_fn = tf.keras.activations.relu
		# Setting names of the layers manually to make variable names more similar
		# with tf.slim variable names/checkpoints.
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			512, activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='fully_connected')
		self.dense2 = tf.keras.layers.Dense(
			num_actions * condition_dim, activation=activation_fn,
			kernel_initializer=self.kernel_initializer,
			name='fully_connected')

		self.decoder_layers = [
			tf.keras.layers.Dense(
				self.hidden_dim if idx < num_layers - 1 else self.reward_dim,
				activation=activation_fn if idx < num_layers - 1 else None,
				kernel_initializer=self.kernel_initializer, name='decoder_layer'
			) for idx in range(num_layers)
		]

	def decode(self, zc):
		for decoder_layer in self.decoder_layers:
			zc = decoder_layer(zc)
		return zc

	def sample(self, c, num_samples=50):
		assert len(c.shape) == 2  # [batch_size * num_actions, condition_dim]
		batch_size, condition_dim = tf.shape(c)[0], tf.shape(c)[1]
		z = tf.random.uniform([batch_size, num_samples, self.latent_dim])
		c = tf.tile(tf.expand_dims(c, axis=1), [1, num_samples, 1])
		zc = tf.concat([z, c], axis=-1)
		q_samples = self.decode(zc)  # [batch_size * num_actions, num_samples, reward_dim]
		return q_samples

	def call(self, state, num_samples=None):
		state = reduce_tensor(state)
		if num_samples is None:
			num_samples = self.num_samples
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * condition_dim]

		c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
		c = tf.reshape(c, [-1, self.condition_dim])  # [batch_size * num_actions, condition_dim]
		q_samples = self.sample(c, num_samples)  # [batch_size * num_actions, num_samples, reward_dim]
		assert len(q_samples.shape) == 3
		q_samples = tf.reshape(q_samples, [-1, self.num_actions, num_samples, self.reward_dim])
		q_values = tf.reduce_sum(tf.reduce_mean(q_samples, axis=2), axis=2)  # [batch_size, num_actions]

		return HRMMDNetworkType(q_values, q_samples)

	def generate_samples(self, state, action_one_hot, num_samples=50):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * condition_dim]

		c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
		c = tf.reduce_sum(
			c * tf.expand_dims(action_one_hot, axis=-1),
			axis=1
		)  # [batch_size, condition_dim]

		q_samples = self.sample(c, num_samples)  # [batch_size, num_samples, reward_dim]
		return q_samples


class NdParticleDQNet(tf.keras.Model):
	def __init__(self, num_actions, num_samples, reward_dim, name=None):
		super(NdParticleDQNet, self).__init__(name=name)
		activation_fn = tf.keras.activations.relu
		self.num_actions = num_actions
		self.num_atoms = num_samples
		self.reward_dim = reward_dim
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			512, activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='fully_connected')
		self.dense2 = tf.keras.layers.Dense(
			num_actions * num_samples * reward_dim, kernel_initializer=self.kernel_initializer,
			name='fully_connected')

	def call(self, state, num_samples=None):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = tf.compat.v1.div(x, 255.)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)
		particles = tf.reshape(x, [-1, self.num_actions, self.num_atoms, self.reward_dim])  # (b,a,n,k)
		q_values = tf.reduce_mean(particles, axis=[2, 3])  # (b,a)

		return HRMMDNetworkType(q_values, particles)

	def generate_samples(self, state, action_one_hot, num_samples=50):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * num_samples * reward_dim]
		q_samples = tf.reshape(x, [-1, self.num_actions, self.num_atoms, self.reward_dim])

		q_samples = tf.reduce_sum(
			q_samples * tf.expand_dims(tf.expand_dims(action_one_hot, axis=-1), axis=-1),
			axis=1
		)  # [batch_size, num_samples, reward_dim]

		return q_samples

	def get_state_feature(self, state):
		# added
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		return tf.stop_gradient(x)


class NdParticleDQNetDRDRL(tf.keras.Model):
	def __init__(self, num_actions, num_samples, reward_dim, name=None):
		super(NdParticleDQNetDRDRL, self).__init__(name=name)
		activation_fn = tf.keras.activations.relu
		self.num_actions = num_actions
		self.num_atoms = num_samples
		self.reward_dim = reward_dim
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			512, activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='fully_connected')
		self.dense2 = [
			tf.keras.layers.Dense(
				num_actions * num_samples, kernel_initializer=self.kernel_initializer,
				name='fully_connected')
			for _ in range(reward_dim)
		]
		self.dense_embedding = tf.keras.layers.Dense(
			512, kernel_initializer=self.kernel_initializer,
			name='fc_embedding'
		)

	def call(self, state, num_samples=None):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = tf.compat.v1.div(x, 255.)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		particles = []
		for i in range(self.reward_dim):
			one_hot = tf.expand_dims(tf.one_hot(i, self.reward_dim), axis=0)  # (1,k)
			embedding = self.dense_embedding(one_hot)  # (1,512)
			x_branch = x * embedding  # (b,512)
			particles_branch = self.dense2[i](x_branch)  # (b,a*n)
			particles.append(particles_branch)
		particles = tf.stack(particles, axis=2)  # (b,a*n,k)
		particles = tf.reshape(particles, [-1, self.num_actions, self.num_atoms, self.reward_dim])  # (b,a,n,k)
		q_values = tf.reduce_mean(particles, axis=[2, 3])  # (b,a)

		return HRMMDNetworkType(q_values, particles)

	def generate_samples(self, state, action_one_hot, num_samples=50):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		particles = []
		for i in range(self.reward_dim):
			one_hot = tf.expand_dims(tf.one_hot(i, self.reward_dim), axis=0)  # (1,k)
			embedding = self.dense_embedding(one_hot)  # (1,512)
			x_branch = x * embedding  # (b,512)
			particles_branch = self.dense2[i](x_branch)  # (b,a*n)
			particles.append(particles_branch)
		particles = tf.stack(particles, axis=2)  # (b,a*n,k)
		q_samples = tf.reshape(particles, [-1, self.num_actions, self.num_atoms, self.reward_dim])  # (b,a,n,k)

		q_samples = tf.reduce_sum(
			q_samples * tf.expand_dims(tf.expand_dims(action_one_hot, axis=-1), axis=-1),
			axis=1
		)  # [batch_size, num_samples, reward_dim]

		return q_samples


class HRMMDNetworkV21(tf.keras.Model):
	def __init__(self, num_actions, reward_dim,
							 num_samples=100, name=None):
		super(HRMMDNetworkV21, self).__init__(name=name)

		self.num_actions = num_actions
		self.reward_dim = reward_dim
		self.num_samples = num_samples
		# Defining layers.
		activation_fn = tf.keras.activations.relu
		# Setting names of the layers manually to make variable names more similar
		# with tf.slim variable names/checkpoints.
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			512, activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='fully_connected')
		self.dense2 = tf.keras.layers.Dense(
			num_actions * reward_dim * num_samples,
			kernel_initializer=self.kernel_initializer,
			name='fully_connected')

	def call(self, state):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * num_samples * reward_dim]

		q_samples = tf.reshape(x, [-1, self.num_actions, self.num_samples, self.reward_dim])
		q_values = tf.reduce_sum(tf.reduce_mean(q_samples, axis=2), axis=2)  # [batch_size, num_actions]

		return HRMMDNetworkType(q_values, q_samples)

	def generate_samples(self, state, action_one_hot, num_samples=50):
		state = reduce_tensor(state)
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size, num_actions * num_samples * reward_dim]
		q_samples = tf.reshape(x, [-1, self.num_actions, self.num_samples, self.reward_dim])

		q_samples = tf.reduce_sum(
			q_samples * tf.expand_dims(tf.expand_dims(action_one_hot, axis=-1), axis=-1),
			axis=1
		)  # [batch_size, num_samples, reward_dim]

		return q_samples


class HRMMDNetworkV22(tf.keras.Model):
	def __init__(self, num_actions, reward_dim, quantile_embedding_dim=64,
							 num_samples=100, name=None):
		super(HRMMDNetworkV22, self).__init__(name=name)

		self.quantile_embedding_dim = quantile_embedding_dim

		self.num_actions = num_actions
		self.reward_dim = reward_dim
		self.num_samples = num_samples
		# Defining layers.
		activation_fn = tf.keras.activations.relu
		self.activation_fn = activation_fn
		# Setting names of the layers manually to make variable names more similar
		# with tf.slim variable names/checkpoints.
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			512, activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='fully_connected')
		self.dense2 = tf.keras.layers.Dense(
			num_actions * reward_dim,
			kernel_initializer=self.kernel_initializer,
			name='fully_connected')

	def call(self, state, num_samples=None):
		state = reduce_tensor(state)
		if num_samples is None:
			num_samples = self.num_samples

		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)

		# state_vector_length = x.get_shape().as_list()[-1]
		# batch_size = state.get_shape().as_list()[0]
		state_vector_length = x.shape[-1]
		batch_size = state.shape[0]

		state_net_tiled = tf.tile(x, [num_samples, 1])
		quantiles_shape = [num_samples * batch_size, 1]  # self.reward_dim
		quantiles = tf.random.uniform(
			quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
		quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
		range_tensor = tf.tile(tf.range(1, self.quantile_embedding_dim + 1, 1), [1])
		pi = tf.constant(math.pi)
		quantile_net = tf.cast(range_tensor, tf.float32) * pi * quantile_net
		quantile_net = tf.cos(quantile_net)
		# Create the quantile layer in the first call. This is because
		# number of output units depends on the input shape. Therefore, we can only
		# create the layer during the first forward call, not during `.__init__()`.
		if not hasattr(self, 'dense_quantile'):
			self.dense_quantile = tf.keras.layers.Dense(
				state_vector_length, activation=self.activation_fn,
				kernel_initializer=self.kernel_initializer)
		quantile_net = self.dense_quantile(quantile_net)
		x = tf.multiply(state_net_tiled, quantile_net)

		x = self.dense1(x)
		x = self.dense2(x)  # [batch_size * num_samples, num_actions * reward_dim]
		q_samples = tf.reshape(x, [-1, num_samples, self.num_actions, self.reward_dim])
		q_samples = tf.transpose(q_samples, [0, 2, 1, 3])  # [batch_size, num_actions, num_samples, reward_dim]
		q_values = tf.reduce_sum(tf.reduce_mean(q_samples, axis=2), axis=2)  # [batch_size, num_actions]
		return HRMMDNetworkType(q_values, q_samples)

	def generate_samples(self, state, action_one_hot, num_samples=None):
		state = reduce_tensor(state)
		q_samples = self.call(state, num_samples=num_samples).particles

		q_samples = tf.reduce_sum(
			q_samples * tf.expand_dims(tf.expand_dims(action_one_hot, axis=-1), axis=-1),
			axis=1
		)  # [batch_size, num_samples, reward_dim]

		return q_samples


class Discriminator(tf.keras.Model):
	"""
	Discriminator in GAN_nd model. with condition on (s,a)
	"""
	def __init__(self, num_actions, num_samples, reward_dim, name=None):
		super(Discriminator, self).__init__(name=name)
		activation_fn = tf.keras.activations.relu
		self.num_actions = num_actions
		self.num_samples = num_samples
		self.reward_dim = reward_dim
		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
		# Defining layers.
		self.conv1 = tf.keras.layers.Conv2D(
			32, [8, 8], strides=4, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv2 = tf.keras.layers.Conv2D(
			64, [4, 4], strides=2, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.conv3 = tf.keras.layers.Conv2D(
			64, [3, 3], strides=1, padding='same', activation=activation_fn,
			kernel_initializer=self.kernel_initializer, name='Conv')
		self.flatten = tf.keras.layers.Flatten()

		#  version 1
		# state embedding
		# self.s_dense = tf.keras.Sequential([
		# 	tf.keras.layers.Dense(512, activation=activation_fn, kernel_initializer=self.kernel_initializer),
		# 	tf.keras.layers.Dense(32, activation=activation_fn, kernel_initializer=self.kernel_initializer),
		# ])
		#
		# action embedding.
		# self.a_dense = tf.keras.Sequential([
		# 	tf.keras.layers.Dense(32, activation=activation_fn, kernel_initializer=self.kernel_initializer),
		# ])
		# Q-value embedding  shape=(batch_size, n_samples, reward_dim)
		# self.q_dense = tf.keras.Sequential([
		# 	tf.keras.layers.Dense(32, activation=activation_fn, kernel_initializer=self.kernel_initializer),
		# ])
		# self.output_dense = tf.keras.layers.Dense(1, activation='linear')

		self.s_dense = tf.keras.Sequential([  # version 2
				tf.keras.layers.Dense(512, activation=activation_fn),
				tf.keras.layers.Dense(128, activation=activation_fn),
				tf.keras.layers.Dense(32, activation=activation_fn),
			])
		self.a_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(32, activation=activation_fn),
			tf.keras.layers.Dense(32, activation=activation_fn),
			tf.keras.layers.Dense(32, activation=activation_fn),
		])
		self.q_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(32, activation=activation_fn),
			tf.keras.layers.Dense(32, activation=activation_fn),
			tf.keras.layers.Dense(32, activation=activation_fn),
		])
		self.output_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(16, activation=activation_fn),
			tf.keras.layers.Dense(1, activation='linear'),
		])

	def call(self, Q, state=None, action_one_hot=None, generator_net=None):
		# for state embedding
		state = reduce_tensor(state)       # (None, x, x, x)
		s = tf.cast(state, tf.float32)
		s = tf.compat.v1.div(s, 255.)
		s = self.conv1(s)
		s = self.conv2(s)
		s = self.conv3(s)
		s = self.flatten(s)
		s = self.s_dense(s)                 # (b, dim)
		s = tf.tile(tf.expand_dims(s, axis=1), [1, self.num_samples, 1])   # (b, num_samples, dim)

		# for action embedding
		a = self.a_dense(action_one_hot)    # (b, action_num) -> (b, action_dim)
		a = tf.tile(tf.expand_dims(a, axis=1), [1, self.num_samples, 1])   # (b, num_samples, dim)

		# for value function embedding
		q = self.q_dense(Q)                 # (b, num_sample, reward_dim) -> (b, num_sample, dim)
		assert s.get_shape().as_list() == a.get_shape().as_list() == q.get_shape().as_list()

		saq = tf.concat([s, a, q], axis=-1)    # (b, num_sample, dim*3)
		saq_output = self.output_dense(saq)    # (b, num_sample, 1)
		return tf.reshape(tf.squeeze(saq_output), [-1, 1])          # (b * num_sample, 1)





	# def call(self, Q, state=None, action_one_hot=None, generator_net=None):
	# 	s = generator_net.get_state_feature(state)
	# 	s = self.s_dense(s)                        # (b, dim)
	# 	s = tf.tile(tf.expand_dims(s, axis=1), [1, self.num_samples, 1])   # (b, num_samples, dim)
	# 	# for action embedding
	# 	a = self.a_dense(action_one_hot)    # (b, action_num) -> (b, action_dim)
	# 	a = tf.tile(tf.expand_dims(a, axis=1), [1, self.num_samples, 1])   # (b, num_samples, dim)
	#
	# 	# for value function embedding
	# 	q = self.q_dense(Q)                 # (b, num_sample, reward_dim) -> (b, num_sample, dim)
	# 	assert s.get_shape().as_list() == a.get_shape().as_list() == q.get_shape().as_list()
	#
	# 	saq = tf.concat([s, a, q], axis=-1)    # (b, num_sample, dim*3)
	# 	# saq = tf.concat([tf.multiply(s, a), q])
	# 	saq_output = self.output_dense(saq)    # (b, num_sample, 1)
	# 	return tf.squeeze(saq_output)          # (b, num_sample)

#
# class Generator(tf.keras.Model):
# 	def __init__(self, num_actions, reward_dim,
# 					   latent_dim=10, hidden_dim=20, condition_dim=40, num_layers=4,
# 					   name=None):
# 		super(Generator, self).__init__(name=name)
#
# 		self.num_actions = num_actions
# 		self.reward_dim = reward_dim
# 		self.latent_dim = latent_dim
# 		self.hidden_dim = hidden_dim
# 		self.condition_dim = condition_dim
# 		# self.num_samples = num_samples
# 		activation_fn = tf.keras.activations.relu
# 		self.kernel_initializer = tf.keras.initializers.VarianceScaling(
# 			scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
# 		# Defining layers.
# 		self.conv1 = tf.keras.layers.Conv2D(
# 			32, [8, 8], strides=4, padding='same', activation=activation_fn,
# 			kernel_initializer=self.kernel_initializer, name='Conv')
# 		self.conv2 = tf.keras.layers.Conv2D(
# 			64, [4, 4], strides=2, padding='same', activation=activation_fn,
# 			kernel_initializer=self.kernel_initializer, name='Conv')
# 		self.conv3 = tf.keras.layers.Conv2D(
# 			64, [3, 3], strides=1, padding='same', activation=activation_fn,
# 			kernel_initializer=self.kernel_initializer, name='Conv')
# 		self.flatten = tf.keras.layers.Flatten()
# 		self.dense1 = tf.keras.layers.Dense(
# 			512, activation=activation_fn,
# 			kernel_initializer=self.kernel_initializer, name='fully_connected')
# 		self.dense2 = tf.keras.layers.Dense(
# 			num_actions * condition_dim, activation=activation_fn,
# 			kernel_initializer=self.kernel_initializer,
# 			name='fully_connected')
#
# 		self.decoder_layers = [
# 			tf.keras.layers.Dense(
# 				self.hidden_dim if idx < num_layers - 1 else self.reward_dim,
# 				activation=activation_fn if idx < num_layers - 1 else None,
# 				kernel_initializer=self.kernel_initializer, name='decoder_layer'
# 			) for idx in range(num_layers)
# 		]
#
# 	def decode(self, zc):
# 		for decoder_layer in self.decoder_layers:
# 			zc = decoder_layer(zc)
# 		return zc
#
# 	def sample(self, c):
# 		assert len(c.shape) == 2  # [batch_size * num_actions, condition_dim]
# 		batch_size, condition_dim = tf.shape(c)[0], tf.shape(c)[1]
# 		z = tf.random.normal([batch_size*self.num_actions, self.latent_dim])
# 		zc = tf.concat([z, c], axis=-1)
# 		assert zc.get_shape().as_list() == [batch_size * self.num_actions, self.latent_dim+self.condition_dim]
# 		q_samples = self.decode(zc)  # [batch_size * num_actions, reward_dim]
#
# 		assert zc.get_shape().as_list() == [batch_size * self.num_actions, self.latent_dim + self.condition_dim]
# 		assert q_samples.get_shape().as_list() == [batch_size * self.num_actions, self.reward_dim]
# 		return q_samples
#
# 	def call(self, state, num_samples=None):
# 		state = reduce_tensor(state)
# 		x = tf.cast(state, tf.float32)
# 		x = x / 255
# 		x = self.conv1(x)
# 		x = self.conv2(x)
# 		x = self.conv3(x)
# 		x = self.flatten(x)
# 		x = self.dense1(x)
# 		x = self.dense2(x)  # [batch_size, num_actions * condition_dim]
#
# 		c = tf.reshape(x, [-1, self.num_actions, self.condition_dim])  # [batch_size, num_actions, condition_dim]
# 		c = tf.reshape(c, [-1, self.condition_dim])  # [batch_size * num_actions, condition_dim]
# 		q_samples = self.sample(c)  # [batch_size * num_actions, reward_dim]
# 		assert len(q_samples.shape) == 2
# 		q_samples = tf.reshape(q_samples, [-1, self.num_actions, self.reward_dim])  # [batch_size, num_actions, reward_dim]
# 		q_values = tf.reduce_mean(q_samples, axis=2)  # [batch_size, num_actions]
#
# 		return HRMMDNetworkType(q_values, q_samples)


# class Discriminator(tf.keras.Model):
# 	"""
# 	Discriminator in GAN_nd model. *without* condition on (s,a)
# 	"""
# 	def __init__(self, num_actions, num_samples, reward_dim, name="Disc"):
# 		super(Discriminator, self).__init__(name=name)
# 		activation_fn = tf.keras.activations.relu
#
# 		# Q-value embedding  shape=(batch_size, n_samples, reward_dim)
# 		self.dense1 = tf.keras.layers.Dense(32, activation=activation_fn, name="dense1")
# 		self.dense2 = tf.keras.layers.Dense(64, activation=activation_fn, name="dense2")
# 		self.dense3 = tf.keras.layers.Dense(32, activation=activation_fn, name="dense3")
# 		self.output_layer = tf.keras.layers.Dense(1, activation=None, name="out")
#
# 	def call(self, Q, state=None, action_one_hot=None):
# 		# for value function embedding
# 		q_dense = self.dense1(Q)
# 		q_dense = self.dense2(q_dense)
# 		q_dense = self.dense3(q_dense)
# 		q_output = self.output_layer(q_dense)
#
# 		# (b, num_sample, reward_dim) -> (b, num_sample, 1)
# 		return tf.squeeze(q_output)       # (b, num_sample)


# if __name__ == "__main__":
# 	batch_size = 2
# 	num_actions = 4
# 	num_samples = 10
# 	reward_dim = 4
#
# 	model = DiscriminatorCond(num_actions, num_samples, reward_dim)
#
# 	state = tf.random.uniform((batch_size, 84, 84, 4), maxval=255.)
# 	action = tf.random.uniform((batch_size, ), maxval=num_actions, dtype=tf.int32)
# 	action_one_hot = tf.one_hot(action, depth=num_actions)
# 	Q = tf.random.uniform((batch_size, num_samples, reward_dim), dtype=tf.float32)
#
# 	out = model(state, action_one_hot, Q)
# 	print(out.shape)
