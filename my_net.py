#!/usr/bin/python

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import *
from stable_baselines.common.policies import ActorCriticPolicy

def myCnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer1 = activ(conv(scaled_images, 'c1',
            n_filters = 32, filter_size = 3,
            stride = 1, pad = 'SAME', **kwargs))
    layer2 = activ(conv(layer1, 'c2',
            n_filters = 64, filter_size = 3,
            stride = 1, pad = 'SAME', **kwargs))
    layer3 = activ(conv(layer2, 'c3',
            n_filters = 64, filter_size = 3,
            stride = 1, pad = 'SAME', **kwargs))
    layer3 = conv_to_fc(layer3)
    return activ(linear(layer3, 'fc1',
            n_hidden = 128, init_scale = np.sqrt(2)))

class MyCnnPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space,
            n_env, n_steps, n_batch, reuse=False,
            cnn_extractor=nature_cnn,
            feature_extraction="cnn", **kwargs):
        super(MyCnnPolicy, self).__init__(
                sess, ob_space, ac_space, n_env,
                n_steps, n_batch, reuse=reuse,
                scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            pi_latent = vf_latent = myCnn(self.processed_obs, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, randomize=False):
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        elif randomize:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp
    
    # def step(self, obs, state=None, mask=None, deterministic=False, randomize=False):
    #     if deterministic:
    #         action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
    #                                                {self.obs_ph: obs})
    #     else:
    #         action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
    #                                                {self.obs_ph: obs})
    #     return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
