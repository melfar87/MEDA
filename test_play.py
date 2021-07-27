#!/usr/bin/python

import sys, argparse
import os
import time
import pygame
from gym.utils import play

def showIsGPU():
    if tf.test.is_gpu_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")


def testEnvironment(env, args):
    for i in range(1000):
        obs = env.reset()
        done, state = False, None
        
        while not done:
            obs, reward, done, _info = env.step(action)


def main(args):
    t0s = time.time()
    vec_env = make_vec_env(MEDAEnv,wrapper_class=None,n_envs=1,env_kwargs=vars(args))
    env = vec_env.envs[0]
    # testEnvironment(env, args)
    play.play(env,zoom=5,fps=30,transpose=False)
    t0e = time.time()
    print("Time = %d seconds" % (t0e-t0s))
    print('### Finished train.py successfully ###')
    
    return


if __name__ == '__main__':
        
    # List of args default values
    def_args = {
        'seed': 123,
        'verbose':  '1',
        'size':     (30,30),
        'droplet_sizes': [[4,4],],
        'n_envs':   8,
        'n_s':      64,
        'n_exps':   1,
        'n_epochs': 21,
        'n_steps': 20000,
        'b_save_model': True,
        's_model_name': 'model',
        's_load_model': '',
        'b_play_mode': True
    }
    
    # Initialize parser
    parser = argparse.ArgumentParser(description='MEDA Training Module')
    parser.add_argument('--seed',type=int,default=def_args['seed'])
    parser.add_argument('-v','--verbose',type=str,default=def_args['verbose'])
    parser.add_argument('-s','--size',type=tuple,default=def_args['size'])
    parser.add_argument('--droplet-sizes',type=list,default=def_args['droplet_sizes'])
    parser.add_argument('--n-envs',type=int,default=def_args['n_envs'])
    parser.add_argument('--n-s',type=int,default=def_args['n_s'])
    parser.add_argument('--n-exps',type=int,default=def_args['n_exps'])
    parser.add_argument('--n-epochs',type=int,default=def_args['n_epochs'])
    parser.add_argument('--n-steps',type=int,default=def_args['n_steps'])
    parser.add_argument('--b-save-model',type=bool,default=def_args['b_save_model'])
    parser.add_argument('--s-model-name',type=str,default=def_args['s_model_name'])
    parser.add_argument('--s-load-model',type=str,default=def_args['s_load_model'])
    parser.add_argument('--b-play-mode',type=bool,default=def_args['b_play_mode'])

    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose
    
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import tensorflow as tf
    # from utils import OldRouter
    # from my_net import MyCnnPolicy
    # from envs.dmfb import *
    from envs.meda import *
    from stable_baselines.common import make_vec_env # tf_util
    # from stable_baselines.common.vec_env import DummyVecEnv
    # from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy
    # from stable_baselines.common.evaluation import evaluate_policy
    # from stable_baselines import PPO2
    
    if args.seed >= 0:
        from stable_baselines.common.misc_util import set_global_seeds
        set_global_seeds(args.seed)
    
    main(args)