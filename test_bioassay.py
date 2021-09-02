#!/usr/bin/python


from meda_biochip import MedaBiochip
from meda_scheduler import MedaScheduler
import sys, argparse
import os
import time
import datetime
import pickle

from matplotlib.pyplot import pause
import matplotlib.animation as animation

from meda_sgs import Bioassays
from meda_utils import showIsGPU, State

# from utils import *

def main(args):
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    # b_backup_model = True
    # t0s = time.time()
    # Sizes are (width, height)
    # sizes = [(30,30),]
    # for s in sizes:
    #     args = {'width': s[1], 'height': s[0],
    #             'n_modules': 0,
    #             'b_degrade': True,
    #             'per_degrade': 0.1}
    # expSeveralRuns(args, n_envs=8, n_policysteps=64, n_experiments=1)
    
    # Load environment and policy model
    if sys.platform != 'win32':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0,allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    env = make_vec_env(MEDAEnv,wrapper_class=None,n_envs=args.n_envs,env_kwargs=vars(args))
    showIsGPU(tf)
    policy = PPO2.load('data/' + args.s_load_model)
    policy.set_env(env)
    
    t_total_s = time.time()
    testBioassay(args, env=env, policy=policy)
    t_total_e = time.time()
    t_total = t_total_e - t_total_s
    print("| Finished training in %26s %4d min %2d sec |" 
          % (" ",t_total//60, t_total%60))
    print("+=================================================================+")
    # print(getTimeStamp())
    print(" ")
    
    return


def testBioassay(args, env=None, policy=None):
    width, height = args.size
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Create bioassay, sequence graph, biochip and schedule
    bioassay = Bioassays()
    sequence_graph = bioassay.sg_Simple
    biochip = MedaBiochip(env=env, policy=policy, width=width, height=height)
    scheduler = MedaScheduler(biochip=biochip, env=env, policy=policy,
                              width=width, height=height)
    bioassay.importBioassay(sequence_graph, scheduler)
    scheduler.preprocessAllMos()
    print("We are doomed...")
    
    for k in range(100):
        sch_state:State = scheduler.tick()
        if sch_state == State.Done:
            break
    
    return

if __name__ == '__main__':
    b_disable_warnings = False
    b_load_tf = True
    # List of args default values
    def_args = {
        'seed':              -1,
        'verbose':           '3',
        's_mode':            'test', # train | test
        'size':              (30,30),
        'obs_size':          (30,30),
        'droplet_sizes':     [[4,4],[5,4],[5,5],[6,5],[6,6],],
        'n_envs':            8,
        'n_policysteps':     64,
        'n_exps':            1,
        'n_epochs':          40,
        'n_total_timesteps': 2**14,
        'b_save_model':      True,
        's_model_name':      'TMP_E',
        's_suffix':          '',
        's_load_model':      '0826a_030x030_E100_NPS64_00', #'0725_030x030_E100__00', 
        'b_play_mode':       False,
        'deg_mode':          'normal',
        'deg_perc':          0.2,
        'deg_size':          2,
        'description':       ''
    }
    
    # Initialize parser
    parser = argparse.ArgumentParser(description='MEDA Training Module')
    parser.add_argument('--seed',type=int,default=def_args['seed'])
    parser.add_argument('--verbose',type=str,default=def_args['verbose'])
    parser.add_argument('--s-mode',type=str,default=def_args['s_mode'])
    parser.add_argument('--size',type=tuple,default=def_args['size'])
    parser.add_argument('--obs-size',type=tuple,default=def_args['obs_size'])
    parser.add_argument('--droplet-sizes',type=list,default=def_args['droplet_sizes'])
    parser.add_argument('--n-envs',type=int,default=def_args['n_envs'])
    parser.add_argument('--n-policysteps',type=int,default=def_args['n_policysteps'])
    parser.add_argument('--n-exps',type=int,default=def_args['n_exps'])
    parser.add_argument('--n-epochs',type=int,default=def_args['n_epochs'])
    parser.add_argument('--n-total-timesteps',type=int,default=def_args['n_total_timesteps'])
    parser.add_argument('--b-save-model',type=bool,default=def_args['b_save_model'])
    parser.add_argument('--s-model-name',type=str,default=def_args['s_model_name'])
    parser.add_argument('--s-suffix',type=str,default=def_args['s_suffix'])
    parser.add_argument('--s-load-model',type=str,default=def_args['s_load_model'])
    parser.add_argument('--b-play-mode',type=bool,default=def_args['b_play_mode'])
    parser.add_argument('--deg-mode',type=str,default=def_args['deg_mode'])
    parser.add_argument('--deg-perc',type=float,default=def_args['deg_perc'])
    parser.add_argument('--deg-size',type=int,default=def_args['deg_size'])
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose
    import warnings
    if b_disable_warnings:
        warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r"The name")
    import numpy as np
    np.set_printoptions(linewidth=np.inf)
    # import matplotlib
    import matplotlib.pyplot as plt
    if b_load_tf:
        import tensorflow as tf
    # from utils import OldRouter
    from my_net import MyCnnPolicy
    # from envs.dmfb import *
    from envs.meda import *
    from stable_baselines.common import make_vec_env # tf_util
    # from stable_baselines.common.vec_env import DummyVecEnv
    # from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy
    # from stable_baselines.common.evaluation import evaluate_policy
    from stable_baselines import PPO2
    # Set random seeds before starting for SB, Random, tensorflow, numpy, gym
    if args.seed >= 0:
        from stable_baselines.common.misc_util import set_global_seeds
        set_global_seeds(args.seed)
    if b_load_tf and b_disable_warnings:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    main(args)