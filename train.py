#!/usr/bin/python

import sys, argparse
import os
import time
import datetime


def getTimeStamp():
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d_%H%M')


def legacyReward(env, b_path = False):
    """ Return the reward of a game if a legacy
        method is used
    """
    router = OldRouter(env)
    return router.getReward(b_path)


def EvaluatePolicy(model, env,
        n_eval_episodes = 100, b_path = False, render=False):
    # t_eval_s = time.time()
    episode_rewards = np.zeros(n_eval_episodes)
    legacy_rewards = np.zeros(n_eval_episodes)
    reached_goal = np.zeros(n_eval_episodes)
    num_cycles = np.zeros(n_eval_episodes)
    # n_steps = 0
    # This takes advantage of the parallel environments
    # n_eval_episodes = n_eval_episodes // env.num_envs + 1
    obs = env.reset()
    episode_reward = np.zeros(env.num_envs)
    episode_count = 0
    while (episode_count < n_eval_episodes):
        # Make predictions and take action on all envs
        action, state = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        episode_reward += reward
        # [TODO] Implement render option for vectorized environments
        # if render:
        #     env.envs[0].render()
        #     # time.sleep(0.001)
        for env_id in range(env.num_envs):
            if done[env_id]:
                # If environment is done, create a record
                episode_rewards[episode_count] = episode_reward[env_id]
                reached_goal[episode_count] = _info[env_id]["b_at_goal"]
                num_cycles[episode_count] = _info[env_id]["num_cycles"]
                # Reset respective environment
                episode_reward[env_id] = 0
                # Increment no. of episodes, break if no more episodes are needed
                episode_count+=1
                if episode_count >= n_eval_episodes: break
    # Compute the mean values
    mean_reward = episode_rewards.mean()
    mean_legacy = legacy_rewards.mean()
    mean_goal   = reached_goal.mean()
    mean_cycles = num_cycles.mean()
    # t_eval_e = time.time()
    # print("      Eval took %d seconds" % (t_eval_e-t_eval_s))
    return mean_reward, mean_legacy, mean_goal, mean_cycles


def showIsGPU():
    if tf.test.is_gpu_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")


def plotAgentPerformance(a_rewards, o_rewards, a_goals, a_cycles, size,
                         env_info, b_path = False):
    a_rewards = np.array(a_rewards)
    a_goals = np.array(a_goals)
    a_cycles = np.array(a_cycles)
    # o_rewards = np.array(o_rewards)
    a_line = np.average(a_rewards, axis = 0)
    a_goals_line = np.average(a_goals, axis = 0)
    a_cycles_line = np.average(a_cycles, axis = 0)
    # o_line = np.average(o_rewards, axis = 0)
    a_max = np.max(a_rewards, axis = 0)
    a_min = np.min(a_rewards, axis = 0)
    # o_max = np.max(o_rewards, axis = 0)
    # o_min = np.min(o_rewards, axis = 0)
    episodes = list(range(len(a_max)))
    with plt.style.context('seaborn-paper'):
        plt.rcParams.update({'font.size': 10, 'figure.figsize': (4,3)})
        plt.figure()
        plt.fill_between(episodes, a_max, a_min, facecolor = 'red', alpha = 0.3)
        # plt.fill_between(episodes, o_max, o_min, facecolor = 'blue',
        #         alpha = 0.3)
        plt.plot(episodes, a_line, 'r-', label = 'Agent')
        plt.plot(episodes, a_goals_line, 'g.', label = 'Success Rate')
        plt.plot(episodes, a_cycles_line, 'k-.', label = 'No. Cycles')
        # plt.plot(episodes, o_line, 'b-', label = 'Baseline')
        if b_path:
            leg = plt.legend(loc = 'upper left', shadow = True, fancybox = True)
        else:
            leg = plt.legend(loc = 'lower right', shadow = True,
                    fancybox = True)
        leg.get_frame().set_alpha(0.5)
        plt.title("DMFB " + size)
        plt.xlabel('Training Epochs')
        if b_path:
            plt.ylabel('Number of Cycles')
        else:
            plt.ylabel('Score')
        plt.tight_layout()
        # Save PNG
        plt.savefig('log/' + size + env_info + '.png')
        # Save TEX
        import tikzplotlib
        tikzplotlib.clean_figure()
        tikzplotlib.save('log/' + size + env_info + '.tex')
        #plt.savefig('log/' + size + env_info + '.pgf')


def runAnExperiment(env, model=None, n_epochs=50, n_steps=20000,
                    policy_steps=128, b_path=False):
    """Run single experiment consisting of a number of episodes
    """
    if model is None:
        model = PPO2(MyCnnPolicy, env, n_steps = policy_steps)

    agent_rewards, old_rewards, episodes = [], [], []
    episode_times, reach_goals, mean_cycles = [], [], []
    
    print("INFO: Starting training")
    print("\tID\tTime\tRewards\tSucc.\tCycles")
    for i in range(n_epochs):
        t2s = time.time()
        # print("INFO: Epoch %2d started" % i)
        model.learn(total_timesteps = n_steps)
        t_eval_s = time.time()
        mean_reward, legacy_reward, reach_goal, mean_cycle = \
            EvaluatePolicy(model, model.get_env(), n_eval_episodes = 100,
                           b_path=b_path, render=False)
        t_eval_e = time.time()
        t_eval = t_eval_e - t_eval_s
        agent_rewards.append(mean_reward)
        # old_rewards.append(legacy_reward)
        episodes.append(i)
        t2e = time.time()
        episode_times.append(t2e-t2s)
        reach_goals.append(reach_goal)
        mean_cycles.append(mean_cycle)
        t2d = t2e-t2s
        # print("INFO: Epoch %2d ended in %d seconds" % (i,t2e-t2s))
        print("INFO: Epoch %3d" % i + 
              "  %3d/%4d sec"   % (t_eval,t2d)  + 
              "  %8.3f rew" % mean_reward +
              "  %4.1f suc" % reach_goal + 
              "  %5.1f cyc" % mean_cycle )
    agent_rewards = agent_rewards[-n_epochs:]
    # old_rewards = old_rewards[-n_epochs:]
    episodes = episodes[:n_epochs]
    return agent_rewards, old_rewards, episodes, reach_goals, mean_cycles, model


# def expSeveralRuns(args, n_envs, n_s, n_experiments):
def expSeveralRuns(args):
    """Run multiple experiments and plot agent performance in one plot
    """
    
    # Load configuration
    n_s = args.n_s
    n_experiments = args.n_exps
    n_envs = args.n_envs
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    b_save_model = args.b_save_model
    str_load_model = args.s_load_model
    
    str_size = str(args.size[0]) + 'x' + str(args.size[1])
    
    # Configure GPU settings, make environment, and report GPU status
    if sys.platform != 'win32':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    env = make_vec_env(MEDAEnv,wrapper_class=None,n_envs=n_envs,env_kwargs=vars(args))
    showIsGPU()
    
    # Initialize agent and old rewards
    a_rewards, a_goals, o_rewards, a_cycles = [], [], [], []
    if str_load_model != '':
        loaded_model = PPO2.load('data/' + str_load_model)
        loaded_model.set_env(env)
    else:
        loaded_model = None
        
    env_info = '_E' + str(n_epochs) + '_X'
    # Run Experiments
    for i in range(n_experiments):
        a_r, o_r, episodes, a_g, a_c, model = runAnExperiment(
            env, model=loaded_model,
            n_epochs=n_epochs, n_steps=n_steps, policy_steps=n_s
        )
        a_rewards.append(a_r)
        a_goals.append(a_g)
        o_rewards.append(o_r)
        a_cycles.append(a_c)
        if b_save_model:
            # model.save("data/model_%s" % getTimeStamp())
            model.save("data/model_%s" % env_info)
            
    plotAgentPerformance(a_rewards, o_rewards, a_goals, a_cycles, str_size, env_info)
    return


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
    # expSeveralRuns(args, n_envs=8, n_s=64, n_experiments=1)
    
    t0s = time.time()
    expSeveralRuns(args)
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
        'obs_size': (30,30),
        'droplet_sizes': [[4,4],[5,4],[5,5],[6,5],[6,6],],
        'n_envs':   8,
        'n_s':      32,
        'n_exps':   1,
        'n_epochs': 101,
        'n_steps': 2**14,
        'b_save_model': True,
        's_model_name': 'model',
        's_load_model': '',
        'b_play_mode': False
    }
    
    # Initialize parser
    parser = argparse.ArgumentParser(description='MEDA Training Module')
    parser.add_argument('--seed',type=int,default=def_args['seed'])
    parser.add_argument('--verbose',type=str,default=def_args['verbose'])
    parser.add_argument('--size',type=tuple,default=def_args['size'])
    parser.add_argument('--obs-size',type=tuple,default=def_args['obs_size'])
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
    
    import numpy as np
    np.set_printoptions(linewidth=np.inf)
    # import matplotlib
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from utils import OldRouter
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
    
    main(args)
