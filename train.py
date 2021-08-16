#!/usr/bin/python

from meda_utils import LearningRateSchedule, showIsGPU
import sys, argparse
import os
import time
import datetime
import pickle
from matplotlib.pyplot import pause
import matplotlib.animation as animation
import stable_baselines


def main(args):    
    t_total_s = time.time()
    expSeveralRuns(args)
    t_total_e = time.time()
    t_total = t_total_e - t_total_s
    print("| Finished training in %26s %4d min %2d sec |" 
          % (" ",t_total//60, t_total%60))
    print("+=================================================================+")
    print(getTimeStamp())
    print(" ")
    
    return


def getTimeStamp():
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d_%H%M')


def legacyReward(env, b_path = False):
    """ Return the reward of a game if a legacy
        method is used
    """
    router = OldRouter(env)
    return router.getReward(b_path)


def EvaluatePolicy(model:stable_baselines.ppo2.PPO2, env,
        n_eval_episodes = 100, b_path = False, render=False,
        b_stable=False):
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
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, _info = env.step(action)
        episode_reward += reward
        # [TODO] Implement render option for vectorized environments
        if render:
            env.envs[0].render()
            # time.sleep(0.001)
        for env_id in range(env.num_envs):
            if done[env_id]:
                # If environment is done, create a record
                episode_rewards[episode_count] = episode_reward[env_id]
                reached_goal[episode_count] = _info[env_id]["b_at_goal"]
                num_cycles[episode_count] = _info[env_id]["num_cycles"]
                # Reset respective environment
                episode_reward[env_id] = 0
                # Check for errors
                if b_stable and reached_goal[episode_count] < 100:
                    print("Failed at %03d-%d" % (episode_count,env_id))
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
    obs = env.reset() # Resets environment
    return mean_reward, mean_legacy, mean_goal, mean_cycles


def plotAgentPerformance(a_rewards, a_goals, a_cycles, str_size,
                         str_filename, b_path = False):
    # a_rewards = np.array(a_rewards)
    # a_goals = np.array(a_goals)
    # a_cycles = np.array(a_cycles)
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
        plt.rcParams.update({'font.size': 10, 'figure.figsize': (6,4)})
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
        plt.title("MEDA " + str_size)
        plt.xlabel('Training Epochs')
        if b_path:
            plt.ylabel('No. Cycles')
        else:
            plt.ylabel('Score')
        plt.tight_layout()
        # Save PNG
        plt.savefig('log/' + str_filename + '.png')
        # Save TEX
        import tikzplotlib
        tikzplotlib.clean_figure()
        tikzplotlib.save('log/' + str_filename + '.tex')
        
    return


def runAnExperiment(env, model=None, n_epochs=50, n_total_timesteps=20000,
                    n_policysteps=128, b_path=False, exp_id=0, n_evals=100):
    """Run single experiment consisting of a number of episodes
    """
    lr_schedule = LearningRateSchedule(base_rate=2.5e-4)
    if model is None:
        model = PPO2(policy=MyCnnPolicy, env=env, n_steps=n_policysteps,
                     learning_rate=lr_schedule.value,
                     verbose=0)

    agent_rewards, old_rewards, episodes = [], [], []
    episode_times, reach_goals, mean_cycles = [], [], []
    b_stable = False
    
    print("+-----------------------------------------------------------------+")
    #     "INFO: Epoch   0    4/  30 sec   -58.630 rew   49.0 suc   49.4 cyc")
    for i in range(n_epochs):
        print("|",end=""), sys.stdout.flush()
        t2s = time.time()
        # print("INFO: Epoch %2d started" % i)
        
        # Learn
        model.learn(total_timesteps=n_total_timesteps)
        
        t_eval_s = time.time()
        
        # Evaluate
        mean_reward, legacy_reward, reach_goal, mean_cycle = \
            EvaluatePolicy(model, model.get_env(), n_eval_episodes=n_evals,
                           b_path=b_path, render=False, b_stable=b_stable)
        if reach_goal == 100:
            b_stable = True
            lr_schedule.base_rate = lr_schedule.base_rate * 0.1
            # model.learning_rate = lr_schedule
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
        print(" Exp/Epoc %02d-%03d" % (exp_id,i) + 
              "  %02d/%03d sec"   % (t_eval,t2d)  + 
              "  %8.3f rew" % mean_reward +
              "  %5.1f suc" % reach_goal + 
              "  %5.1f cyc" % mean_cycle +
              " |")
    print("+-----------------------------------------------------------------+")
    agent_rewards = agent_rewards[-n_epochs:]
    # old_rewards = old_rewards[-n_epochs:]
    episodes = episodes[:n_epochs]
    return agent_rewards, old_rewards, episodes, reach_goals, mean_cycles, model


def runTest(env, model=None, n_epochs=50, exp_id=0, render=False,
            n_eval_episodes=300):
    """Run test cases
    """
    if model is None:
        print("ERROR: No model found. Aborting...")
        return

    # agent_rewards, old_rewards, episodes = [], [], []
    # episode_times, reach_goals, mean_cycles = [], [], []
    
    n_eval_episodes = 100
    
    print("+-----------------------------------------------------------------+")
    #     "INFO: Epoch   0    4/  30 sec   -58.630 rew   49.0 suc   49.4 cyc")
    for i in range(n_epochs):
        print("|",end=""), sys.stdout.flush()
        t2s = time.time()
        t_eval_s = time.time()

        obs = env.reset()
        episode_reward = np.zeros(env.num_envs)
        episode_count = 0
        for env_idx in range(env.num_envs):
            obs[env_idx] = env.envs[env_idx].setState(
                 dr_s=np.array([2,2,6,6]),
                 dr_g=np.array([20,10,24,14]),
                 m_taus=np.ones_like(env.envs[env_idx].m_taus)*0.7,
                 m_c1s=np.zeros_like(env.envs[env_idx].m_C1s),
                 m_c2s=np.ones_like(env.envs[env_idx].m_C2s)*300,
                 m_actcount=np.zeros_like(env.envs[env_idx].m_actcount)
            )
        while (episode_count < n_eval_episodes):
            # Make predictions and take action on all envs
            action, state = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            # [TODO] Implement render option for vectorized environments
            if render:
                env.envs[0].render(mode='human_frame')
                # time.sleep(0.001)
            for env_id in range(env.num_envs):
                if done[env_id]:
                    # If environment is done, create a record
                    episode_rewards[episode_count] = episode_reward[env_id]
                    reached_goal[episode_count] = _info[env_id]["b_at_goal"]
                    num_cycles[episode_count] = _info[env_id]["num_cycles"]
                    
                    # Reset respective environment
                    episode_reward[env_id] = 0
                    episode_reward[env_id] = 0
                    # Increment no. of episodes, break if no more episodes are needed
                    episode_count+=1
                    if episode_count >= n_eval_episodes: break
        
        t_eval_e = time.time()
        t_eval = t_eval_e - t_eval_s
        # agent_rewards.append(mean_reward)
        # old_rewards.append(legacy_reward)
        # episodes.append(i)
        t2e = time.time()
        # episode_times.append(t2e-t2s)
        # reach_goals.append(reach_goal)
        # mean_cycles.append(mean_cycle)
        t2d = t2e-t2s
        # print("INFO: Epoch %2d ended in %d seconds" % (i,t2e-t2s))
        print(" Exp/Epoc %02d-%03d" % (exp_id,i) + 
              "  %02d/%03d sec"   % (t_eval,t2d)  + 
              "  %8.3f rew" % 0 +
              "  %5.1f suc" % 0 + 
              "  %5.1f cyc" % 0 +
              " |")
    print("+-----------------------------------------------------------------+")
    # agent_rewards = agent_rewards[-n_epochs:]
    # old_rewards = old_rewards[-n_epochs:]
    # episodes = episodes[:n_epochs]
    return


# def expSeveralRuns(args, n_envs, n_policysteps, n_experiments):
def expSeveralRuns(args):
    """Run multiple experiments and plot agent performance in one plot
    """
    
    # Load configuration
    n_policysteps = args.n_policysteps
    n_experiments = args.n_exps
    n_envs = args.n_envs
    n_epochs = args.n_epochs
    n_total_timesteps = args.n_total_timesteps
    n_evals = args.n_evals
    b_save_model = args.b_save_model
    str_load_model = args.s_load_model
    str_model_name = args.s_model_name
    str_suffix = args.s_suffix
    str_mode = args.s_mode
    
    # Configure GPU settings, make environment, and report GPU status
    if sys.platform != 'win32':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0,allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    if int(str(args.verbose)[1]) > 0:    
        log_dir = "/home/elfar/MEDA/a_log/"
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None
    
    env = make_vec_env(MEDAEnv,
                       monitor_dir=log_dir,
                       wrapper_class=None,n_envs=n_envs,env_kwargs=vars(args))
    
    showIsGPU(tf)
    
    # Initialize agent and old rewards
    a_rewards, a_goals, o_rewards, a_cycles = [], [], [], []
    if str_load_model != '':
        loaded_model = PPO2.load('data/' + str_load_model)
        loaded_model.set_env(env)
    else:
        loaded_model = None
        
    str_size = str(args.size[0]).zfill(3) + 'x' + str(args.size[1]).zfill(3)
    str_env_info = str_size + '_E' + str(n_epochs).zfill(3)
    str_filename = str_model_name + '_' + str_env_info + '_' + str_suffix
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if str_mode == 'train':
        print("\n\n### TRAIN ### " + getTimeStamp() + " ### " + args.s_description)
        print("+=================================================================+")
        print("|            ID          Time       Rewards    Success     Cycles |")
        # Run Experiments
        for i in range(n_experiments):
            a_r, o_r, episodes, a_g, a_c, model = runAnExperiment(
                env, model=loaded_model, n_epochs=n_epochs,
                n_total_timesteps=n_total_timesteps, n_policysteps=n_policysteps,
                exp_id=i, n_evals=n_evals
            )
            a_rewards.append(a_r)
            a_goals.append(a_g)
            o_rewards.append(o_r)
            a_cycles.append(a_c)
            if b_save_model:
                # model.save("data/model_%s" % getTimeStamp())
                model.save("data/%s" % (str_filename + "_" + str(i).zfill(2)))
                print("| Model saved as  %47s |" 
                    % ("./data/" + str_filename + "_" + str(i).zfill(2) + ".zip"))
            
        data_obj = {
            'str_filename': str_filename,
            'a_rewards': a_rewards,
            'a_goals': a_goals,
            'a_cycles': a_cycles,
            'args': args
        }
        with open("data/%s.pickle" % str_filename, 'wb') as f:
            pickle.dump(data_obj, f)
        print("| Data saved as   %47s |" % ("./data/"+str_filename+".pickle"))
        plotAgentPerformance(a_rewards, a_goals, a_cycles, str_size, str_filename)
        print("| Figure saved as %47s |" % ("./log/"+str_filename+".png"))
        print("| Tikz saved as   %47s |" % ("./log/"+str_filename+".tex"))
        print("+-----------------------------------------------------------------+")
    
    elif str_mode=='test':
        print("### TEST ### " + getTimeStamp() + " ###")
        
        for i in range(n_experiments):
            runTest(env, model=loaded_model, n_epochs=n_epochs, exp_id=i,
                    render=True)
    
    else:
        print('ERROR: Unknown mode. Aborting...')        
            

    return


if __name__ == '__main__':
    
    # List of args default values
    def_args = {
        'seed':              -1,
        'verbose':           '00', # (TF|Monitor)
        's_mode':            'train', # train | test
        's_model_name':      '0816a',
        'size':              (30,30),
        'obs_size':          (30,30),
        's_description':     's30/o30 Fix SG, R E, Col., LR Sc, NPS4, TT14 x',
        'droplet_sizes':     [[4,4],[5,4],[5,5],[6,5],[6,6],],
        'n_envs':            8,
        'n_policysteps':     8, # [NOTE][2021-07-25] Was 32
        'n_exps':            1,
        'n_epochs':          50,
        'n_total_timesteps': 2**14,
        'b_save_model':      True,
        's_suffix':          'NPS04', #'T30V300TL_D22',#'T30V300TL_D12', #'T30V300TL_D23', #'T30V300TL_D12', # T30V300TL_D22
        's_load_model':      '',#'0727a_030x030_E010_NPS32_00', #'0726b_030x030_E050_NPS16_00', #'TMP_E_030x030_E040__00',#'TMP_D_030x030_E025_T30V300TL_D22_00', #'MDL_C_090x090_E025_T30V300TL_D12_00',#'MDL_C_090x090_E025_T30V300TL60_00',#'MDL_C_060x060_E025_T30V300_00',#'MDL_C_060x060_E025_T30V300TL_D22_00', # 'MDL_C_060x060_E025_T30V300_00', # 'MDL_C_030x030_E025_T30V300TL_D12_00', # MDL_C_030x030_E031_S30V300_00 MDL_A_030x030_E101_NS30_00 TMP_B_030x030_E005_S30V300_00
        'b_play_mode':       False,
        'deg_mode':          'normal',
        'deg_perc':          0.2,
        'deg_size':          2,
        'n_evals':           500
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
    parser.add_argument('--n-evals',type=int,default=def_args['n_evals'])
    parser.add_argument('--s-description',type=str,default=def_args['s_description'])
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.verbose)[0]
    import warnings
    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r"The name")
    import numpy as np
    np.set_printoptions(linewidth=np.inf)
    # import matplotlib
    import matplotlib.pyplot as plt
    import tensorflow as tf
    # from utils import OldRouter
    from my_net import MyCnnPolicy
    # from envs.dmfb import *
    from envs.meda import *
    from stable_baselines.common import make_vec_env # tf_util
    # from stable_baselines.common.vec_env import DummyVecEnv
    # from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy
    from stable_baselines.common.evaluation import evaluate_policy
    from stable_baselines import PPO2
    
    if int(str(args.verbose)[1]) > 0:
        from stable_baselines.bench.monitor import Monitor, load_results
        from stable_baselines import results_plotter
    # Set random seeds before starting for SB, Random, tensorflow, numpy, gym
    if args.seed >= 0:
        from stable_baselines.common.misc_util import set_global_seeds
        set_global_seeds(args.seed)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    main(args)