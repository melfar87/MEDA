#!/usr/bin/python

import sys, argparse
import os
import time
import datetime

# import matplotlib
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from utils import OldRouter
# from my_net import MyCnnPolicy
# from envs.dmfb import *
# from envs.meda import *

# from stable_baselines.common import make_vec_env, tf_util
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines import PPO2


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
    episode_rewards = []
    legacy_rewards = []
    reached_goal = []
    num_cycles = []
    n_steps = 0
    for i in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        this_loop_steps = 0
        while not done:
            action, state = model.predict(obs)
            obs, reward, done, _info = env.step(action)
            if render:
                env.envs[0].render()
                # time.sleep(0.001)
            b_at_goal = _info[0]["b_at_goal"]
            # cycles = _info[0]["num_cycles"]
            reward = reward[0]
            done = done[0]
            episode_reward += reward
            n_steps += 1
            this_loop_steps += 1
            legacy_r = legacyReward(env.envs[0], b_path)
        if b_path:
            episode_rewards.append(this_loop_steps)
        else:
            episode_rewards.append(episode_reward)
        legacy_rewards.append(legacy_r)
        reached_goal.append(b_at_goal)
        num_cycles.append(this_loop_steps)
        # num_cycles.append(cycles)
    mean_reward = np.mean(episode_rewards)
    mean_legacy = np.mean(legacy_rewards)
    mean_goal   = np.mean(reached_goal)
    mean_cycles = np.mean(num_cycles)
    return mean_reward, n_steps, mean_legacy, mean_goal, mean_cycles


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
    o_rewards = np.array(o_rewards)
    a_line = np.average(a_rewards, axis = 0)
    a_goals_line = np.average(a_goals, axis = 0)
    a_cycles_line = np.average(a_cycles, axis = 0)
    o_line = np.average(o_rewards, axis = 0)
    a_max = np.max(a_rewards, axis = 0)
    a_min = np.min(a_rewards, axis = 0)
    o_max = np.max(o_rewards, axis = 0)
    o_min = np.min(o_rewards, axis = 0)
    episodes = list(range(len(a_max)))
    with plt.style.context('seaborn-paper'):
        plt.rcParams.update({'font.size': 10, 'figure.figsize': (4,3)})
        plt.figure()
        plt.fill_between(episodes, a_max, a_min, facecolor = 'red', alpha = 0.3)
        plt.fill_between(episodes, o_max, o_min, facecolor = 'blue',
                alpha = 0.3)
        plt.plot(episodes, a_line, 'r-', label = 'Agent')
        plt.plot(episodes, a_goals_line, 'g.', label = 'Success Rate')
        plt.plot(episodes, a_cycles_line, 'k-.', label = 'No. Cycles')
        plt.plot(episodes, o_line, 'b-', label = 'Baseline')
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
    for i in range(n_epochs):
        t2s = time.time()
        print("INFO: Epoch %2d started" % i)
        model.learn(total_timesteps = n_steps)
        mean_reward, n_steps, legacy_reward, reach_goal, mean_cycle = \
            EvaluatePolicy(model, model.get_env(), n_eval_episodes = 100,
                           b_path=b_path, render=False)
        agent_rewards.append(mean_reward)
        old_rewards.append(legacy_reward)
        episodes.append(i)
        t2e = time.time()
        episode_times.append(t2e-t2s)
        reach_goals.append(reach_goal)
        mean_cycles.append(mean_cycle)
        print("INFO: Epoch %2d ended in %d seconds" % (i,t2e-t2s))
    agent_rewards = agent_rewards[-n_epochs:]
    old_rewards = old_rewards[-n_epochs:]
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
            model.save("data/model_%s" % getTimeStamp())
            
    env_info = '_E' + str(n_epochs) + '_I'
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
        'verbose':  '1',
        'size':     (60,60),
        'droplet_sizes': [[4,4],],
        'n_envs':   8,
        'n_s':      64,
        'n_exps':   1,
        'n_epochs': 21,
        'n_steps': 20000,
        'b_save_model': True,
        's_model_name': 'model',
        's_load_model': ''
    }
    
    # Initialize parser
    parser = argparse.ArgumentParser(description='MEDA Training Module')
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
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose
    
    # import matplotlib
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from utils import OldRouter
    from my_net import MyCnnPolicy
    from envs.dmfb import *
    from envs.meda import *
    from stable_baselines.common import make_vec_env # tf_util
    # from stable_baselines.common.vec_env import DummyVecEnv
    # from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy
    # from stable_baselines.common.evaluation import evaluate_policy
    from stable_baselines import PPO2
    
    main(args)
