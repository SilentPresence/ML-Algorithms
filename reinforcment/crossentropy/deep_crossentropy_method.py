import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Based on https://colab.research.google.com/github/yandexdataschool/Practical_RL/blob/coursera/week1_intro/deep_crossentropy_method.ipynb
class DeepCrossEntropyMethod():
    """
    Train a policy using using the cross entropy method while using a deep classifier

    :param env: gym environment
    :param session_generator_fn: a function that generates sessions
    :param agent_factory_fn: a function that creates an deep learning agent, must implement partial_fit and predict
    """
    def __init__(self,env,agent_factory_fn,session_generator_fn) -> None:
        self.session_generator_fn=session_generator_fn
        env.reset()
        self.env=env
        self.agent_factory_fn=agent_factory_fn

    def select_elites(self,states_batch, actions_batch, rewards_batch, percentile):
        """
        Select states and actions from games that have rewards >= percentile
        :param states_batch: list of lists of states, states_batch[session_i][t]
        :param actions_batch: list of lists of actions, actions_batch[session_i][t]
        :param rewards_batch: list of rewards, rewards_batch[session_i]

        :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
        """

        reward_threshold = np.percentile(a=rewards_batch,q=percentile)
        elite_idx=np.where(rewards_batch>reward_threshold)[0]
        if(elite_idx.shape[0]==0):
            elite_idx=np.where(rewards_batch>=reward_threshold)[0]
        elite_states = np.concatenate(np.array(states_batch)[elite_idx])
        elite_actions = np.concatenate(np.array(actions_batch)[elite_idx])

        return elite_states, elite_actions

    def train(self,mean_reward_stop,show_progress=True,epochs=100,t_max=1000,n_sessions = 250,percentile = 30):
        """
        Train a policy using the provided environment using the cross entropy method

        :param show_progress: plot the progress of the training
        :param epochs: number of training iterations (how many times the policy should be updated)
        :param t_max: the maximal time a session can last
        :param n_sessions: how many sessions to generate
        :param percentile: take this percent of session with highest rewards
        """
        def plot_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
            mean_reward = np.mean(rewards_batch)
            threshold = np.percentile(rewards_batch, percentile)
            log.append([mean_reward, threshold])
            
            plt.figure(figsize=[8, 4])
            plt.subplot(1, 2, 1)
            plt.plot(list(zip(*log))[0], label='Mean rewards')
            plt.plot(list(zip(*log))[1], label='Reward thresholds')
            plt.legend()
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.hist(rewards_batch, range=reward_range)
            plt.vlines([np.percentile(rewards_batch, percentile)],
                    [0], [100], label="percentile", color='red')
            plt.legend()
            plt.grid()
            clear_output(True)
            print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
            plt.show()

        log = []
        # reset policy just in case
        self.agent=self.agent_factory_fn()
        for _ in range(epochs):
            sessions = [self.session_generator_fn(self.env, self.agent, t_max) for _ in range(n_sessions)]
            states_batch, actions_batch, rewards_batch = zip(*sessions)
            elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
            self.agent.partial_fit(elite_states,elite_actions)
            if show_progress:
                plot_progress(rewards_batch, log, percentile)
            if np.mean(rewards_batch)>mean_reward_stop:
                break


    def evaluate_policy(self,t_max=1000,n_sessions = 250):
        if self.agent is None:
            raise ValueError('To evaluate a policy, it should be trained first using the train() method')
        sessions = [self.session_generator_fn(self.env, self.agent, t_max) for _ in range(n_sessions)]

        return sessions
    
    def render_policy(self,t_max=1000):
        if self.agent is None:
            raise ValueError('To evaluate a policy, it should be trained first using the train() method')
        self.session_generator_fn(self.env, self.agent, t_max,True)