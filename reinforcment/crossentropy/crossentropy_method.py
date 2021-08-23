import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Based on https://colab.research.google.com/github/yandexdataschool/Practical_RL/blob/coursera/week1_intro/crossentropy_method.ipynb
class CrossEntropyMethod():
    """
    Train a policy using using the cross entropy method

    :param env: gym environment
    :param session_generator_fn: a function that generates sessions
    :param n_states: plot the progress of the training
    :param n_actions: number of training iterations (how many times the policy should be updated)
    :param t_max: the maximal time a session can last
    :param n_sessions: how many sessions to generate
    :param percentile: take this percent of session with highest rewards
    :param learning_rate: how quickly the policy is updated, on a scale from 0 to 1
    """
    def __init__(self,env,session_generator_fn,n_states, n_actions) -> None:
        self.n_states=n_states
        self.n_actions=n_actions
        self.session_generator_fn=session_generator_fn
        self.env=env
        env.reset()

    def select_elites(self,states_batch, actions_batch, rewards_batch, percentile):
        """
        Select states and actions from games that have rewards >= percentile
        :param states_batch: list of lists of states, states_batch[session_i][t]
        :param actions_batch: list of lists of actions, actions_batch[session_i][t]
        :param rewards_batch: list of rewards, rewards_batch[session_i]

        :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
        """

        reward_threshold = np.percentile(a=rewards_batch,q=percentile)
        elite_idx=np.where(rewards_batch>=reward_threshold)[0]
        elite_states = np.concatenate(np.array(states_batch)[elite_idx])
        elite_actions = np.concatenate(np.array(actions_batch)[elite_idx])

        return elite_states, elite_actions
    
    def initialize_policy(self):
        """
        Initialize the policy using an uniform distribution (1/n_actions for all states).
        """
        return np.ones((self.n_states,self.n_actions))/self.n_actions

    def get_new_policy(self,elite_states, elite_actions):
        """
        Given a list of elite states/actions from select_elites,
        return a new policy where each action probability is proportional to

            policy[s_i,a_i] ~ #[occurrences of s_i and a_i in elite states/actions]

        :param elite_states: 1D list of states from elite sessions
        :param elite_actions: 1D list of actions from elite sessions

        """
        elite_states=np.array(elite_states)
        elite_actions=np.array(elite_actions)
        new_policy = self.initialize_policy()
        for state in range(self.n_states):
            state_index=np.where(elite_states==state)[0]
            if state_index.shape[0]>0:
                for action in range(self.n_actions):
                    action_index=np.where(elite_actions[state_index]==action)[0]
                    new_policy[state,action]=action_index.shape[0]/state_index.shape[0]
        return new_policy

    def train(self,show_progress=True,epochs=100,t_max=1000,n_sessions = 250,percentile = 30,learning_rate = 0.5):
        """
        Train a policy using the provided environment using the cross entropy method

            policy[s_i,a_i] ~ #[occurrences of s_i and a_i in elite states/actions]

        :param show_progress: plot the progress of the training
        :param epochs: number of training iterations (how many times the policy should be updated)
        :param t_max: the maximal time a session can last
        :param n_sessions: how many sessions to generate
        :param percentile: take this percent of session with highest rewards
        :param learning_rate: how quickly the policy is updated, on a scale from 0 to 1
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
        self.policy = self.initialize_policy()

        for _ in range(epochs):
            sessions = [self.session_generator_fn(self.env, self.policy, t_max) for _ in range(n_sessions)]

            states_batch, actions_batch, rewards_batch = zip(*sessions)

            elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)

            new_policy = self.get_new_policy(elite_states, elite_actions)

            self.policy = learning_rate * new_policy + (1 - learning_rate) * self.policy
            if show_progress:
                plot_progress(rewards_batch, log, percentile)

    def evaluate_policy(self,t_max=1000,n_sessions = 250):
        if self.policy is None:
            raise ValueError('To evaluate a policy, it should be trained first using the train() method')
        sessions = [self.session_generator_fn(self.env, self.policy, t_max) for _ in range(n_sessions)]

        return sessions
    
    def render_policy(self,t_max=1000):
        if self.policy is None:
            raise ValueError('To evaluate a policy, it should be trained first using the train() method')
        self.session_generator_fn(self.env, self.policy, t_max,True)