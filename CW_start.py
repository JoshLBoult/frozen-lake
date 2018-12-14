from __future__ import division
import gym
from gym.envs.registration import register
import numpy as np
import random, math, time
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
register(
    id          ='FrozenLakeNotSlippery-v0',
    entry_point ='gym.envs.toy_text:FrozenLakeEnv',
    kwargs      ={'map_name' : '8x8', 'is_slippery': False},
)

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class Agent:
    def __init__(self, env):
        self.stateCnt      = self.env.observation_space.n
        self.actionCnt     = self.env.action_space.n # left:0; down:1; right:2; up:3
        self.learning_rate = 0.1
        self.gamma         = 0.5
        self.epsilon       = 0.8
        self.Q             = self._initialiseModel()

    def _initialiseModel(self):

    def predict_value(self, s):
        # Get value at each state in four directions
        # Change s into row and column format
        row = s // self.env.ncol
        col = s % self.env.ncol
        # Initialise the array with zeros, as that is the most likely reward
        value_array = np.zeros(4)

        # Iterate through the outcome of all actions, updating the array if
        # the goal is reachable
        for i in range(actionCnt):
            new_state = self.env.inc(row, col, i)
            letter = desc[new_state]
            if letter in 'HSF':
                # do nothing, value already 0
            else:
                value_array[i] = 1

        return value_array

    def update_value_Qlearning(self, s,a,r,s_next, goalNotReached):
        # Simply add the value of s_next and gamma*(value of state after s_next
        # and action a)
        # Or q(k+1)(s,a) = q(k)(s,a) + lr*(rew(t+1) + "max_action" gamma*q(k)(s_next,a_next) - q(k)(s,a))
        # In this case estimate = (lr*?)(reward at s_next) + lr*(reward at second_state + gamma*(second_state from s_next and max_action - (lr*?)(reward at s_next)))
        # if in a terminal state with no s_next/s_nextnext
        value_estimate = 0
        row = s_next // self.env.ncol
        col = s_next % self.env.ncol
        second_state = self.env.inc(row, col, a)

        letter_s_next = desc[row, col]
        letter_second_state = desc[second_state]

        if letter_s_next in 'G':
            value_estimate += 1
        if letter_second_state in 'G':
            value_estimate += self.gamma

        return value_estimate

    def update_value_SARSA(self, s,a,r,s_next, a_next, goalNotReached):

    def choose_action(self, s):

    def updateEpsilon(self, episodeCounter):


class World:
    def __init__(self, env):
        self.env = env
        print('Environment has %d states and %d actions.' % (self.env.observation_space.n, self.env.action_space.n))
        self.stateCnt           = self.env.observation_space.n
        self.actionCnt          = self.env.action_space.n
        self.maxStepsPerEpisode =
        self.q_Sinit_progress   = # ex: np.array([[0,0,0,0]])

    def run_episode_qlearning(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            # self.env.step(a): "step" will execute action "a" at the current agent state and move the agent to the nect state.
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached, and some diagnostic information useful for debugging.
            # self.env.render(): "render" will print the current enviroment state.
            # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress throughout training episodes for all available actions at the initial state.
        return r_total, episodeStepsCnt

    def run_episode_sarsa(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            # self.env.step(a): "step" will execute action "a" at the current agent state and move the agent to the nect state.
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached, and some diagnostic information useful for debugging.
            # self.env.render(): "render" will print the current enviroment state.
            # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress throughout training episodes for all available actions at the initial state
        return r_total, episodeStepsCnt

    def run_evaluation_episode(self):
        agent.epsilon = 0
        return success


if __name__ == '__main__':
    env                      = gym.make('FrozenLakeNotSlippery-v0')
    world                    = World(env)
    agent                    = Agent(env) # This will creat an agent
    r_total_progress         = []
    episodeStepsCnt_progress = []
    nbOfTrainingEpisodes     =
    for i in range(nbOfEpisodes):
        print '\n========================\n   Episode: {}\n========================'.format(i)
        # run_episode_qlearning or run_episode_sarsa
        # append to r_total_progress and episodeStepsCnt_progress
    # run_evaluation_episode

    ### --- Plots --- ###
    # 1) plot world.q_Sinit_progress
    fig1 = plt.figure(1)
    plt.ion()
    plt.plot(world.q_Sinit_progress[:,0], label='left',  color = 'r')
    plt.plot(world.q_Sinit_progress[:,1], label='down',  color = 'g')
    plt.plot(world.q_Sinit_progress[:,2], label='right', color = 'b')
    plt.plot(world.q_Sinit_progress[:,3], label='up',    color = 'y')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop = fontP, loc=1)
    plt.pause(0.001)

    # 2) plot the evolution of the number of steps per successful episode throughout training. A successful episode is an episode where the agent reached the goal (i.e. not any terminal state)
    fig2 = plt.figure(2)
    plt1 = plt.subplot(1,2,1)
    plt1.set_title("Number of steps per successful episode")
    plt.ion()
    plt.plot(episodeStepsCnt_progress)
    plt.pause(0.0001)
    # 3) plot the evolution of the total collected rewards per episode throughout training. you can use the running_mean function to smooth the plot
    plt2 = plt.subplot(1,2,2)
    plt2.set_title("Rewards collected per episode")
    plt.ion()
    r_total_progress = running_mean(r_total_progress)
    plt.plot(r_total_progress)
    plt.pause(0.0001)
    ### --- ///// --- ###
