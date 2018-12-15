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
        self.gamma         = 0.9
        self.epsilon       = 0.1
        self.Q             = self._initialiseModel()

    def _initialiseModel(self):
        # Need to initialise each state action pair as 0
        # Q should be an array of structure:
        # [[a1, a2, a3, a4], -- State 0 ([0][0])
        #  [a1, a2, a3, a4], -- State 1 ([0][1]) etc
        Q = np.zeros((stateCnt,actionCnt))

        return Q


    def predict_value(self, s):
        # Simply return the relevant row from Q
        return self.Q[s]


    def update_value_Qlearning(self, s,a,r,s_next, terminalStateNotReached):
        # Update if s is not a terminal state
        if terminalStateNotReached:
            self.Q[s][a] = self.Q[s][a] + self.learning_rate*(r - self.Q[s][a] + self.gamma*np.amax(self.Q[s_next]))

        # Update if s is any type of terminal state
        else:
            self.Q[s][a] = self.Q[s][a] + self.learning_rate*(r - self.Q[s][a])


    def update_value_SARSA(self, s,a,r,s_next, a_next, terminalStateNotReached):
        # Update if s is not a terminal state
        if terminalStateNotReached:
            self.Q[s][a] = self.Q[s][a] + self.learning_rate*(r - self.Q[s][a] + self.gamma*self.Q[s_next][a_next]))

        # Update if s is any type of terminal state
        else:
            self.Q[s][a] = self.Q[s][a] + self.learning_rate*(r - self.Q[s][a])


    def choose_action(self, s):
        # Random float between 0 and 1
        # If greater than epsilon value, choose optimal action
        if random.random() > self.epsilon:
            a = np.amax(self.Q[s])
        # If less than or equal to epsilon, choose any random action (inc. optimal)
        else:
            a = random.randrange(0,4,1)
        return a


    def updateEpsilon(self, episodeCounter):


class World:
    def __init__(self, env):
        self.env = env
        print('Environment has %d states and %d actions.' % (self.env.observation_space.n, self.env.action_space.n))
        self.stateCnt           = self.env.observation_space.n
        self.actionCnt          = self.env.action_space.n
        self.maxStepsPerEpisode = 20
        self.q_Sinit_progress   = agent.Q[0] # ex: np.array([[0,0,0,0]])

    def run_episode_qlearning(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False

        for i in range(self.maxStepsPerEpisode):
            # Take step to the next state
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached, and some diagnostic information useful for debugging.
            s_prev = s
            a = agent.choose_action(s)
            s, r, done = self.env.step(a)

            # Update value function
            agent.update_value_Qlearning(s_prev,a,r,s,!done)

            # Print the current environment state
            self.env.render()

            # Break if terminal state reached
            if done == True:
                break
            else:

            r_total += r
            episodeStepsCnt += 1
        # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress throughout training episodes for all available actions at the initial state.
        self.q_Sinit_progress = np.append(agent.predict_value(0))

        return r_total, episodeStepsCnt

    def run_episode_sarsa(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False

        for i in range(self.maxStepsPerEpisode):
            # Take step to the next states
            s_prev = s
            a = agent.choose_action(s)
            s, r, done = self.env.step(a)
            a_next = agent.choose_action(s)

            # Update value function
            agent.update_value_SARSA(s_prev,a,r,s,a_next, !done)

            # Print the current environment state
            self.env.render()

            # Break if terminal state reached
            if done == True:
                break
            else:

            r_total += r
            episodeStepsCnt += 1

        self.q_Sinit_progress = np.append(agent.predict_value(0))
        
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
    nbOfTrainingEpisodes     = 50
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
