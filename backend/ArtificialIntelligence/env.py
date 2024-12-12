import yfinance as yf
import gymnasium as gym
import numpy as np


class BuySell(gym.Env):
    # --- TRADING ENVIRONMENT --- #
    # This will be a buy or sell type of environment so 2 discrete actions

    def __init__(self):

        self.action_space = gym.spaces.Discrete(2) # 0: sell, 1: buy
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, sha

        