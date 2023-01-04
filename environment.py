import numpy as np
import random


class BlackJack:
    def __init__(self, mode="normal"):
        self.p = 0
        self.d = 0
        self.cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.player_ace_n = 0
        self.dealer_ace_n = 0
        self.terminated = False
        self.mode = mode
        self.action_space_n = len(self.get_actions())
        self.base_reward = 1
        np.random.seed(27)

    def get_state(self):
        return (self.p, self.d, self.player_ace_n > 0)

    def get_actions(self):
        if self.mode == "normal":
            return [0, 1]
        else:
            return [0, 1, 2]

    def reset(self):
        self.dealer_ace_n = 0
        self.player_ace_n = 0
        self.p = self.draw("p")
        self.p += self.draw("p")
        self.is_bust()
        self.d = self.draw("d")
        self.terminated = False
        self.base_reward = 1
        return self.get_state()

    def draw(self, participant="p"):
        value = random.choice(self.cards)
        if value == 11:
            if participant == "p":
                self.player_ace_n += 1
            else:
                self.dealer_ace_n += 1
        return value

    def is_bust(self, participant="p"):
        if participant == "p":
            if self.p > 21:
                if self.player_ace_n > 0:
                    self.p -= 10
                    self.player_ace_n -= 1
                    return False
                else:
                    return True
        if participant == "d":
            if self.d > 21:
                if self.dealer_ace_n > 0:
                    self.d -= 10
                    self.dealer_ace_n -= 1
                    return False
                else:
                    return True
        return False

    def step(self, action):
        reward = 0
        if action == 1 or action == 2:
            if action == 2:
                self.base_reward *= 2
            self.p += self.draw("p")
            if self.is_bust("p"):
                self.terminated = True
                reward = -self.base_reward
            else:
                self.terminated = False
                reward = 0

        elif action == 0:
            self.terminated = True
            while self.d < 17 and self.d <= self.p:
                self.d += self.draw("d")
            if self.p == self.d:
                reward = 0
            elif self.p > self.d or self.is_bust("d"):
                reward = self.base_reward
            elif self.p < self.d <= 21:
                reward = -self.base_reward
        #print("State ", self.get_state() , " Terminated ", self.terminated, " Reward ", reward, " action", action)
        return self.get_state(), self.terminated, reward
