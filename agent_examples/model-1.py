"""
Simple reinforcement learning player based on
https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0

"""

import gym
import numpy as np
import pickle
import gym_racecar

def main():
    env = gym.make("racecar-v0")
    env.render()

    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
            total_reward += r
            if steps % 200 == 0 or done:
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()

main()