import numpy as np
import gym
import random

from collections import deque
from keras.layers import Input, Activation, Dense, Flatten, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

class Agent:
    def __init__(self, env):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.create_model()

    def create_model(self, hidden_dims=[64, 64]):
        X = Input(shape=(self.input_dim, ))

        net = RepeatVector(self.input_dim)(X)
        net = Reshape([self.input_dim, self.input_dim, 1])(net)

        for h_dim in hidden_dims:
            net = Conv2D(h_dim, [3, 3], padding='SAME')(net)
            net = Activation('relu')(net)

        net = Flatten()(net)
        net = Dense(self.output_dim)(net)

        self.model = Model(inputs=X, outputs=net)

        # save the model architecture to png
        plot_model(self.model, to_file='model.png')

        self.model.compile('adam', 'mse')

    def act(self, X, eps=1.0):
        if np.random.rand() < eps:
            return self.env.action_space.sample()

        X = X.reshape(-1, self.input_dim)
        Q = self.model.predict_on_batch(X)
        return np.argmax(Q, 1)[0]

    def train(self, X_batch, y_batch):
        return self.model.train_on_batch(X_batch, y_batch)

    def predict(self, X_batch):
        return self.model.predict_on_batch(X_batch)

    def save_model(self):
        # creates a HDF5 file 'my_model.h5'
        self.model.save('my_model.h5')
        pass


def create_batch(agent, memory, batch_size, discount_rate):
    sample = random.sample(memory, batch_size)
    sample = np.asarray(sample)

    s = sample[:, 0]
    a = sample[:, 1].astype(np.int8)
    r = sample[:, 2]
    s2 = sample[:, 3]
    d = sample[:, 4] * 1.

    X_batch = np.vstack(s)
    y_batch = agent.predict(X_batch)

    y_batch[np.arange(batch_size), a] = r + discount_rate * np.max(agent.predict(np.vstack(s2)), 1) * (1 - d)

    return X_batch, y_batch


def print_info(episode, reward, eps):
    msg = f"[Episode {episode:>5}] Reward: {reward:>5} EPS: {eps:>3.2f}"
    print(msg)


def train(n_episode, discount_rate, n_memory, batch_size, eps, min_eps, env_name, env, agent, memory):
    # CartPole-v0 Clear Condition
    # Average reward per episode > 195.0 over 100 episodes
    LAST_100_GAME_EPISODE_REWARDS = deque()

    for episode in range(n_episode):
        done = False
        s = env.reset()
        eps = max(min_eps, eps - 1/(n_episode/2))
        episode_reward = 0
        while not done:
            a = agent.act(s, eps)
            s2, r, done, info = env.step(a)
            episode_reward += r

            if done and episode_reward < 200:
                r = -100

            memory.append([s, a, r, s2, done])

            if len(memory) > n_memory:
                memory.popleft()

            if len(memory) > batch_size:
                X_batch, y_batch = create_batch(agent, memory, batch_size, discount_rate)
                agent.train(X_batch, y_batch)

            s = s2

        print_info(episode, episode_reward, eps)
        LAST_100_GAME_EPISODE_REWARDS.append(episode_reward)
        if len(LAST_100_GAME_EPISODE_REWARDS) > 100:
            LAST_100_GAME_EPISODE_REWARDS.popleft()

        if np.mean(LAST_100_GAME_EPISODE_REWARDS) >= 195.0:
            print(f"Game solved in {episode + 1} with average reward {np.mean(LAST_100_GAME_EPISODE_REWARDS)}")

    env.close()

    # save model
    agent.save_model()

    return agent, eps


def play(env, agent, eps):
    s = env.reset()
    done = False
    while not done:
        for i in range(50):
            a = agent.act(s, eps)
            env.render(a)
            s2, r, done, info = env.step(a)
            s = s2
    env.close()
    return "done"


if __name__ == '__main__':
    n_episode = 100
    discount_rate = 0.99
    n_memory = 50000
    batch_size = 32
    eps = 1.0
    min_eps = 0.1
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    agent = Agent(env)
    memory = deque()

    agent, opt_eps = train(n_episode, discount_rate, n_memory, batch_size, eps, min_eps, env_name, env, agent, memory)
    print(play(env, agent, opt_eps))