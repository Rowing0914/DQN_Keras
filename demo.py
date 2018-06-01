import gym
from keras.models import load_model
from dqn import Agent

env_name = 'CartPole-v0'
eps = 0.8
episodes = 5
env = gym.make(env_name)
model = load_model('./model/my_model.h5')
agent = Agent(env)


for episode in range(episodes):
	# initial state
	s = env.reset()

	done = False
	while not done:
	    for i in range(50):
	        a = agent.act(s, eps)
	        env.render(a)
	        s2, r, done, info = env.step(a)
	        s = s2
env.close()