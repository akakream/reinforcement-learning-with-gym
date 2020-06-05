import gym
import numpy as np

env = gym.make('MountainCar-v0')

'''
https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
'''

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

# Give me some feedback every ... episodes
SHOW_EVERY = 2000

print(env.observation_space.high) #[0.6 0.07]
print(env.observation_space.low)  #[-1.2 -0.07]
print(env.action_space.n) #3

# Creates [20,20] size for 400, 20 position * 20 velocity, discrete observation values
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# [0.09 0.007] --> step size for every position and velocity value
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

# probability of taking a random action
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# The reward for the unsuccessful actions is -1. low=-2,high=0 --> makes an average of -1  	
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape) #(20,20,3)


def get_discrete_state(state):
	''' Extracts the discrete state out of non discrete one '''
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):

	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False	

	# This is the intial state of the car
	discrete_state = get_discrete_state(env.reset())
	#print(discrete_state) # (7,10)

	# Starting q-values for the initial state
	#print(q_table[discrete_state]) # [-1.13042324 -0.41334632 -0.29485741]
	# Get the maximum q-value of the state
	#print(np.argmax(q_table[discrete_state])) # 2

	done = False

	while not done:

		if np.random.random() > epsilon:
			# Choose the action with the maximum q-value
			action = np.argmax(q_table[discrete_state])
		else:
			# Explore == take a random action
			action = np.random.randint(low=0, high=env.action_space.n)
		# Take a step
		new_state, reward, done, _ = env.step(action)	
		# Get the discrete state, because what you will get will be poooooooointy
		new_discrete_state = get_discrete_state(new_state)
		# Render every (SHOW_EVERY) episodes
		if render:
			env.render()

		if not done:
			# Q-LEARNING STUFF
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]

			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q
		elif new_state[0] >= env.goal_position:
			print(f'We made it on episode {episode}')
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state		

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

env.close()	