import numpy as np
import gym
from gym import wrappers
import random
from neural_network import Network
import matplotlib.pyplot as plt
from collections import deque


















#Network
MAX_EPISODES = 50
START_EPISODE = 1
MAX_TIME = 1000
NUM_POSSIBLE_ACTIONS = 50
BATCH_SIZE = 50
LAYERS = [3,12,1]
LEARNING_RATE = 0.001
L1=0
L2=0.1

#DQN
RANDOM_ACTION_PROBABBILITY = 0.01
GAMMA = 0.99
BUFFER_SIZE = 5000

#uonoise
IS_NOISE = True
EXPLORATION_EPISODES = 50
SIGMA_UON = 0.2
THETA_UON = 0.15

#начальное распределение весов сетки:
W_B_INIT_DISTRIBUTION = ('uniform',-0.003,0.003) #lower_bound and upper_bound
#W_B_INIT_DISTRIBUTION = ('normal',0,1) # mean and standart_deviation
#W_B_INIT_DISTRIBUTION = ('beta',4,4) # a and b

#рендерить ли ведео?
IS_VIDEO = False

#кривая обучения
REWARD_FILE = "reward1.txt"
NETWORK_CHANGE_FILE = "network1.txt"





















class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=12345):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, old_state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done) #новая запись в истории

        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        new_s_batch = np.array([_[3] for _ in batch])
        d_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, new_s_batch, d_batch


def argmaxQ(state):
	Q_to_actions_map = {}

	for possible_action in possible_actions:
		input_vector_representation = np.append(state, possible_action)[:, np.newaxis]
		predicted_Q = float(Q.feedforward(input_vector_representation))
		Q_to_actions_map[predicted_Q] = possible_action
	
	Qs=list(Q_to_actions_map.keys())
	highest_Q = max(Qs)
	
	return Q_to_actions_map[highest_Q]

def UONoise():
    state = 0
    while True:
        yield state
        state += -THETA_UON*state+SIGMA_UON*np.random.randn()





















#создаем буфер для истории
history_buffer = ReplayBuffer(BUFFER_SIZE)

#create a Neural Network
#l1 and l2 for regularization
Q=Network(LAYERS,output_function=lambda x: x, output_derivative=lambda x: 1,
			l1=L1, l2=L2, init_dist=W_B_INIT_DISTRIBUTION,flag=False)

#start an enviroment
env = gym.make('MountainCarContinuous-v0')

if IS_VIDEO:
	env = wrappers.Monitor(env, "./video", force=True, video_callable=lambda episode_id: True)

#дискретизируем пространство возможных действий
possible_actions = np.linspace(-1, 1, NUM_POSSIBLE_ACTIONS)

#шум
noise = UONoise()

#награды гля кривой обучения
rewards_curve = []

network_chages = []





















for episode in range(START_EPISODE,MAX_EPISODES+1):

	#начальное состояние машинки
	state = env.reset()

	#наша метрика, которую хотим улучшить
	episode_total_reward = 0

	episode_network_change = 0

	for episode_time in range(1,MAX_TIME+1):

		if np.random.rand() < RANDOM_ACTION_PROBABBILITY:
			action = np.random.rand()*2-1
		else:
			action = argmaxQ(state)


		if IS_NOISE and episode < EXPLORATION_EPISODES:
			p = episode/EXPLORATION_EPISODES
			action = action*p + (1-p)*next(noise)

		#выполняем действие
		new_state, reward, done, info = env.step([action])

		#записываем историю
		history_buffer.add(state, action, reward, new_state, done)

		# Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
		if history_buffer.size() >= BATCH_SIZE:

			#формируем батч
			state_batch, action_batch, reward_batch, new_state_batch, done_batch = history_buffer.sample_batch(BATCH_SIZE)

			#формируем ответы для батча
			y_train_batch = reward_batch
			for i, is_done in enumerate(done_batch):
				if not is_done:
					y_train_batch[i]+= GAMMA * argmaxQ(new_state_batch[i])

			#формируем обучающую выборку
			X_train_batch = np.concatenate((state_batch, action_batch[np.newaxis, :].T), axis=1)

			#тренируем, learning rate еще не делился на len(batch)
			train_data_batch = [(x[:, np.newaxis], y) for x, y in zip(X_train_batch, y_train_batch)]
			episode_network_change += Q.update_mini_batch(mini_batch=train_data_batch, eta=LEARNING_RATE)

		#накапливаем общую метрику
		episode_total_reward += reward

		#проверяем не достигли ли цели или закончилось время
		if done or episode_time == int(MAX_TIME):
			noise = UONoise()
			print('| Reward: {:.5} | Episode: {:} | Time: {:}'.format(episode_total_reward, episode, episode_time))
			break
	network_chages.append(episode_network_change)
	rewards_curve.append(episode_total_reward)
#закрываем окружение
env.close()

Q.write_in_files()

with open(REWARD_FILE, 'w') as f:
    for item in rewards_curve:
        f.write("%s\n" % item)

with open(NETWORK_CHANGE_FILE, 'w') as f:
    for item in network_chages:
        f.write("%s\n" % item)