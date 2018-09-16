import numpy as np
import gym
from gym import wrappers
import random
from neural_network import Network
import matplotlib.pyplot as plt
from collections import deque
















#Эпизоды и время
MAX_EPISODES = 100
MAX_TIME = 1000
NUM_POSSIBLE_ACTIONS = 250 

#Размазывание награды
REWARD_DEPTH = 50 #глубина размазывания
REWARD_FACTOR = 0.95
DONE_FACTOR_REWARD = 5 #бонус за done
DONE_FACTOR_DEPTH = 5 # бонус за done

#Обучение нейросети
HISTORY_SIZE = 10000 #размер буфера истории
BATCH_SIZE = 250
LAYERS = [3,15,1]
LEARNING_RATE = 0.001
L1=0
L2=0.1
INNER_FUNC = 'relu'

#параметры Exploration policy на начальных этапах обучения
IS_EXPLORE = True
EXPLORATION_EPISODES = 100 #сколько эпизодов будет действовать Exploration policy
DURATION = 25 
CHANCE = 0.05 
PERTUB_BETA_COEF1 = 2 #скошенное бета-распределение
PERTUB_BETA_COEF2 = 10

#Другой тип Exploration policy -Ornstein Uhlenbeck Action Noise
IS_NOISE = False
EXPLORATION_EPISODES_UON = 10
SIGMA_UON = 0.5
THETA_UON = 0.15


#стратегия - максимизация аналога энергии
#из простых соображений можно понять что коеф при кинетической энергии ~ в 10 раз больше
KINETC_COEF = 25.0
POTENTIAL_COEF = 2.0

#начальное распределение весов:
W_B_INIT_DISTRIBUTION = ('uniform',-0.003,0.003) #lower_bound and upper_bound
#W_B_INIT_DISTRIBUTION = ('normal',0,1) # mean and standart_deviation

#записывать ли ведео?
IS_VIDEO = True

#кривая обучения
IS_PLOT = True


















def UONoise():
    state = 0
    while True:
        yield state
        state += -THETA_UON*state+SIGMA_UON*np.random.randn()



#аналог энергии
def strategy(now_state):
	return KINETC_COEF*(now_state[1]**2) + POTENTIAL_COEF*abs(now_state[0] + 0.5)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


class HistoryBuffer(object):

    def __init__(self, buffer_size, random_seed=12345):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward):
        experience = [state, action, reward]
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

        return s_batch, a_batch, r_batch

def define_best_action(observation):
	#награды для следующих действий
	rewards_to_controls_map = {}

	#рассматриваем набор возможных следующих действий
	for possible_action in np.linspace(-1, 1, NUM_POSSIBLE_ACTIONS):
		input_vector_representation = np.append(observation, possible_action)
		input_vector_representation = input_vector_representation.flatten()[:, np.newaxis]
		predicted_reward = float(NN.feedforward(input_vector_representation))
		rewards_to_controls_map[predicted_reward] = possible_action

	#ищем действие, которое обещает максимальную пользу 
	rewards=list(rewards_to_controls_map.keys())
	highest_reward = max(rewards)
	return rewards_to_controls_map[highest_reward]




















#создаем буфер для истории
history_buffer = HistoryBuffer(HISTORY_SIZE)

#внутренние активационные функции сети
if INNER_FUNC == 'relu':
	in_func = ReLU
	in_func_der = dReLU
elif INNER_FUNC == 'sigm':
	in_func = sigmoid
	in_func_der = sigmoid_prime

#Инициализируем сеть, оценивающая энергетическую полезность некоторого дейсвтия в данном состоянии
NN=Network(sizes=LAYERS,inner_function=in_func, inner_function_prime=in_func_der,
			output_function=lambda x: x, output_derivative=lambda x: 1,
			l1=L1, l2=L2, init_dist=W_B_INIT_DISTRIBUTION)

#запускаем окружение
env = gym.make('MountainCarContinuous-v0')

#шум
noise = UONoise()

#запись видео
if IS_VIDEO:
	env = wrappers.Monitor(env, "./video", force=True, video_callable=lambda episode_id: True)

#изменение весов и cмещений поэпизодно
network_changes = []

#награда поэпизодно
total_rewards_for_episodes = []

#максимальная награда и номер лучшего эпизода
max_total_reward = 0
index_of_best_episode = 1

#вначале 
count_till_normal = 0
is_pertub=False



















for episode in range(1,MAX_EPISODES+1):

	#начальное состояние машинки
	observation = env.reset()

	#наша метрика, которую хотим улучшить
	episode_reward = 0

	#изменение вектора весов и смещений для кривой обучения
	episode_weight_bias_change = 0

	for episode_time in range(1,MAX_TIME+1):

		if episode < EXPLORATION_EPISODES:

			#проверяем не закончился ли период активного исследования
			if count_till_normal == 0:
				is_pertub = False
			
			if (random.random() < CHANCE) and not is_pertub:
				#вошли в состояние активного исследования
				is_pertub = True
				count_till_normal = DURATION

				#влево или вправо будем поддавать
				if random.random() <= 0.5:
					beta_pertub_a, beta_pertub_b = PERTUB_BETA_COEF1, PERTUB_BETA_COEF2
				else:
					beta_pertub_b, beta_pertub_a = PERTUB_BETA_COEF1, PERTUB_BETA_COEF2
			if is_pertub:
				best_action = np.random.beta(beta_pertub_a, beta_pertub_b)*2-1
				count_till_normal-=1
			else:
				best_action = define_best_action(observation)
		elif IS_NOISE == True:
			best_action = define_best_action(observation)
			if episode < EXPLORATION_EPISODES:
				p = episode/EXPLORATION_EPISODES
				est_action = best_action*p + (1-p)*next(noise)
		else:
			#просто определеяем следующее дейсвтие
			best_action = define_best_action(observation)

		#записываем историю (0.0 пока не размазали)
		history_buffer.add(observation, best_action, 0.0)       	

		#выполняем действие
		observation, reward, done, info = env.step([best_action])

		#считаем реальную энергию
		actual_energy = strategy(observation)

		#размазывание энергии
		k=-1
		if done:
			factor = actual_energy * DONE_FACTOR_REWARD
			while history_buffer.size() > abs(k) and abs(k) < REWARD_DEPTH * DONE_FACTOR_DEPTH:
				history_buffer.buffer[k][2] += factor
				factor *= REWARD_FACTOR
				k -= 1
		else:
			factor = actual_energy
			while history_buffer.size() > abs(k) and abs(k) < REWARD_DEPTH:
				history_buffer.buffer[k][2] += factor
				factor *= REWARD_FACTOR
				k -= 1

		# Если у нас накопилось хоть чуть-чуть данных, потренируем нейросеть
		if history_buffer.size() >= BATCH_SIZE:

			#формируем батч
			state_batch, action_batch, reward_batch = history_buffer.sample_batch(BATCH_SIZE)

			#формируем обучающую выборку
			X_train = np.concatenate((state_batch, action_batch[np.newaxis, :].T), axis=1)
			Y_train = reward_batch

			#обучение
			train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, Y_train)]
			weight_bias_change_forsec = NN.update_mini_batch(mini_batch=train_data, eta=LEARNING_RATE)
			episode_weight_bias_change += weight_bias_change_forsec

		#накапливаем общую награду
		episode_reward += reward

		if done:
			#обновляем шум для отсутсвия антикорреляций
			noise = UONoise()
			print('Episode Reward: {:.5} , Episode: {:} , Time: {:}'.format(episode_reward, episode, episode_time))
			if episode_reward > max_total_reward:
				max_total_reward = episode_reward
				index_of_best_episode = episode
			break

	#добавляем очередную запись об изменении весов и наград
	network_changes.append(episode_weight_bias_change)
	total_rewards_for_episodes.append(episode_reward)

#закрываем окружение
env.close()

#информация о лучшем эпизоде
if max_total_reward > 0:
	print('Maximum reward for {:} episode: {:.5} (video 00{})'.format(index_of_best_episode,max_total_reward,index_of_best_episode-1))



#кривая обучения и наград
if IS_PLOT:
	plt.plot(network_changes)
	plt.xlabel('episodes')
	plt.ylabel('модуль изменений весов и смещений')
	plt.savefig('network_weights')
	plt.close()

	plt.plot(total_rewards_for_episodes)
	plt.xlabel('episodes')
	plt.ylabel('rewards')
	plt.savefig('rewards')
	plt.close()