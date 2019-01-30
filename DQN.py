import numpy as np
import tensorflow as tf
import time

actions = ['up','down','right','left']
features = 1


class maze:
	def __init__(self):
		self.state_now = 0
		self.state_next = 0
		self.WIDTH = 4
		self.LENGTH = 4
		self.TARGET = 15
		self.FRESH_TIME = 0.1

	def reset(self):
		self.state_now = 0
		self.state_next =0

		return self.state_now


	def update_env(self,S,episode,step_counter):
		env_list = ((['-']*(self.WIDTH-1)+['-\n'])*(self.LENGTH-1))+(['-']*(self.WIDTH-1) + ['T\n'])
		if S == self.TARGET:
			interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
			print('\r{}'.format(interaction), end='')
			time.sleep(1)
			print('\r                                ', end='')
		else:
			if (S%self.WIDTH == (self.WIDTH-1)):
				env_list[S] = 'o\n'
			else:
				env_list[S] = 'o'
			interaction = ''.join(env_list)
			print('\r{}'.format(interaction), end='')
			time.sleep(self.FRESH_TIME)


	def get_env_feedback(self,action_idx):
		action_now = actions[action_idx]
		reward = 0.0
		end = False
		#state_next = self.state_now
		print('get_env')
		print(self.state_now)
		print(self.WIDTH)
		print(self.state_now/self.WIDTH)
		print(int(self.state_now/self.WIDTH))
		input()
		if action_now == 'right':
			if((self.state_now % self.WIDTH) == (self.WIDTH-1)):
				self.state_next = self.state_now
			else:
				self.state_next = self.state_now +1
		elif action_now == 'left':
			if((self.state_now % self.WIDTH) == 0):
				self.state_next = self.state_now
			else:
				self.state_next = self.state_now -1
		elif action_now == 'up':
			if(int(self.state_now/self.WIDTH) == 0):
				self.state_next = self.state_now
			else:
				self.state_next = self.state_now -self.WIDTH
		elif action_now == 'down':
			if(int(self.state_now/self.WIDTH) == (self.LENGTH-1)):
				self.state_next = self.state_now
			else:
				self.state_next = self.state_now + self.WIDTH

		#if self.state_next <=89 and self.state_next>=81 :
		#	self.state_next = self.state_now
		#	reward = -1.0
		#	end = True

		print(self.state_now,action_now,self.state_next)
		if(self.state_next == self.TARGET):
			reward = 1.0
			end = True 
		return self.state_next,reward,end





class DeepQNetwork:
	def __init__(
		self,
		n_actions,
		n_features,
		learning_rate=0.1,
		reward_decay=0.9,
		e_greedy=0.9,
		replace_target_iter=300,
		memory_size=500,
		batch_size=32,
		e_greedy_increment=None,
		output_graph=False
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.learning_rate = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		self.learn_step_counter = 0

		self.memory = np.zeros((self.memory_size,n_features*2+2))

		self._build_net()

		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()

		if output_graph:
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []



	def _build_net(self):
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
		with tf.variable_scope('eval_net'):
			c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
			n_l1 = 10
			w_initializer = tf.random_normal_initializer(0., 0.3)
			b_initializer = tf.constant_initializer(0.1)

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1,n_l1], initializer = b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_eval = tf.matmul(l1, w2) + b2

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
		with tf.variable_scope('target_net'):
			c_names = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1,n_l1], initializer = b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_next = tf.matmul(l1, w2) + b2

	def choose_action(self,observation):

		observation = np.array([observation]) 
		observation = observation[np.newaxis,:]
		if np.random.uniform() < self.epsilon:
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			action = np.argmax(actions_value)
		else:
			action = np.random.randint(0, self.n_actions)
		return action

	def store_transition(self,s,a,r,s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a, r], s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)
			print('\ntarget_params_replaced\n')
		if self.memory_counter>self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		q_next, q_eval = self.sess.run(
			[self.q_next, self.q_eval],
			feed_dict={self.s_: batch_memory[:, -self.n_features:],self.s: batch_memory[:, :self.n_features]}
		)
		q_target = q_eval.copy()
		batch_index = np.arange(self.batch_size, dtype=np.int32)
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		reward = batch_memory[:, self.n_features + 1]
		q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

		_, self.cost = self.sess.run([self._train_op, self.loss],feed_dict={self.s: batch_memory[:, :self.n_features],self.q_target: q_target})
		self.cost_his.append(self.cost)
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1



def run_maze():
	step = 0
	for episode in range(300):
		step_counter = 0
		observation = my_maze.reset()
		while True:
			now_action = DQN.choose_action(observation)
			observation_, reward, done = my_maze.get_env_feedback(now_action)
			DQN.store_transition(observation,now_action,reward,observation_)
			if (step>200) and (step%5==0) :
				DQN.learn()
			
			print(actions[now_action],observation,observation_,reward)

			observation = observation_
			my_maze.update_env(observation,episode,step_counter)
			
			step += 1
			step_counter +=1

			if done:
				break
			

	print('game over!')

if __name__ == '__main__':
	my_maze = maze()
	DQN = DeepQNetwork(
		len(actions),
		features,
		learning_rate = 0.01,
		reward_decay = 0.9,
		replace_target_iter = 200,
		memory_size = 2000
		)

	run_maze()
	
