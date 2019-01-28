import numpy as np
import tensorflow as tf

actions = ['up','down','left','right']
features = 2

class DeepQNetwork:
	def __init__(
		self,
		n_actions,
		n_features,
		learning_rate=0.01,
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

	def _build_net(self):
		self.s  = tf.placeholder(tf.float32,[None,self.n_features],name='s')
		self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')
		self.r = tf.placeholder(tf.float32,[None,],name='r')
		self.a = tf.placeholder(tf.int32,[None,],name='a')

		w_initializer, b_initializer = tf.random_normal_initializer(0.,0.3), tf.constant_initializer(0.1)

		with tf.variable_scope('eval_net'):
			e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e1')
			self.q_eval = tf.layers.dense(e1,self.n_actions, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='q')

		with tf.variable_scope('target_net'):
			t1 = tf.layers.dense(self.s_, 20,tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t1')
			self.q_next = tf.layers.dense(t1,self.n_actions,kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t2')

		with tf.variable_scope('q_target'):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next,axis=1,name='Qmax_s_')
			self.q_target = tf.stop_gradient(q_target)
			




def run_maze():
	step = 0
	for episode in range(300):
		while True:
			action = DQN.choose_action()



if __name__ == '__main__':

	DQN = DeepQNetwork(
		len(actions),
		features,
		learning_rate = 0.01,
		reward_decay = 0.9,
		replace_target_iter = 200,
		memory_size = 2000
		)
	#run_maze()
	

