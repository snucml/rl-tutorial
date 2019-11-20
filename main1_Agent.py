import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from replayBuffer import ReplayBuffer
class Agent_DQN:
    def __init__(self,
            action_size,
            state_size,
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon_initial=1,
            epsilon_decay = 0.995,
            batch_size=32):

        # 생성자에 다양한 변수 지정하기
        self.action_size = action_size
        self.state_size = state_size

        # LR 과 global step을 지정한다.
        self.global_step = tf.Variable(0, trainable=False)

        # decayed_learning_rate = learning_rate *
        #                         decay_rate ^ (global_step / decay_steps)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate,
            self.global_step,
            100,
            0.9999,
            staircase=False,
            name='learning_rate'
        )

        # Discount factor 도 지정해 준다.
        self.gamma = discount_factor

        # epsilon greedy 방식으로 탐험을 할 것이므로 epsilon 과 decay를 정해준다.
        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay


        # 배치 사이즈 정하기
        self.batch_size = batch_size
        self.learning_iteration = 0

        # 메모리 정의해주기. 메모리에는 s,a,r,s_ 를 저장해야 한다. 따라서 s, s_ 저장공간, a, r을 위한 저장공간을 만든다.
        self.memory_size = 2000
        self.replayBuffer = ReplayBuffer(self.memory_size)

        # 두가지 네트워크를 정의해서 하나는 Fixed Q-target으로 사용한다.
        self.build_evaluation_network()
        self.build_target_network()

        # target net과 eval net의 파라미터를 모아준다. scope의 상위 directory를 이용해서 모아줄 수 있다.
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tn')
        self.t_params = t_params
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='en')

        # tf assign을 이용하면 특정 텐서 변수값에 다른 하나를 넣을 수 있다.
        self.replace_target_op = [tf.assign(t, (1 - 0.03) * t + 0.03 * e) for t, e in zip(t_params, e_params)]

        # 세션을 정의한다.
        self.sess = tf.Session()

        # initializer 실행
        self.sess.run(tf.global_variables_initializer())
        self.loss_history = []
        self.learning_rate_history = []


    def build_evaluation_network(self):
        '''
        eval net을 만들 땐 target net과는 다르게 loss를 구하는 net이 추가되어야 함.
        target net 은 fixed Q-target을 위해서 쓰는 것이지 업데이트를 하지 않는다. 때문에 이 eval net만 tarinable = Ture 로 설정되어야 함.
        :return:
        '''
        # evaluation net 으로 들어갈 data 를 넣을 placeholder 이다.
        self.eval_input = tf.placeholder(tf.float32, [None, self.state_size], name = 'eval_input')

        #  self.y 와 self.a 는 placeholder 로써, loss 를 구하기 위한 placeholder 이다.
        self.y = tf.placeholder(tf.float32, [None], name='Q_target')
        self.a = tf.placeholder(tf.int64, [None], name='action')

        #  실제 네트워크
        with tf.variable_scope('en'):
            hidden1 = tf.layers.dense(self.eval_input, 10, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.5),
                                      bias_initializer=tf.random_normal_initializer(0., 0.1), name='layer1',
                                      trainable=True)
            self.q_eval = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.5),
                                      bias_initializer=tf.random_normal_initializer(0., 0.1), name='layer2',
                                      trainable=True)

        # loss를 구하는 부분
        with tf.variable_scope('loss'):
            self.a_one_hot = tf.one_hot(self.a, depth=self.action_size)
            self.q_predict = tf.reduce_sum(tf.multiply(self.q_eval, self.a_one_hot), axis=1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.q_predict))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate)\
                .minimize(self.loss, global_step=self.global_step)


    def build_target_network(self):
        self.target_input = tf.placeholder(tf.float32, [None, self.state_size], name = 'target_input')
        with tf.variable_scope('tn'):
            hidden1 = tf.layers.dense(self.target_input, 10, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.5),
                                      bias_initializer=tf.random_normal_initializer(0., 0.1), name='layer1',
                                      trainable=False)
            self.get_q_target = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(0., 0.5),
                                          bias_initializer=tf.random_normal_initializer(0., 0.1), name='layer2',
                                          trainable=False)

    def store_transition(self, s, a, r, s_):
        self.replayBuffer.add(s,a,r,s_)

    def get_action(self, observation):
        '''
        x : 카트 위치
        dx/dt : 카트 속도
        θ : 막대기 각도
        dθ/dt : 각속도
        이 함수는 epsilon 값에 따라 Neural Network 또는 임의의 값 하나를 action으로 선택하여 return 한다.
        '''
        if np.random.uniform() > self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.eval_input: [observation]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.action_size)
        return action

    def learn(self):
        '''
        인공신경망의 업데이트가 이루어지는 함수
        '''
        # 메모리를 적당히 채우면 learn 하고 그렇지 않으면 learn을 생략한다.
        if self.learning_iteration >= self.memory_size:
            # eval_net 과 fixed_q_target을 적절한 비율로 교체해준다.
            self.sess.run(self.replace_target_op)

            batch = self.replayBuffer.get_batch(self.batch_size)
            batch_s = np.asarray([x[0] for x in batch])
            batch_a = np.asarray([x[1] for x in batch])
            batch_r = np.asarray([x[2] for x in batch])
            batch_s_ = np.asarray([x[3] for x in batch])

            # q_eval 은 현재 Q함수값을 구하기 위해, get_q_target은 max함수에 포함되어있는 Q값을 구하기 위해 사용한다.
            get_q_target, q_eval = self.sess.run(
                [self.get_q_target, self.q_eval],
                feed_dict={
                    self.target_input: batch_s_,  # fixed params
                    self.eval_input: batch_s,  # newest params
                })

            # action 은 배치 메모리에서 state가 저장된 다음부분부터가 action이므로 그 값을 가져오면 된다.
            a = batch_a
            # reward는 action 다음에 저장했으므로 그 다음 값을 가져오면 된다.
            reward = batch_r
            # self.y placeholder에 넣어줄 값을 위에서 구한 값으로 적절히 만들어서 넣는다.
            _, self.loss_out = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.eval_input: batch_s,
                                                    self.y: reward + self.gamma * np.max(get_q_target, axis=1),
                                                    self.a: a
                                                    })
            self.loss_history.append(self.loss_out)

            # epsilon -greedy 탐험을 하기 위해 epsilon 값을 주기적으로 낮춰주어야한다.
            self.epsilon = self.epsilon * self.epsilon_decay

        # iteration을 세어주기 위한 변수, 러닝레이트 출력을 위해 히스토리에 하나씩 추가해본다.
        self.learning_iteration += 1
        self.learning_rate_history.append(self.sess.run([self.learning_rate]))


    def plot_loss(self):
        # 파이썬에서 Times New Roman 글씨체를 이용하여 그래프를 출력할 수 있음!  
        plt.title('History')
        ms = 0.1
        me = 1
        line_width = 0.5
        plt.ylabel('Loss')
        plt.xlabel('Training steps')
        plt.plot(np.arange(len(self.loss_history)), self.loss_history, '--^', color='r', markevery=me,
                 label=r'critic loss', lw=line_width, markersize=ms)
        plt.grid()
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim(0, 2)
        plt.show()

    def plot_reward(self, reward_history):
        plt.plot(np.arange(len(reward_history)), reward_history)
        plt.grid()
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.show()


