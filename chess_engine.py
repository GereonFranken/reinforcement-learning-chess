import chess.svg
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from Modified_Tensorboard import ModifiedTensorBoard
from collections import deque
import time
import random
import os
from BlobEnv import BlobEnv
from tqdm import tqdm

MODEL_NAME = "Chess_Engine"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINI_BATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -20  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 50

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


class DQNAgent:

    def __init__(self, color):
        """
       Anfangs gibt es sehr viel zufälliges Aussuchen von Aktionen und updated von Gewichten.
       Um keinem zufälligen Lernen zum Opfer zu fallen, wird erst nach einem bestimmten Durchlauf das target model mit
       dem richtigen model überschrieben
       """
        # Main model => gets trained every step
        self.model = self.create_model()
        self.policy_head = []  # probability distribution over all moves
        self.value_head = []  # expected reward for the best move

        # Target model => used to predict every step
        self.target_model = tf.keras.Sequential()
        # self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}.{int(time.time())}")
        self.target_update_counter = 0
        self.color = color  # determine for which color to play

    def create_model(self):
        input_layer = tf.keras.layers.Input(shape=env.BOARD_REPRESENTATION_SHAPE)
        layers = Conv2D(256, (3, 3), padding="same")(input_layer)
        layers = BatchNormalization()(layers)
        layers = Activation(tf.keras.activations.relu)(layers)
        for _ in range(3):
            layers = self.add_residual_layer(layers)
        self.policy_head = self.set_policy_head(layers)
        self.value_head = self.set_value_head(layers)
        model = tf.keras.Model(inputs=[input_layer], outputs=[self.policy_head, self.value_head], name="Main NN")
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])
        return model

    def add_conv_layer(self, x):
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)
        return x

    def add_residual_layer(self, input_block):
        new_res_block = self.add_conv_layer(input_block)
        new_res_block = Conv2D(256, (3, 3), padding="same")(new_res_block)
        new_res_block = BatchNormalization()(new_res_block)
        new_res_block = tf.keras.layers.add([input_block, new_res_block])
        new_res_block = Activation(tf.keras.activations.relu)(new_res_block)
        return new_res_block

    def set_value_head(self, input_block):
        x = Conv2D(256, (1, 1))(input_block)
        x = BatchNormalization(axis=3)(x)
        x = Activation(tf.keras.activations.relu)(x)
        x = Flatten()(x)
        x = Dense(60, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = Dense(1, activation=tf.keras.activations.tanh, name="value_head")(x)
        return x

    def set_policy_head(self, input_block):
        x = Flatten()(input_block)
        x = Dense(4_672, activation=tf.keras.activations.softmax, name="policy_head")(x)
        return x

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        current_states = np.array([transition[0] for transition in mini_batch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in mini_batch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINI_BATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we wanna update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


env = BlobEnv()
agents = {
    'True': DQNAgent(chess.WHITE),
    'False': DQNAgent(chess.BLACK),
}

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agents['True'].tensorboard.step = episode
    agents['False'].tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # env.update_action_space()
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agents[str(env.board.turn)].get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space_size)

        new_state, reward, done = env.step(action, agents[str(env.board.turn)].color)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agents[str(env.board.turn)].update_replay_memory((current_state, action, reward, new_state, done))
        agents[str(env.board.turn)].train(done)
        current_state = new_state
        step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agents[str(env.board.turn)].tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                agents[str(env.board.turn)].model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
