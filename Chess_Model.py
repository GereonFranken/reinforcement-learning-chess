import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, BatchNormalization
from collections import deque
import random
import datetime

from BlobEnv import BlobEnv
from MCTS_Helper import MCTSHelper as Helper

env = BlobEnv()
REPLAY_MEMORY_SIZE = 1_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 8  # Minimum number of steps in a memory to start training
MINI_BATCH_SIZE = 8  # How many steps (samples) to use for training
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 3  # Terminal states (end of episodes)


class ChessModel:

    def __init__(self):
        # Main model => gets trained every step
        self.model = self.create_model()
        self.policy_head = []  # probability distribution over all moves
        self.value_head = []  # expected reward for the best move
        # Target model => used to predict every step
        self.target_model = self.create_model()
        # self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

    def __new__(cls):  # make class singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(ChessModel, cls).__new__(cls)
        return cls.instance

    def create_model(self):
        input_layer = tf.keras.layers.Input(shape=env.BOARD_REPRESENTATION_SHAPE)
        # input_layer = tf.keras.layers.InputLayer(input_shape=env.BOARD_REPRESENTATION_SHAPE)
        layers = Conv2D(256, (3, 3), padding="same")(input_layer)
        layers = BatchNormalization()(layers)
        layers = Activation(tf.keras.activations.relu)(layers)
        for _ in range(3):
            layers = self._add_residual_layer(layers)
        self.policy_head = self._set_policy_head(layers)
        self.value_head = self._set_value_head(layers)
        model = tf.keras.Model(inputs=input_layer, outputs=[self.policy_head, self.value_head], name="Main_NN")
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def _add_conv_layer(x):
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)
        return x

    def _add_residual_layer(self, input_block):
        new_res_block = self._add_conv_layer(input_block)
        new_res_block = Conv2D(256, (3, 3), padding="same")(new_res_block)
        new_res_block = BatchNormalization()(new_res_block)
        new_res_block = tf.keras.layers.add([input_block, new_res_block])
        new_res_block = Activation(tf.keras.activations.relu)(new_res_block)
        return new_res_block

    @staticmethod
    def _set_value_head(input_block):
        x = Conv2D(256, (1, 1))(input_block)
        x = BatchNormalization(axis=3)(x)
        x = Activation(tf.keras.activations.relu)(x)
        x = Flatten()(x)
        x = Dense(60, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = Dense(1, activation=tf.keras.activations.tanh, name="value_head")(x)
        return x

    @staticmethod
    def _set_policy_head(input_block):
        x = Flatten()(input_block)
        x = Dense(1968, activation=tf.keras.activations.softmax, name="policy_head")(x)
        return x

    def get_qs(self, represented_board: np.ndarray):
        return self.model.predict(np.expand_dims(represented_board, axis=0))

    def update_replay_memory(self, transition):
        """
        :param transition: Tuple(current_board, best_child.move, best_child.q, reward, new_state, done)
        :return:
        """
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        current_boards = np.array([transition[0] for transition in mini_batch])
        current_qs_list = [self.get_qs(env.get_board_representation(current_board, current_board.turn))
                           for current_board in current_boards]

        new_boards = np.array([transition[4] for transition in mini_batch])
        future_qs_list = [self.target_model.predict(np.expand_dims(
            env.get_board_representation(new_board, new_board.turn), axis=0))
                          for new_board in new_boards]
        X = []
        y = []

        for index, (current_board, move, value, reward, new_board, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index][0])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = np.array(current_qs_list[index][0]).reshape(1968)
            current_qs[Helper().find_index_of_move(move)] = new_q

            X.append(env.get_board_representation(current_board, current_board.turn))
            y.append([current_qs, value])

        self.model.fit(np.array(X), np.array(y).reshape((8, 2)), batch_size=MINI_BATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard])

        # updating to determine if we wanna update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0