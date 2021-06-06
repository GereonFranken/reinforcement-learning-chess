import glob
import os
import random
import re
import shutil
import time

import chess.svg
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import MCTS_Node
from BlobEnv import BlobEnv
from Chess_Model import ChessModel
from MCTS_Controller import MCTSController as MCTS

MODEL_NAME = "Chess_Engine"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINI_BATCH_SIZE = 8  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -3  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 61

#  Stats settings
AGGREGATE_STATS_EVERY = 5  # episodes
SHOW_PREVIEW = False

env = BlobEnv()


class ChessAgent:

    def __init__(self, color):
        self.model_controller = ChessModel()
        self.color = color  # determine for which color to play

    def play_best_move(self):
        best_mcts_child, y_moves_priors = MCTS(env.board, self.model_controller.model).find_best_move()
        return env.step(best_mcts_child.move, self.color), best_mcts_child, y_moves_priors


agents = {
    'True': ChessAgent(chess.WHITE),
    'False': ChessAgent(chess.BLACK),
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
    agents['True'].model_controller.tensorboard.step = episode
    agents['False'].model_controller.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        (new_state, reward, done), best_child, y_moves_priors = agents[str(env.board.turn)].play_best_move()

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agents[str(env.board.turn)].model_controller.update_replay_memory(
            (env.board, best_child.move, best_child.q, reward, new_state, y_moves_priors, done))
        agents[str(env.board.turn)].model_controller.train(done)
        step += 1
agents[str(env.board.turn)].model_controller.model.save(f'models/{MODEL_NAME}.model')

        # # Append episode reward to a list and log stats (every given number of episodes)
        # ep_rewards.append(episode_reward)
        # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     # agents[str(env.board.turn)].model_controller.tensorboard.update_stats(
        #     #     reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)
        #
        #     # Save model, but only when min reward is greater or equal a set value
        #     if average_reward >= MIN_REWARD:
        #         agents[str(env.board.turn)].model_controller.model.save(
        #             f'models/{MODEL_NAME}__{max_reward:.2f}max_{average_reward:.2f}avg_'
        #             f'{min_reward:.2f}min__'f'{int(time.time())}.model')
        #
        # if not episode % 20:
        #     """ Get best model by average reward, maximum reward and minimum reward in that priority order.
        #         Then delete the other models
        #     """
        #     best_model = ChessModel.get_current_best_model()
        #     agents[str(env.board.turn)].model_controller.model = tf.keras.models.load_model("models/" + best_model)
        #     for model_folder in glob.glob("./models/*"):
        #         if best_model not in model_folder:
        #             shutil.rmtree(model_folder)
