#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import os
from tensorflow.python.ops import summary_ops_v2

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1


def createNetwork():

    # input layer
    s = tf.keras.layers.Input(shape=(80, 80, 4), dtype='float32')  # tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,  padding='same', activation="relu")(s)
    h_pool1 = tf.keras.layers.MaxPool2D(strides=2, padding='same')(h_conv1)

    h_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation="relu")(h_pool1)

    h_conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="relu")(h_conv2)

    h_conv3_flat = tf.keras.layers.Flatten()(h_conv3)

    h_fc1 = tf.keras.layers.Dense(units=512, activation='relu')(h_conv3_flat)

    # readout layer
    readout = tf.keras.layers.Dense(units=ACTIONS)(h_fc1)

    model = tf.keras.Model(
        inputs=[s],
        outputs=[readout])

    model.summary()

    return model
import pickle
def compute_loss(readout, a, y):

    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)

    cost = tf.reduce_mean(tf.square(tf.convert_to_tensor(y, dtype="float32") - readout_action))

    return cost


@tf.function
def train_step(model, minibatch, optimizer):
    # Record the operations used to compute the loss, so that the gradient
    # of the loss with respect to the variables can be computed.
    with tf.GradientTape() as tape:
    # if True:
        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        y_batch = []
        s_j1_batch = tf.stack(s_j1_batch)
        readout_j1_batch = model(s_j1_batch, training=True)
        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # if terminal, only equals reward
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * tf.keras.backend.max(readout_j1_batch[i]))

        # # perform gradient step

        s_j_batch = tf.stack(s_j_batch)
        readout = model(s_j_batch, training=True)
        # print("readout ", readout)
        # print("a_batch ", a_batch)
        # pickle.dump((readout, a_batch, y_batch), open('preprocess.p', 'wb'))
        loss = compute_loss(readout, a_batch, y_batch)
        # accuracy = self.compute_accuracy(logits, labels)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def trainNetwork(model):

    optimizer = tf.keras.optimizers.Adam(1e-6)  # tf.train.AdamOptimizer(1e-6).minimize(cost)
    MODEL_DIR = "./saved_networks"
    if tf.io.gfile.exists(MODEL_DIR):
        #             print('Removing existing model dir: {}'.format(MODEL_DIR))
        #             tf.io.gfile.rmtree(MODEL_DIR)
        pass
    else:
        tf.io.gfile.makedirs(MODEL_DIR)

    train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
    test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')

    train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
    test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

    checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # x_t = np.expand_dims(x_t, 2)
    # print(x_t.shape)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # print(s_t.shape)
    s_t = np.expand_dims(s_t, 0)
    # print(s_t.shape)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = model(s_t.astype(np.float32), training=True)
        # print("readout_t.shape ", readout_t.shape)
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (1, 80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        # print("s_t.shape ", s_t.shape)  # (1, 80, 80, 4)
        # print("x_t1.shape ", x_t1.shape)  # (1, 80, 80, 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        # print("s_t1.shape", s_t1.shape)  # (1, 80, 80, 4)
        # store the transition in D
        D.append((s_t[0].astype(np.float32), a_t, r_t, s_t1[0].astype(np.float32), terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            train_step(model, minibatch, optimizer)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            checkpoint.save(checkpoint_prefix)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    model = createNetwork()
    trainNetwork(model)

def main():
    playGame()

if __name__ == "__main__":
    main()
