# The code is rewritten based on source code from tensorflow tutorial for Recurrent Neural Network.
# https://www.tensorflow.org/versions/0.6.0/tutorials/recurrent/index.html
# You can get source code for the tutorial from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
#
# There is dropout on each hidden layer to prevent the model from overfitting
#
# Here is an useful practical guide for training dropout networks
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# You can find the practical guide on Appendix A
import numpy as np
import tensorflow as tf
import operator
from tensorflow.models.rnn import rnn_cell
import time
import csv
from random import shuffle
import random
from tensorflow.models.rnn import rnn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import networkx as nx
import matplotlib.pyplot as plt
import itertools

class StudentModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size
        self._min_lr = config.min_lr
        self.num_skills = num_skills = config.num_skills
        self.hidden_size = config.hidden_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        input_size = num_skills*2

        inputs = self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._target_id = target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])
        self._early_stop = early_stop = tf.placeholder(tf.int32, [batch_size])

        self._early_stop = early_stop = tf.placeholder(tf.float32, [batch_size])
        hidden1 = rnn_cell.LSTMCell(size, input_size)
        #hidden2 = rnn_cell.LSTMCell(size, size)
        #hidden3 = rnn_cell.LSTMCell(size, size)

        #add dropout layer between hidden layers
        if is_training and config.keep_prob < 1:
            hidden1 = rnn_cell.DropoutWrapper(hidden1, output_keep_prob=config.keep_prob)
            #hidden2 = rnn_cell.DropoutWrapper(hidden2, output_keep_prob=config.keep_prob)
            #hidden3 = rnn_cell.DropoutWrapper(hidden3, output_keep_prob=config.keep_prob)

        #cell = rnn_cell.MultiRNNCell([hidden1])
        cell = hidden1

        # initial state
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        input_data = tf.reshape(self._input_data, [-1])
        #one-hot encoding
        with tf.device("/cpu:0"):
            labels = tf.expand_dims(input_data, 1)
            indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
            concated = tf.concat(1, [indices, labels])
            inputs = tf.sparse_to_dense(concated, tf.pack([batch_size*num_steps, input_size]), 1.0, 0.0)
            inputs.set_shape([batch_size*num_steps, input_size])

        # [batch_size, num_steps, input_size]
        inputs = tf.reshape(inputs, [batch_size, num_steps, input_size])
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state, sequence_length=early_stop)


        #state = self._initial_state
        #with tf.variable_scope("RNN"):
        #    (cell_output, state) = cell(inputs, state)
        #    self._final_state = state

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        # calculate the logits from last hidden layer to output layer
        softmax_w = tf.get_variable("softmax_w", [size, num_skills])
        softmax_b = tf.get_variable("softmax_b", [num_skills])
        logits = tf.matmul(output, softmax_w) + softmax_b

        # from output nodes to pick up the right one we want
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, self.target_id)

        #make prediction
        self._pred = self._pred_values = pred_values = tf.sigmoid(selected_logits)

        if not is_training:
            return
        #loss = -tf.reduce_sum(target_correctness*tf.log(pred_values)+(1-target_correctness)*tf.log(1-pred_values))
        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(selected_logits, target_correctness))

        #self._cost = cost = tf.reduce_mean(loss)
        self._cost = cost = loss

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # apply gradient descent to minimize loss function
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Momentum algorithm
        #optimizer = tf.train.MomentumOptimizer(self.lr, config.momentum)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self._train_op = optimizer.minimize(grads, tvars)

    def assign_lr(self, session, lr_value):
        if (lr_value > self.min_lr):
            session.run(tf.assign(self._lr, lr_value))
        else:
            session.run(tf.assign(self._lr, self.min_lr))

    @property
    def early_stop(self):
        return self._early_stop

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_data(self):
        return self._input_data

    @property
    def min_lr(self):
        return self._min_lr

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def pred_values(self):
        return self._pred_values

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op




class HyperParamsConfig(object):
  """Small config."""
  init_scale = 0.05
  num_steps = 0
  learning_rate = 0.25
  num_layer = 1
  max_grad_norm = 4
  hidden_size = 300
  max_epoch = 5
  max_max_epoch = 500
  keep_prob = 0.6
  lr_decay = 0.9
  num_skills = 123
  momentum = 0.95
  min_lr = 0.0001
  batch_size = 150


def run_epoch(session, m, students, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    index = 0
    actual_labels = []
    count = 1
    weighted_p = {}
    pred_labels = []
    mask = []
    while(index+m.batch_size <= len(students)):
        weighted_p[index] = {}
        early_stop = []
        if verbose:
            print "Running " + str(count) + " batch"
        x = np.zeros((m.batch_size, m.num_steps))
        target_id = []
        target_correctness = []
        early_stop = []
        count+=1
        for i in range(m.batch_size):
            student = students[index+i]
            num_problems = int(student[0][0])
            early_stop.append(num_problems)
            problem_ids = student[1]
            correctness = student[2]
            masks = student[3]
            for j in range(len(problem_ids)-1):
                if(j >= m.num_steps):
                    break
                problem_id = int(problem_ids[j])
                label_index = 0
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + m.num_skills
                x[i, j] = label_index
                target_id.append(i*m.num_steps*m.num_skills+j*m.num_skills+int(problem_ids[j+1]))
                target_correctness.append(int(correctness[j+1]))
                actual_labels.append(int(correctness[j+1]))
                mask.append(int(masks[j+1]))

        if verbose:
            print "feeding variables into graph"
        pred, _ = session.run([m.pred, eval_op], feed_dict={
            m.input_data: x, m.target_id: target_id,
            m.target_correctness: target_correctness, m.early_stop: early_stop})

        for p in pred:
            pred_labels.append(p)

        #average_p = np.average(pred_labels[0])
        #count = 0
        #for pred in pred_labels[0]:
            #weighted_p[index][count] = pred/average_p
            #count += 1
        index += m.batch_size
    #print pred_labels
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    #calculate r^2
    r2 = r2_score(actual_labels, pred_labels)
    return pred_labels, mask, actual_labels, rmse, auc, r2


def read_data_from_csv_file(fileName):
    config = HyperParamsConfig()
    inputs = []
    targets = []
    rows = []
    skills_num = config.num_skills
    max_num_problems = 0
    with open(fileName, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    index = 0
    i = 0
    print "the number of rows is " + str(len(rows))
    tuple_rows = []
    #turn list to tuple
    while(index < len(rows)-1):
        problems_num = int(rows[index][0])
        if(problems_num <= 2):
            index += 4
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index], rows[index+1], rows[index+2], rows[index+3])
            tuple_rows.append(tup)
            index += 4
    #shuffle the tuple

    random.shuffle(tuple_rows)
    print "The number of students is ", len(tuple_rows)
    print "Finish reading data"
    return tuple_rows, max_num_problems

def get_max_steps(fileName):
    max_steps = 0
    rows = []
    index = 0
    with open(fileName, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    while(index < len(rows)-1):
        problems_num = int(rows[index][0])
        if problems_num > max_steps:
            max_steps = problems_num
        index += 1
    return max_steps

def main(unused_args):

  config = HyperParamsConfig()
  eval_config = HyperParamsConfig()
  eval_config.batch_size = 50

  train_data_path = "data/2015_builder_train.csv"
  test_data_path = "data/10_skill_builder_mask_test.csv"
  result_file_path = "2010_skill_builder_0224"

  model_name = "10_skill_builder_0224_benchmark"

  #train_students, train_max_num_problems = read_data_from_csv_file(train_data_path)
  #config.num_steps = train_max_num_problems
  #print "the max number of steps in train is %d" % config.num_steps
  #test_students, test_max_num_problems = read_data_from_csv_file(test_data_path)
  #eval_config.num_steps = test_max_num_problems
  #print "the max number of steps in test is %d" % eval_config.num_steps

  #config.num_steps = get_max_steps(train_data_path)
  #config.num_steps = 200
  #print "The max number of steps for train is " + str(config.num_steps)
  #eval_config.num_steps = get_max_steps(test_data_path)
  eval_config.num_steps = 2000
  print "The max number of steps for test is " + str(eval_config.num_steps)
  start_over = False #if False, will store variables from disk

  with tf.Graph().as_default(), tf.Session() as session:

    # training model
    #print "building model"
    #with tf.variable_scope("model", reuse=None, initializer=initializer):
    #  m = StudentModel(is_training=True, config=config)
    #print "finish building model"
    # testing model
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                            config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      mtest = StudentModel(is_training=False, config=eval_config)
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    if(start_over == False):
        #restore variables from disk, make sure file exists
        saver.restore(session, model_name)
        print "Model restored"


    # log hyperparameters to results file
    with open(result_file_path, "a+") as f:
        print("Writing hyperparameters into file")
        f.write("Hidden layer size: %d \n" % (config.hidden_size))
        f.write("Dropout rate: %.3f \n" % (config.keep_prob))
        f.write("Batch size: %d \n" % (config.batch_size))
        f.write("Max grad norm: %d \n" % (config.max_grad_norm))

    print("Start to test model....")
    test_students, test_max_num_problems = read_data_from_csv_file(test_data_path)
    pred_labels, mask, actual_labels, rmse, auc, r2 = run_epoch(session, mtest, test_students, tf.no_op())
    print("Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (rmse, auc, r2))

    filteredPred = list(itertools.compress(pred_labels, mask))
    filteredActual = list(itertools.compress(actual_labels, mask))

    rmse = sqrt(mean_squared_error(filteredActual, filteredPred))
    fpr, tpr, thresholds = metrics.roc_curve(filteredActual, filteredPred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    #calculate r^2
    r2 = r2_score(filteredActual, filteredPred)
    print("Subproblem Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (rmse, auc, r2))
    print "Writing pred_labels, mask, actual_labels to csv file..."

    with open("subprolbem.csv", "wb") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(pred_labels)
        wr.writerow(mask)
        wr.writerow(actual_labels)
    #top_weighted_p = {}
    #values = []
    #for key, value in weighted_p.iteritems():
    #    if key not in top_weighted_p:
    #        top_weighted_p[key] = {}
    #    twp = dict(sorted(value.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
    #    for key1, value1 in twp.iteritems():
    #        values.append(value1)
    #        top_weighted_p[key][key1] = value1
    #values = sorted(values)

    #G=nx.Graph()
    #labels = {}
    #for key, value in top_weighted_p.iteritems():
    #    for key1, value1 in value.iteritems():
    #        if key != key1:
    #            if values.index(value1) > 460:
    #                print(str(key) + " -- " + str(key1))
    #                G.add_edge(key, key1, weight= value1)
    #                labels[key] = key
    #                labels[key1] = key1
    #pos = nx.spring_layout(G)
    #nx.draw_networkx_labels(G, pos, labels)
    #nx.draw(G, pos)
    #plt.savefig("skill_graph.png")
    #plt.show()
    #with open(result_file_path, "a+") as f:
    #    f.write("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % ((i+1)/2, rmse, auc, r2))
    #    f.write("\n")

    print("*"*10)
    #for i in range(config.max_max_epoch):
      #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
      #m.assign_lr(session, config.learning_rate * lr_decay)
      #train_students, train_max_num_problems = read_data_from_csv_file(train_data_path)
      #print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      #rmse, auc, r2 = run_epoch(session, m, train_students, m.train_op,
    #                                verbose=True)
      #
      #valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      #if((i+1) % 2 == 0):
          #print "Save variables to disk"
          #save_path = saver.save(session, model_name)
          #print("*"*10)


if __name__ == "__main__":
    tf.app.run()
