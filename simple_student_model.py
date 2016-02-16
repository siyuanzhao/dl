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

class SimpleStudentModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size
        self._min_lr = config.min_lr
        self.num_skills = num_skills = config.num_skills
        self.hidden_size = config.hidden_size
        size = config.hidden_size
        input_size = num_skills*2

        inputs = self._input_data = tf.placeholder(tf.int32, [batch_size])
        self._target_id = target_id = tf.placeholder(tf.int32, [batch_size])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [batch_size])

        #hidden1 = rnn_cell.LSTMCell(size, input_size)
        #hidden2 = rnn_cell.LSTMCell(size, size)
        #hidden3 = rnn_cell.LSTMCell(size, size)

        #add dropout layer between hidden layers
        #if is_training and config.keep_prob < 1:
            #hidden1 = rnn_cell.DropoutWrapper(hidden1, output_keep_prob=config.keep_prob)
            #hidden2 = rnn_cell.DropoutWrapper(hidden2, output_keep_prob=config.keep_prob)
            #hidden3 = rnn_cell.DropoutWrapper(hidden3, output_keep_prob=config.keep_prob)

        #cell = rnn_cell.MultiRNNCell([hidden1])

        # initial state
        #self._initial_state = cell.zero_state(batch_size, tf.float32)

        #one-hot encoding
        with tf.device("/cpu:0"):
            labels = tf.expand_dims(self._input_data, 1)
            indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            inputs = tf.sparse_to_dense(concated, tf.pack([batch_size, input_size]), 1.0, 0.0)
            inputs.set_shape([batch_size, input_size])

        #state = self._initial_state
        #with tf.variable_scope("RNN"):
            #(cell_output, state) = cell(inputs, state)
            #self._final_state = state
        hidden_w = tf.get_variable("hidden_w", [input_size, size])
        hidden_b = tf.get_variable("hidden_b", [size])
        hidden_logits = tf.matmul(inputs, hidden_w) + hidden_b

        # calculate the logits from last hidden layer to output layer
        output_w = tf.get_variable("output_w", [size, num_skills])
        output_b = tf.get_variable("output_b", [num_skills])
        logits = tf.matmul(hidden_logits, output_w) + output_b

        # from output nodes to pick up the right one we want
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, self.target_id)

        #make prediction
        self._pred = self._pred_values = pred_values = tf.sigmoid(selected_logits)

        #loss = -tf.reduce_sum(target_correctness*tf.log(pred_values)+(1-target_correctness)*tf.log(1-pred_values))
        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(selected_logits, target_correctness))

        #self._cost = cost = tf.reduce_mean(loss)
        self._cost = cost = loss / batch_size

        if not is_training:
            return

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
  learning_rate = 0.35
  num_layer = 1
  max_grad_norm = 4
  hidden_size = 300
  max_epoch = 10
  max_max_epoch = 500
  keep_prob = 0.6
  lr_decay = 0.9
  num_skills = 111
  momentum = 0.95
  min_lr = 0.0001
  batch_size = 100


def run_epoch(session, m, fileName, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()

    #state = m.initial_state.eval()
    inputs, targets = read_data_from_csv_file(fileName)
    index = 0
    pred_labels = []
    actual_labels = []
    while(index+m.batch_size < len(inputs)):
        x = inputs[index:index+m.batch_size]
        y = targets[index:index+m.batch_size]
        target_id = []
        target_correctness = []
        count = 0
        for item in y:
            target_id.append(count*m.num_skills + item[0])
            target_correctness.append(item[1])
            actual_labels.append(item[1])
            count += 1

        index += m.batch_size

        pred, _ = session.run([m.pred, eval_op], feed_dict={
            m.input_data: x,m.target_id: target_id,
            m.target_correctness: target_correctness})

        for p in pred:
            pred_labels.append(p)
    #print pred_labels
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    #calculate r^2
    r2 = r2_score(actual_labels, pred_labels)
    return rmse, auc, r2


def read_data_from_csv_file(fileName):
    config = HyperParamsConfig()
    inputs = []
    targets = []
    rows = []
    skills_num = config.num_skills
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
            index += 3
        else:
            tup = (rows[index], rows[index+1], rows[index+2])
            tuple_rows.append(tup)
            index += 3
    #shuffle the tuple

    random.shuffle(tuple_rows)
    print "The number of students is ", len(tuple_rows)
    while(i < len(tuple_rows)):
        #skip the num is smaller than 2
        tup = tuple_rows[i]
        problems_num = int(tup[0][0])
        if(problems_num <= 2):
            i += 1
        else:
            problem_ids = tup[1]
            correctness = tup[2]
            for j in range(len(problem_ids)-1):

                problem_id = int(problem_ids[j])

                label_index = 0
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + skills_num
                inputs.append(label_index)
                target_instance = [int(problem_ids[j+1]), int(correctness[j+1])]
                targets.append(target_instance)
            i += 1
    print "Finish reading data"
    return inputs, targets

'''
def read_data_from_csv_file(fileName):
    config = HyperParamsConfig()

    rows = []
    skills_num = config.num_skills
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
            index += 3
        else:
            tup = (rows[index], rows[index+1], rows[index+2])
            tuple_rows.append(tup)
            index += 3
    #shuffle the tuple

    random.shuffle(tuple_rows)
    print "The number of students is ", len(tuple_rows)
    student_data = []
    while(i < len(tuple_rows)):
        #skip the num is smaller than 2
        tup = tuple_rows[i]
        problems_num = int(tup[0][0])
        if(problems_num <= 2):
            i += 1
        else:
            inputs = []
            targets = []
            problem_ids = tup[1]
            correctness = tup[2]
            for j in range(len(problem_ids)-1):

                problem_id = int(problem_ids[j])

                label_index = 0
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + skills_num
                inputs.append(label_index)
                target_instance = [int(problem_ids[j+1]), int(correctness[j+1])]
                targets.append(target_instance)
            student_data.append((inputs, targets))
            i += 1
    print "Finish reading data"
    return student_data
'''


def main(unused_args):

  config = HyperParamsConfig()
  eval_config = HyperParamsConfig()
  eval_config.batch_size = 1

  train_data_path = "data/2015_builder_train.csv"
  test_data_path = "data/2015_builder_test.csv"
  result_file_path = "2015_builder_results"

  model_name = "2015_builder_variables"

  start_over = True #if False, will store variables from disk

  with tf.Graph().as_default(), tf.Session() as session:

    if(start_over):
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    else:
        #restore variables from disk, make sure file exists
        saver.restore(session, model_name)
        print "Model restored"
    # training model
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = SimpleStudentModel(is_training=True, config=config)
    # testing model
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mtest = SimpleStudentModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver()

    # log hyperparameters to results file
    with open(result_file_path, "a+") as f:
        print("Writing hyperparameters into file")
        f.write("Hidden layer size: %d \n" % (config.hidden_size))
        f.write("Dropout rate: %.3f \n" % (config.keep_prob))
        f.write("Batch size: %d \n" % (config.batch_size))
        f.write("Max grad norm: %d \n" % (config.max_grad_norm))

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      rmse, auc, r2 = run_epoch(session, m, train_data_path, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (i + 1, rmse, auc, r2))
      #valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      if((i+1) % 3 == 0):
          print "Save variables to disk"
          save_path = saver.save(session, model_name)
          print("*"*10)
          print("Start to test model....")
          rmse, auc, r2 = run_epoch(session, mtest, test_data_path, tf.no_op())
          print("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % ((i+1)/3, rmse, auc, r2))
          with open(result_file_path, "a+") as f:
              f.write("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % ((i+1)/3, rmse, auc, r2))
              f.write("\n")

          print("*"*10)

if __name__ == "__main__":
    tf.app.run()
