#! /usr/bin/env python

"""
Now includes a split in train/validation/test set. See if it works and how we can use the result
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import pickle
import yaml
import math
import shutil #For copying the best model to a different folder (else it will be deleted)
import fnmatch
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("validation_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for testing")
###tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
###tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings",True,"Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")  #TODO: Check if this causes our best to drop out!?
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
### decay coefficient


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#TODO: check what yaml does
with open("config.yml",'r') as ymlfile:
    cfg = yaml.load(ymlfile)

dataset_name = cfg["datasets"]["default"]
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dim = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dim = FLAGS.embedding_dim    


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
datasets = None
if dataset_name == 'mrpolarity':
    datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                    cfg["datasets"][dataset_name]["negative_data_file"]["path"])
elif dataset_name == "localdata":
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
    
x_text, y = data_helpers.load_data_labels(datasets)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
#BRAM very much happens in the next line:
#Map words to numbers +
#Pad sentences that are too short +
#Shorten sentences that are too long
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/val/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
validation_index = dev_sample_index + -1*int(FLAGS.validation_percentage * float(len(y)))
x_train, x_val, x_dev = x_shuffled[:validation_index], x_shuffled[validation_index:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_val, y_dev = y_shuffled[:validation_index], y_shuffled[validation_index:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Val/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_val), len(y_dev)))

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))  ### Make it a customizable folder name plz!
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir+'/testdata/'):
            os.makedirs(out_dir+'/testdata/')
with open(out_dir+'/testdata/dev_data.pickle','wb') as f:
    pickle.dump([x_dev, y_dev], f)


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)        

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation Summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
        
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # Directory for saving the best model
        if not os.path.exists(out_dir+'/bestmodel/'):
            os.makedirs(out_dir+'/bestmodel/')
        
        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                #load embedding word2vec vectors
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
            sess.run(cnn.W.assign(initW))


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
        
        def val_step(x_batch, y_batch, best_acc, writer=None):
            """
            Evaluates model on a validation subset. Might also be incorporated in function dev_step.
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob:1.0}
            step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return [accuracy>best_acc, accuracy]
        
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        best_acc = 0;
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                newbest, accuracy = val_step(x_val, y_val, best_acc, writer=val_summary_writer)
                
                if newbest:
                    best_acc = accuracy
                    try:  #Remove (or try to) former best file if we have found an improvement
                        os.remove(checkpoint_prefix+"best")
                    except OSError:
                        pass
                    saver.save(sess, checkpoint_prefix+"best")
                    #Copy this best model to a different file
                    for file in os.listdir(out_dir+'/bestmodel/'):
                        file_path = os.path.join(out_dir+'/bestmodel/',file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(e)
                    for file in os.listdir(checkpoint_dir):
                        if fnmatch.fnmatch(file,'modelbest*'):
                            shutil.copy2(checkpoint_dir+'/'+file, out_dir+'/bestmodel/')
                    #TODO: make al folders and filenames more clear: it is beginning to become a mess
                    
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
