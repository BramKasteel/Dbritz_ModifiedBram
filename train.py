#! /usr/bin/env python

"""
Now includes a split in train/validation/test set. See if it works and how we can use the result

Also includes a document vector embedding, for projection purpose
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import pickle
import yaml
import shutil #For copying files
import fnmatch #For searching filenames
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector
import re

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
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
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

# Create all necessary output folders (if they do not yet exist)
# Runs dir
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))  ### Make it a customizable folder name plz!
print("Writing to {}\n".format(out_dir))
# Testdata dir
testdata_dir = os.path.abspath(os.path.join(out_dir,"testdata"))
if not os.path.exists(testdata_dir):
    os.makedirs(testdata_dir)
# Checkpoint dir
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# Best model dir (Tensorflow usually only saves the newest 5 checkpoints)
bestmodel_dir = os.path.abspath(os.path.join(out_dir,"bestmodel"))
if not os.path.exists(bestmodel_dir):
    os.makedirs(bestmodel_dir)

with open(os.path.join(testdata_dir,'dev_data.pickle'),'wb') as f:
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
        
        # Initialize saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        
        # Try to incorporate projector
#        #1) Write tsv file. Seems to work okay
#        with open(os.path.join(checkpoint_dir, 'metadata.tsv'),'w') as metadata_file:
#            metadata_file.write("Sentence\tLabel\n")
#            sentences = list(vocab_processor.reverse(list(x_train)))
#            for sentence in sentences:
#                sentence = re.sub(r"<UNK>", "", sentence)  #TODO: somewhere else
#                metadata_file.write(sentence+"\t0\n")   
#                
                
        #config_projector = projector.ProjectorConfig()
        #embedding = config_projector.embeddings.add()
        #embedding.tensor_name = cnn.sentence_embedding.name
        #embedding.metadata_path = os.path.join(checkpoint_dir, 'metadata.tsv')
        #projector.visualize_embeddings(tf.summary.FileWriter(checkpoint_dir), config_projector)
        
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
                    #1) First remove the old best file
                    for file in os.listdir(checkpoint_dir):
                        if fnmatch.fnmatch(file,'modelbest*'):
                            file_path = os.path.abspath(os.path.join(checkpoint_dir,file))
                            os.unlink(file_path)
                    #2) Save the new best file as such
                    saver.save(sess, checkpoint_prefix+"best")
                    #3) Copy this new best model to a different folder bestmodel_dir
                    #      (first remove what was in that folder)
                    for file in os.listdir(bestmodel_dir):
                        file_path = os.path.join(bestmodel_dir,file)
                        try: #TODO: is dit niet een beetje voorzichtig?
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(e)
                    for file in os.listdir(checkpoint_dir):
                        if fnmatch.fnmatch(file,'modelbest*'):
                            shutil.copy2(os.path.join(checkpoint_dir,file), bestmodel_dir)
                    
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
