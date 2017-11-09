#! /usr/bin/env python

# Script written by Brammie, to see which words are important for our classifier.
# Here, important is defined as to be ignored less by the max-pooling layer.
# TODO: there is currently no CPU-implementation of tf.max_pool_with_argmax, 
# so I have written something nasty myself. Either improve it sometime, or wait
# for CPU support of the tf function.

import tensorflow as tf
import numpy as np
import os
#import time
#import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pandas


# Parameters
# ==================================================

#TODO: update this function to also test on local datafiles, like the training already does

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_string("dev_set","","Datafile for testing accuracy of model")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# The names of convolutional layers have filter_size in them
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#Map filter_sizes to a list of integers
filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),

#TODO: modify so we can load something arbitrary in a correct manner
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
else:
    x_test, y_test = data_helpers.load_dev_set(os.path.abspath(FLAGS.dev_set))
    y_test = np.argmax(y_test, axis=1)
    # Transform back to real words
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_raw = np.array(list(vocab_processor.reverse(x_test)))  #TODO: remove <UNK>'s



print("\nEvaluating...\n")

# Evaluation
# ==================================================
#checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)  #To get the latest checkpoint file in some folder
#Use best performing network
checkpoint_file = os.path.abspath(FLAGS.checkpoint_dir + "modelbest")



#New feature: get the argmax of the pooling layer and the corresponding words
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        
        
        sequence_length = x_test.shape[1]
        histo_total = np.zeros((len(x_raw),sequence_length))
        for filter_size in [3,4,5]: #TODO: HARDCODEEDDDDD
            # Tensors we want to evaluate
            #New::::::::::::::::::::::::
            convrelu = graph.get_operation_by_name("conv-maxpool-%s/relu" %filter_size).outputs[0]
            #We also want the weights of the fully connected layer
            weights = graph.get_operation_by_name("output/W").outputs[0]
            

            num_filters = 10 #TODO: WATCH OUT HARDCODEDDDDDD
           
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            impwords = []
            
            for x_test_batch in batches:
                #batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                #all_predictions = np.concatenate([all_predictions, batch_predictions])
            
                batch_convrelu = sess.run(convrelu, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_weights = sess.run(weights, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                
                print(batch_weights.shape) !!! HERE WAS IK BLEAVEN
                
                #Because of the way we set up padding in our max-pooling layer, we can retrieve the argmax of the pooling layer
                #by a simple argmax along the second dimension
                batch_impwords = np.squeeze(np.argmax(batch_convrelu,axis=1))
                impwords.append(batch_impwords)
            impwords = np.concatenate(impwords)
        
            # Count how many times certain words were important for the outcome
            histogram = np.zeros((impwords.shape[0],sequence_length))
            ran = int((filter_size-1)/2)
            for word in range(0,sequence_length):
                count = (impwords == word).sum(axis=1)
                for copy in range(0,filter_size):   #TODO: can be changed into some kind of running sum over columns
                    if (word-ran+copy) in range(0,sequence_length):
                        histogram[:,(word-ran+copy)] = histogram[:,(word-ran+copy)] + count
            histogram = histogram/filter_size/num_filters
            histo_total = histo_total + histogram

histo_total = histo_total/3
           
#Create some kind of output
sentences_human_readable = x_raw
output = []
for row in range(0,len(sentences_human_readable)):
    sentence = sentences_human_readable[row]
    sentence = sentence.split(" ")
    #print(sentence)
    count = histo_total[row,:]
    zipped = zip(sentence,count)
    
    stringed = ["%s %.2f" %t for t in zipped]
    stringed = " ".join(stringed)
    stringed = stringed + "\n\n"
    output.append(stringed)

out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    for item in output:
        f.write(item)


        
        
        
        
        
        
            
            #batch_pooled_actual = sess.run(pooled_actual, {input_x: x_test_batch, dropout_keep_prob:1.0})
            #batch_pooled_here = sess.run(pooled_here, {input_x: x_test_batch, dropout_keep_prob:1.0})
            #print(batch_convrelu.shape)
            #check = tf.reduce_mean(tf.cast(tf.equal(batch_maxpool_argmax[0],batch_pooled_here),tf.float32))
            #print(batch_pooled_actual.shape)
            #print(batch_pooled_here.shape)
            #print(batch_maxpool_argmax[0].shape)
            #print(batch_maxpool_argmax[1].shape)
            #print(check.eval())     ##### SHAPOWWWWWW
            



##OLD
#graph = tf.Graph()
#with graph.as_default():
#    session_conf = tf.ConfigProto(
#      allow_soft_placement=FLAGS.allow_soft_placement,
#      log_device_placement=FLAGS.log_device_placement)
#    sess = tf.Session(config=session_conf)
#    with sess.as_default():
#        # Load the saved meta graph and restore variables
#        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#        saver.restore(sess, checkpoint_file)
#
#        # Get the placeholders from the graph by name
#        input_x = graph.get_operation_by_name("input_x").outputs[0]
#        # input_y = graph.get_operation_by_name("input_y").outputs[0]
#        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#        # Tensors we want to evaluate
#        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#
#        # Generate batches for one epoch
#        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
#
#        # Collect the predictions here
#        all_predictions = []
#
#        for x_test_batch in batches:
#            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
#            all_predictions = np.concatenate([all_predictions, batch_predictions])
#
## Print accuracy if y_test is defined
#if y_test is not None:
#    correct_predictions = float(sum(all_predictions == y_test))
#    print("Total number of test examples: {}".format(len(y_test)))
#    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
#
##TODO: remove <UNK>s
## Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
#out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
#print("Saving evaluation to {0}".format(out_path))
#with open(out_path, 'w') as f:
#    csv.writer(f).writerows(predictions_human_readable)
#    
