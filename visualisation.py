#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:54:14 2017

@author: bram, actually bram for realz

Loads the (learned) network, including embedding.
Applies the network to a set of data, creating labels for this set.
Performs some kind of sum-of-vectors to create single vector for a sentence.
Plots these vectors with labels and original text in an appealing way.
"""

import tensorflow as tf
import numpy as np
import os
import data_helpers
#from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector
import re
import fnmatch #For searching filenames

#TODO: check below if we need all these variables
###############################################################################
# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model to load
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Dataset to test the model on
tf.flags.DEFINE_string("dev_set","","Datafile for testing accuracy of model") #Word vectors

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
#tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data") #can be ommitted

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#TODO: write this in a multiclass way
x_test, y_test = data_helpers.load_dev_set(os.path.abspath(FLAGS.dev_set))
y_test = np.argmax(y_test, axis=1)
# Transform back to real words
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_raw = list(vocab_processor.reverse(x_test))
###############################################################################
    
checkpoint_file = os.path.abspath(FLAGS.checkpoint_dir + "modelbest")

#Load graph and 1) predict classes
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

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

#Load graph and 2) retrieve word embeddings.
#Tranform these into some kind of single vector per sentence
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        
        #Things we want to input
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        batches = data_helpers.batch_iter(list(x_test),FLAGS.batch_size,1,shuffle=False)
        
        #Tensors we want to evaluate (/run)
        embed = graph.get_operation_by_name("embedding/embedded_chars").outputs[0]

        sentence_embedding = []
        
        for x_test_batch in batches:
            batch_embed = sess.run(embed, {input_x: x_test_batch})
            #Guess: dimension will be [64,59,300,1] 
            #TODO: >> IT IS 56!!! IS THAT BECAUSE SEQUENCE LENGTH IS RECOMPUTED???
            
            #How to handle these embedded word tensors/ arrays? (np.sum or tf.reduce_mean both work)
            batch_sum_of_vectors = tf.reduce_mean(batch_embed,axis=1,keep_dims=False)
            #print(batch_sum_of_vectors.shape)
            
            sentence_embedding.append(batch_sum_of_vectors)
        sentence_embedding = tf.concat(sentence_embedding,axis=0)  #This is the new embedding 'doc2vec'-like
        sentence_embedding = tf.squeeze(sentence_embedding)  #Dims #sentences, embedding_dim=300
        print(type(sentence_embedding))        

#Lets see if we can visualize this TING GOES SKRRRAA

#TODO: automate the following step:
#1) create a projector_config.pbtxt in the checkpoint folder
#TODO: this metafile only has to be created once, of course
#2) create the metadata file with labels
for file in os.listdir(FLAGS.checkpoint_dir): #first remove the old file
    if fnmatch.fnmatch(file,'*.tsv'):
        file_path = os.path.abspath(os.path.join(FLAGS.checkpoint_dir,file))
        os.unlink(file_path)
#Write new tsv file, corresponding to this ordering of the test set
with open(os.path.abspath(os.path.join(FLAGS.checkpoint_dir,"metadata.tsv")), "w") as record_file:
    record_file.write("Sentence\tLabel\n")
    for i in range(0,sentence_embedding.shape[0]):   #SeqIO.parse("/home/fil/Desktop/420_2_03_074.fastq", "fastq"):
        sentence = x_raw[i]
        sentence = re.sub(r"<UNK>", "", sentence)
        record_file.write(sentence+"\t"+str(all_predictions[i])+"\n")



## SHORT TUTORIAL

graph2 = tf.Graph()
with graph2.as_default():
    with tf.Session() as sess:
        a = tf.constant(5.0)
        b = tf.constant(6.0)
        c = a * b

        LOG_DIR = FLAGS.checkpoint_dir
        fixed_embedding = tf.constant(sentence_embedding,'sentence_embedding')
        #print(sess.run(fixed_embedding))
        #config = projector.ProjectorConfig()
        #embedding = config.embeddings.add()
        #embedding.tensor_name = fixed_embedding.name
        #embedding.metadata_path = os.path.join(LOG_DIR,'metadata.tsv')
        #summary_writer = tf.summary.FileWriter(LOG_DIR)
        #projector.visualize_embeddings(summary_writer,config)
        
        #saver = tf.train.Saver([embedding_var])
        sess.run(tf.global_variables_initializer())
        print(sess.run(c))
        #saver.save(sess, os.path.join(LOG_DIR, 'embedding_var.ckpt'))



































