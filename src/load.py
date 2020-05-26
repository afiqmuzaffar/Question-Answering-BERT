import numpy as np
import pandas as pd
import tensorflow as tf
from bert_tf import run_squad as baseline
from bert_tf import modeling as modeling
from bert_tf import optimization
from bert_tf import tokenization
import json
import absl
import sys

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(absl.flags.FLAGS)

flags = absl.flags

flags.DEFINE_string(
    "bert_config_file", "Question-Answering-BERT/src/bert-config/bert_config.json",
    "Configuration json file corresponding to the pre-trained BERT model that specifies the model architecture ")


FLAGS = flags.FLAGS
FLAGS(sys.argv) # Parse the flags

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
baseline.validate_flags_or_throw(bert_config)
tf.io.gfile.makedirs(FLAGS.output_dir)

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

run_config = tf.estimator.RunConfig(
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps)