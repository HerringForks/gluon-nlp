# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

""" Sagemaker compatible BERT pretraining training script launcher """

import argparse
import os
import shutil
import subprocess
from argparse import ArgumentError

def main():
    # CLI flags
    # parser
    parser = argparse.ArgumentParser(description='BERT pretraining example.')
    # logging and serialization
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_dir',
                        help='Path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=250, help='Report interval')
    parser.add_argument('--ckpt_interval', type=int, default=25000, help='Checkpoint interval')
    # model
    parser.add_argument('--pretrained', action='store_true',
                        help='Initialize the model with pretrained weights')
    parser.add_argument('--model', type=str, default='bert_12_768_12',
                        choices=['bert_12_768_12', 'bert_24_1024_16'],
                        help='Model to pre-train.')
    parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                                 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                                 'wiki_cn_cased'],
                        help='The pre-defined dataset from which the vocabulary is created.')
    # training
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data file. File name with wildcard such as '
                             'dir/*.train is accepted.')
    parser.add_argument('--total_batch_size', type=int, default=256,
                        help='Global effective batch size. '
                             'total_batch_size = batch_size_per_worker * num_worker * accumulate.')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='Number of batches for gradient accumulation. '
                             'total_batch_size = batch_size_per_worker * num_worker * accumulate.')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
    parser.add_argument('--optimizer', type=str, default='bertadam',
                        help='The optimization algorithm')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Start optimization step from the checkpoint.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--dtype', type=str, default='float16', help='data dtype')
    parser.add_argument('--no_compute_acc', action='store_true',
                        help='skip accuracy metric computation during training')
    # validation
    parser.add_argument('--eval_interval', type=int, default=50000, help='Evaluation interval')
    parser.add_argument('--total_batch_size_eval', type=int, default=256,
                        help='Global batch size for evaluation. total_batch_size_eval = '
                             'batch_size_eval_per_worker * num_worker * accumulate.')
    parser.add_argument('--data_eval', type=str, required=True,
                        help='Path to evaluation data file. File name with wildcard such as '
                             'dir/*.dev is accepted.')
    parser.add_argument('--eval_use_npz', action='store_true',
                        help='Set to True if --data_eval provides npz files instead of raw text files')
    # debugging
    parser.add_argument('--synthetic_data', action='store_true',
                        help='If provided, synthetic data is used for training')
    parser.add_argument('--verbose', action='store_true', help='verbose logging')
    parser.add_argument('--profile', type=str, default=None,
                        help='output profiling result to the provided file path')
    parser.add_argument('--skip_save_states', action='store_true',
                    help='Skip saving training states')
    # data pre-processing
    parser.add_argument('--num_buckets', type=int, default=1,
                        help='Number of buckets for variable length sequence sampling')
    parser.add_argument('--raw', action='store_true',
                        help='If set, both training and dev samples are generated on-the-fly '
                             'from raw texts instead of pre-processed npz files. ')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length. Effective only if --raw is set.')
    parser.add_argument('--short_seq_prob', type=float, default=0,
                        help='The probability of producing sequences shorter than max_seq_length. '
                             'Effective only if --raw is set.')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                        help='Probability for masks. Effective only if --raw is set.')
    parser.add_argument('--max_predictions_per_seq', type=int, default=80,
                        help='Maximum number of predictions per sequence. '
                             'Effective only if --raw is set.')
    parser.add_argument('--cased', action='store_true',
                        help='Whether to tokenize with cased characters. '
                             'Effective only if --raw is set.')
    parser.add_argument('--whole_word_mask', action='store_true',
                        help='Whether to use whole word masking rather than per-subword masking.'
                             'Effective only if --raw is set.')
    parser.add_argument('--sentencepiece', default=None, type=str,
                        help='Path to the sentencepiece .model file for both tokenization and vocab. '
                             'Effective only if --raw is set.')
    parser.add_argument('--num_dataset_workers', type=int, default=4,
                        help='Number of workers to pre-process dataset.')
    parser.add_argument('--num_batch_workers', type=int, default=2,
                        help='Number of workers to pre-process mini-batch.')
    parser.add_argument('--circle_length', type=int, default=2,
                        help='Number of files to be read for a single GPU at the same time.')
    parser.add_argument('--repeat', type=int, default=8,
                        help='Number of times that files are repeated in each shuffle.')
    parser.add_argument('--dataset_cached', action='store_true',
                        help='Whether or not to cache the last processed training dataset.')
    parser.add_argument('--num_max_dataset_cached', type=int, default=0,
                        help='Maximum number of cached processed training dataset.')
    # stage 2
    parser.add_argument('--phase2', action='store_true', help='phase 2 training')
    parser.add_argument('--phase1_num_steps', type=int, help='number of steps for phase 1')
    # communication
    parser.add_argument('--comm_backend', type=str, default='smddp',
                        choices=['smddp', 'dist_sync_device', 'device'],
                        help='Communication backend.')
    parser.add_argument('--gpus', type=str, default=None,
                        help='List of gpus to run when device or dist_sync_device is used for '
                             'communication, e.g. 0 or 0,2,5. empty means using cpu.')
    args = parser.parse_args()

    flags, unknown = parser.parse_known_args()
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'run_pretraining.py'))

    data_train_pattern = flags.data + '/*.train'
    data_eval_pattern = flags.data_eval + '/*.dev'

    # build command as a string, happening after sagemaker parses it
    cmd = (
        f'python {main_path}'
        f' --ckpt_dir={flags.ckpt_dir}'
        f' --log_interval={flags.log_interval}'
        f' --ckpt_interval={flags.ckpt_interval}'
        f' --model={flags.model}'
        f' --dataset_name={flags.dataset_name}'
        f' --data={data_train_pattern}'
        f' --total_batch_size={flags.total_batch_size}'
        f' --accumulate={flags.accumulate}'
        f' --num_steps={flags.num_steps}'
        f' --optimizer={flags.optimizer}'
        f' --lr={flags.lr}'
        f' --warmup_ratio={flags.warmup_ratio}'
        f' --dtype={flags.dtype}'
        f' --eval_interval={flags.eval_interval}'
        f' --total_batch_size_eval={flags.total_batch_size_eval}'
        f' --data_eval={data_eval_pattern}'
        f' --max_seq_length={flags.max_seq_length}'
        f' --max_predictions_per_seq={flags.max_predictions_per_seq}'
        f' --circle_length={flags.circle_length}'
        f' --comm_backend={flags.comm_backend}'
        f' {"--no_compute_acc" if flags.no_compute_acc else ""}'
        f' {"--eval_use_npz" if flags.eval_use_npz else ""}'
        f' {"--verbose" if flags.verbose else ""}'
        f' {"--skip_save_states" if flags.skip_save_states else ""}'
        f' {"--raw" if flags.raw else ""}'
    )

    # print command
    line = '-' * shutil.get_terminal_size()[0]
    print(line, cmd, line, sep='\n')

    # run model
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
