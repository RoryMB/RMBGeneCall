"""Acquire arguments."""

import argparse

def get_arguments():
    """Set up argument parsing for CS-1 model running."""

    parser = argparse.ArgumentParser("CS-1 evaluation std arguments")

    parser.add_argument('--model_dir',
                        default='./model_dir',
                        type=str,
                        help='TensorFlow model_dir')

    parser.add_argument('--data_dir',
                        default='',
                        type=str,
                        help='TensorFlow data_dir')

    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train the model')

    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate the model')

    parser.add_argument('--xla',
                        default=False,
                        action='store_true',
                        help='Test for XLA compatibility')

    parser.add_argument('--epochs',
                        default=1,
                        type=int,
                        help='Epochs')

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='Batch size')

    parser.add_argument('--learning_rate',
                        default=0.001,
                        help='Learning rate')

    parser.add_argument('--inpfx',
                        default='',
                        type=str,
                        help='Prefix prepended to tfrecord, JSON and log files')

    parser.add_argument('--log_frequency',
                        default=1000,
                        type=int,
                        help='Training run summarization interval')

    return parser.parse_args()
