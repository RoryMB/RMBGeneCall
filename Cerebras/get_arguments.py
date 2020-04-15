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

    parser.add_argument('--mode',
                        choices=[
                            'train',
                            'evaluate',
                            'predict',
                            'validate_only',
                            'compile_only',
                        ],
                        required=True,
                        help='Execution mode')

    parser.add_argument('--epochs',
                        default=1,
                        type=int,
                        help='Epochs')

    parser.add_argument('--batch_size',
                        default=4096,
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

    parser.add_argument('--cs_ip',
                        default=None,
                        help='CS-1 IP address, if applicable')

    parser.add_argument('--gpu',
                        default='-1',
                        type=str,
                        help='Which GPU to use, if applicable')

    return parser.parse_args()
