import argparse

import warnings
warnings.filterwarnings("ignore")

from bilm.data import BidirectionalLMDataset
from bilm.training import train, load_vocab


def main(args):
    tf_save_dir = args.save_dir
    tf_log_dir = args.log_dir
    
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 4  # batch size for each GPU
    n_gpus = -1

    # number of tokens in training data
    n_train_tokens = args.size

    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 16],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': 261,
                     'n_highway': 1},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 1024,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 128,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 16,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False, shuffle_on_load=True)

    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--log_dir', help='Log folder')
    parser.add_argument('--size', type=int, help='Number of training tokens'
                        , default=21100)

    arguments = parser.parse_args()
    main(arguments)
