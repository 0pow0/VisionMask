import argparse
from configargparse import ArgumentParser

# This file contains the declaration of our argument parser

# Needed to parse booleans from command line properly
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = ArgumentParser(description='VM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--arg_log', default=False, type=str2bool, help='save arguments to config file')

    # Dataset parameters
    parser.add_argument('--dataset', choices=['MARIO', 'ATARI', 'DOOM', 'HIGHWAY'], default='MARIO', type=str, help='which dataset to use')
    parser.add_argument('--data_base_path', default='../datasets/', type=str, help='Base bath of the datasets. Should contain subdirectories with the different datasets.')

    # Data processing parameters
    parser.add_argument('--train_batch_size', default=16, type=int, help='batch size used for training')
    parser.add_argument('--val_batch_size', default=16, type=int, help='batch size used for validation')
    parser.add_argument('--test_batch_size', default=16, type=int, help='batch size used for testing')
    parser.add_argument('--use_data_augmentation', default=False, type=str2bool, help='set to true to enable data augmentation on training images')

    # Trainer Parameters
    parser.add_argument('--seed', default=42, type=int, help='seed for all random number generators in pytorch, numpy, and python.random')
    parser.add_argument('--use_tensorboard_logger', default=False, type=str2bool, help='whether to use tensorboard')
    parser.add_argument('--checkpoint_callback', default=True, type=str2bool, help='if true, trained model will be automatically saved')

    # Early stopping Parameters
    parser.add_argument('--early_stop_min_delta', default=0.001, type=float, help='threshold for early stopping condition')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='patience for early stopping to trigger')

    # General Model Parameters
    parser.add_argument('--train_model', default=True, type=str2bool, help='If True, specified model will be trained. If False, model will be tested.')
    parser.add_argument('--agent_type', choices=['ppo', 'dqn'], default='vgg16', type=str, help='type of classifier architecture to use')
    parser.add_argument('--vm_checkpoint', default=None, type=str, help='Path to the .ckpt file that contains the weights of a pretrained explainer. Also contains the weights for the associated classifier.')
    parser.add_argument('--agent_ckpt', default=None, type=str, help='Path to the .ckpt file that contains the weights of a pretrained classifier.')

    # Model-specific parameters
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate used by the Adam optimizer')
    parser.add_argument('--entropy_regularizer', default=1.0, type=float, help='loss weighting term for entropy loss')

    parser.add_argument('--use_stalder_mask_area_loss', default=False, type=str2bool)

    # Metrics parameters
    parser.add_argument('--metrics_threshold', default=-1.0, type=float, help='Threshold for logit to count as positive vs. negative prediction. Use -1.0 for Explainer and 0.0 for classifier.')
    parser.add_argument('--use_deletion_insertion', default=False, type=str2bool, help='If true, compute the deletion/insertion metrics during the testing stage')

    # Mario parameters
    parser.add_argument('--mario_dataset', choices=['black', 'blue', ''], default='', type=str, help='which mario dataset to use, with blue background OR with black backgroudn OR both')
    parser.add_argument('--mario_use_greyscale', default=False, type=str2bool, help='If true, use greyscaled image as input for explainer')
    parser.add_argument('--deletion_fraction', default=0.5, type=float, help='fraction of pixels in mask to be deleted when testing with deletion metric')
    parser.add_argument('--insertion_fraction', default=0.5, type=float, help='fraction of pixels in mask to be inserted when testing with insertion metric')

    # Atari parameters
    parser.add_argument('--atari_env', choices=['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders', 'MsPacman', 'Asteroids', 'RoadRunner'], default='BeamRider', type=str, help='which atari env dataset to use')
    parser.add_argument('--atari_ppo_trained_path', default='/home/rzuo02/work/rl-baselines3-zoo/rl-trained-agents', type=str, help='Path to trained ppo models of stable baselines3')
    
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay for adam (L2 regularization)')
    parser.add_argument('--explainer_type', choices=['deeplab', 'fcn', 'unet'], default='deeplab', type=str, help='which seg model architecture should be used for training or testing')
    parser.add_argument('--ref_value_mode', choices=['background', 'black', 'mean', 'blur'], default='background', type=str)

    return parser

def write_config_file(args, path='config.cfg'):
    with open(path, 'w') as f:
        for k in sorted(args.__dict__):
            if args.__dict__[k] is not None:
                print(k, '=', args.__dict__[k], file=f)
