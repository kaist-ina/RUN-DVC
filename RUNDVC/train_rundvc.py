import logging
import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tables
import os
import sys
from itertools import accumulate
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


import models as model_path
import torchvision.models as pt_models
import torch.optim as optim

# for data loader and torch
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import tqdm

from methods.rundvc import RUNDVC
import params.param_rundvc as param
from sequence_dataset_DA import Get_Dataloader_DA, Get_Dataloader_augmentation, get_dataset_size

logging.basicConfig(format='%(message)s', level=logging.INFO)
tables.set_blosc_max_threads(512)
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

def train_model(args):
    pileup = args.pileup
    ochk_prefix = args.ochk_prefix if args.ochk_prefix is not None else ""
    os.makedirs(os.path.dirname(ochk_prefix), exist_ok=True)

    label_shape = param.label_shape
    label_shape_cum = param.label_shape_cum
    param.trainBatchSize = args.trainBatchSize
    batch_size, chunk_size = param.trainBatchSize, param.chunk_size
    assert batch_size % chunk_size == 0
    chunks_per_batch = batch_size // chunk_size
    param.RANDOM_SEED = args.seed
    random.seed(param.RANDOM_SEED)
    np.random.seed(param.RANDOM_SEED)
    param.initialLearningRate = args.learning_rate if args.learning_rate else param.initialLearningRate
    param.maxEpoch = args.maxEpoch if args.maxEpoch else param.maxEpoch
    param.u_ratio = args.u_ratio
    param.tau = args.tau
    param.mu = args.mu
    param.l2RegularizationLambda = args.l2RegularizationLambda
    param.RELATIVE_THRESHOLD = args.RELATIVE_THRESHOLD
    param.UNIFY_MASK = args.UNIFY_MASK
    param.FIXED_mu = args.FIXED_mu
    param.USE_DIST_ALIGN = args.USE_DIST_ALIGN
    param.opt_name = args.opt_name if args.opt_name else param.opt_name
    param.use_scheduler = True if args.lr_scheduler else False
    param.lr_scheduler = args.lr_scheduler if args.lr_scheduler else None
    param.momentum = args.momentum
    param.lr_min = args.lr_min
    param.lr_gamma = args.lr_gamma
    param.lr_step_size = args.lr_step_size
    param.randaug_num = args.randaug_num
    param.randaug_intensity = args.randaug_intensity
    param.clip_grad_norm = args.clip_grad_norm
    param.USE_SWA = args.USE_SWA
    param.swa_step_size = args.swa_step_size
    param.swa_ema_decay = args.swa_ema_decay
    param.swa_start_epoch = args.swa_start_epoch
    param.SSDA = args.SSDA

    param.FULL_LABEL = args.FULL_LABEL
    if param.FULL_LABEL:
        assert(not args.USE_RLI)
        assert(not args.USE_SSL)
    param.USE_CORAL = args.USE_CORAL
    param.CORAL_weight = args.CORAL_weight

    param.USE_RLI = args.USE_RLI
    param.USE_SSL = args.USE_SSL
    if not param.USE_RLI:
        print("\n\n[WARNING] You are not using random logit interpolation\n\n")

    if not param.USE_SSL:
        print("\n\n[WARNING] You are not using Semi-supervised learning module\n\n")

    task_num = 4 if args.add_indel_length else 2
    mini_epochs = args.mini_epochs
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        
    assert(args.random_validation)
    
    data = Get_Dataloader_DA(param, args.bin_fn, args.bin_fn_ul, 8, bin_fn_tl=args.bin_fn_tl if args.SSDA is None else None, tl_size=args.tl_size if args.tl_size else None ,validation_fn = args.validation_fn,
            seed=args.seed, device=device, platform= args.platform, batch_size= batch_size, exclude_training_samples=args.exclude_training_samples, 
            random_validation=args.random_validation, add_indel_length= args.add_indel_length)
    
    source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
    target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]

    if args.SSDA is not None:
        # our dataloader selecte split training and validation set in a ordered way, see get_chunk_list functoin in sequence_dataset_da.py
        param.trainingDatasetPercentage =  args.SSDA
        target_labeled_size = get_dataset_size(args.bin_fn_tl, exclude_training_samples=args.exclude_training_samples)
        total_target_labeled = target_labeled_size * args.SSDA 
        total_target_labeled /= len(source_dataloader_train_weak) 
        
        # param.trainBatchSize_tl = int(total_target_labeled )
        param.trainBatchSize_tl = 1000
        param.SSDA_ITER = int(param.trainBatchSize_tl // total_target_labeled)
        
        # print("param.SSDA_ITER",param.trainBatchSize_tl, total_target_labeled,param.SSDA_ITER)
        # unlabeled true: iterate data by param.UNLABELED_SCALER times more
        # augmentation 1: weak augmentation for target labeled weak loader
        if args.bin_fn_tl is None:
            args.bin_fn_tl = args.bin_fn_ul
        target_labeled_weak_loader, target_labeled_strong_loader = Get_Dataloader_augmentation(param, args.bin_fn_tl, 7, unlabeled=True, 
            batch_size= param.trainBatchSize_tl, validation_fn = args.validation_fn,
            device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
            random_validation=True, add_indel_length= args.add_indel_length, augmentation=1)

    source_total_chunks = len(source_dataloader_train_weak) * chunks_per_batch
    target_total_chunks = int( len(target_dataloader_train_weak) / param.UNLABELED_SCALER * chunks_per_batch  )
    logging.info("[INFO] The size of source dataset: {} num batch:{} {}".format(source_total_chunks * chunk_size, len(source_dataloader_train_weak), args.bin_fn))
    logging.info("[INFO] The size of target dataset: {} num batch:{} {}".format(target_total_chunks * chunk_size, len(target_dataloader_train_weak)/ param.UNLABELED_SCALER , args.bin_fn_ul))
    logging.info("[INFO] The size of Source validation dataset: {}".format(len(source_dataloader_test)*chunks_per_batch * chunk_size))
    logging.info("[INFO] The source training batch size: {}".format(batch_size))
    # logging.info("[INFO] Relative thresholding: {}".format(param.RELATIVE_THRESHOLD))
    logging.info("[INFO] Threshold Tau: {}".format(param.tau))
    # logging.info("[INFO] The Fixed mu: {}".format(param.FIXED_mu))
    logging.info("[INFO] mu value: {}".format(param.mu))
    logging.info("[INFO] Optimizer used: {}".format(param.opt_name))
    logging.info("[INFO] lr_scheduler used: {}".format(param.lr_scheduler))
    logging.info("[INFO] clip_grad_norm used: {}".format(param.clip_grad_norm))
    logging.info("[INFO] unlabeled/labeled ratio: {}".format(param.u_ratio))
    logging.info("[INFO] Weight Decay: {}".format(param.l2RegularizationLambda))
    logging.info("[INFO] learning_rate: {}".format(param.initialLearningRate))
    if args.SSDA:
        logging.info("[INFO-SSDA] Portion: {}".format(param.SSDA))
        logging.info("[INFO-SSDA] The target labeled training batch size: {}".format(param.trainBatchSize_tl))
        logging.info("[INFO-SSDA] The size of labeled target dataset: {}".format(len(target_labeled_weak_loader) / param.UNLABELED_SCALER *  param.trainBatchSize_tl))
    

    logging.info("[INFO] Maximum training epoch: {}".format(param.maxEpoch))
    # logging.info("[INFO] Unified Masking: {}".format(param.UNIFY_MASK))
    # logging.info("[INFO] Use Dist Align: {}".format(param.USE_DIST_ALIGN))
    # logging.info("[INFO] Use CORAL: {}".format(param.USE_CORAL))
    logging.info("[INFO] Use SWA: {}".format(param.USE_SWA))
    if param.USE_SWA:
        logging.info("[INFO] The swa_step_size: {}".format(param.swa_step_size))
        logging.info("[INFO] The swa_ema_decay: {}".format(param.swa_ema_decay))
        logging.info("[INFO] The swa_start_epoch: {}".format(param.swa_start_epoch))
    logging.info("[INFO] randaug_num value: {}".format(param.randaug_num))
    logging.info("[INFO] randaug_intensity value: {}".format(param.randaug_intensity))
    if args.tl_size:
        logging.info("[INFO] Use target labeled dataset size: {}".format(args.tl_size))
    logging.info("[INFO] Use RLI: {}".format(param.USE_RLI))
    logging.info("[INFO] Use SSL: {}".format(param.USE_SSL))
    logging.info("[INFO] Dataset shuffle seed: {}".format(args.seed))
    logging.info("[INFO] Platform: {}".format(args.platform))
    logging.info("[INFO] ochk_prefix: {}".format(ochk_prefix))

    config_text=[]
    config_text.append("[INFO] The size of source dataset: {} num batch:{} {}".format(source_total_chunks * chunk_size, len(source_dataloader_train_weak), args.bin_fn))
    config_text.append("[INFO] The size of target dataset: {} num batch:{} {}".format(target_total_chunks * chunk_size, len(target_dataloader_train_weak)/ param.UNLABELED_SCALER , args.bin_fn_ul))
    config_text.append("[INFO] The size of Source validation dataset: {}".format(len(source_dataloader_test)*chunks_per_batch * chunk_size))
    config_text.append("[INFO] The source training batch size: {}".format(batch_size))
    # config_text.append("[INFO] Relative thresholding: {}".format(param.RELATIVE_THRESHOLD))
    config_text.append("[INFO] Threshold Tau: {}".format(param.tau))
    # config_text.append("[INFO] The Fixed mu: {}".format(param.FIXED_mu))
    config_text.append("[INFO] mu value: {}".format(param.mu))
    config_text.append("[INFO] Optimizer used: {}".format(param.opt_name))
    config_text.append("[INFO] lr_scheduler used: {}".format(param.lr_scheduler))
    config_text.append("[INFO] clip_grad_norm used: {}".format(param.clip_grad_norm))
    config_text.append("[INFO] unlabeled/labeled ratio: {}".format(param.u_ratio))
    config_text.append("[INFO] Weight Decay: {}".format(param.l2RegularizationLambda))
    config_text.append("[INFO] learning_rate: {}".format(param.initialLearningRate))
    if args.SSDA:
        config_text.append("[INFO-SSDA] Portion: {}".format(param.SSDA))
        config_text.append("[INFO-SSDA] The target labeled training batch size: {}".format(param.trainBatchSize_tl))
        config_text.append("[INFO-SSDA] The size of labeled target dataset: {}".format(len(target_labeled_weak_loader) / param.UNLABELED_SCALER *  param.trainBatchSize_tl))
    

    config_text.append("[INFO] Maximum training epoch: {}".format(param.maxEpoch))
    # config_text.append("[INFO] Unified Masking: {}".format(param.UNIFY_MASK))
    # config_text.append("[INFO] Use Dist Align: {}".format(param.USE_DIST_ALIGN))
    # config_text.append("[INFO] Use CORAL: {}".format(param.USE_CORAL))
    config_text.append("[INFO] Use SWA: {}".format(param.USE_SWA))
    if param.USE_SWA:
        config_text.append("[INFO] The swa_step_size: {}".format(param.swa_step_size))
        config_text.append("[INFO] The swa_ema_decay: {}".format(param.swa_ema_decay))
        config_text.append("[INFO] The swa_start_epoch: {}".format(param.swa_start_epoch))
    config_text.append("[INFO] randaug_num value: {}".format(param.randaug_num))
    config_text.append("[INFO] randaug_intensity value: {}".format(param.randaug_intensity))
    if args.tl_size:
        config_text.append("[INFO] Use target labeled dataset size: {}".format(args.tl_size))
    config_text.append("[INFO] Use RLI: {}".format(param.USE_RLI))
    config_text.append("[INFO] Use SSL: {}".format(param.USE_SSL))
    config_text.append("[INFO] Dataset shuffle seed: {}".format(args.seed))
    config_text.append("[INFO] Platform: {}".format(args.platform))
    config_text.append("[INFO] ochk_prefix: {}".format(ochk_prefix))
    

    param.s2t = os.path.basename(args.bin_fn[:-1] if args.bin_fn[-1] == "/" else args.bin_fn) + "_2_"+os.path.basename(args.bin_fn_ul[:-1] if args.bin_fn_ul[-1]=="/" else args.bin_fn_ul)+"\n"
    param.config_write = "\n".join(config_text)
    
    if args.test_inception:
        from inception_v3 import Inception3
        encoder = Inception3(platform=args.platform, aux_logits=False, dropout=0)
    else:
        encoder = model_path.Clair3_Feature( [1,1,1], args.platform)
        
    classifier = model_path.Clair3_cls( add_indel_length=args.add_indel_length)
    
    trainer = RUNDVC(encoder, classifier)
    
    if args.chkpnt_fn is not None:
        trainer.load_model_SWA(args.chkpnt_fn)

    trainer.train(source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test,
               target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
               param, ochk_prefix, target_labeled_dataloader = target_labeled_weak_loader if args.SSDA is not None else None, target_labeled_strong_dataloader = target_labeled_weak_loader if args.SSDA is not None else None)
    


def main():
    parser = ArgumentParser(description="Train a Clair3 model")

    parser.add_argument('--platform', type=str, default="ilmn",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--bin_fn', type=str, default="", required=True,
                        help="labeled Binary tensor input from source domain")

    parser.add_argument('--bin_fn_ul', type=str, default="", required=True,
                        help="unlabeled Binary tensor input from target domain")

    parser.add_argument('--bin_fn_tl', type=str, default=None, required=False,
                        help="labeled Binary tensor input from target domain")

    # semisupervised domain adaptation, load target data and use the value
    parser.add_argument('--SSDA',  type=float, default=None,
                        help=SUPPRESS)

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model to resume training or for fine-tuning")

    parser.add_argument('--ochk_prefix', type=str, default=None, required=True,
                        help="Prefix for model output after each epoch")

    # options for advanced users
    parser.add_argument('--maxEpoch', type=int, default=40,
                        help="Maximum number of training epochs")

    parser.add_argument('--tl_size', type=int, default=None,
                        help="Target Labeled dataset size to use in SSDA setting")
    
    parser.add_argument('--seed', type=int, default=param.RANDOM_SEED,
                        help="Random seed for dataset shuffle")
    
    parser.add_argument('--learning_rate', type=float, default=param.initialLearningRate,
                        help="Set the initial learning rate, default: %(default)s")

    parser.add_argument('--opt_name', type=str, default=param.opt_name,
                        help="Optimizer type")
    
    parser.add_argument('--lr_scheduler', type=str, default="exponentiallr",
                        help="Optimizer type")

    parser.add_argument('--momentum', type=float, default=param.momentum,
                        help="Set the momentum, default: %(default)s")

    parser.add_argument('--lr_min', type=float, default=param.lr_min,
                        help="Set the lr_min, default: %(default)s")

    parser.add_argument('--lr_gamma', type=float, default=param.lr_gamma,
                        help="Set the lr_gamma, default: %(default)s")
            
    parser.add_argument('--lr_step_size', type=int, default=param.lr_step_size,
                        help="set lr_step_size for steplr scheduler")

    parser.add_argument('--randaug_num', type=int, default=param.randaug_num,
                        help="set lr_step_size for steplr scheduler")

    parser.add_argument('--randaug_intensity', type=int, default=param.randaug_intensity,
                        help="set lr_step_size for steplr scheduler")

    parser.add_argument('--clip_grad_norm', type=float, default=None,
                        help="Set the clip_grad_norm, default: %(default)s")

    parser.add_argument('--test_inception', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--USE_SWA', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--FULL_LABEL', type=str2bool, default=False,
                        help=SUPPRESS)

    parser.add_argument('--USE_CORAL', type=str2bool, default=False,
                        help=SUPPRESS)
    
    parser.add_argument('--CORAL_weight', type=float, default=1,
                        help="Set the CORAL_weight, default: %(default)s")

    parser.add_argument('--USE_RLI', type=str2bool, default=True,
                        help=SUPPRESS)

    parser.add_argument('--USE_SSL', type=str2bool, default=True,
                        help=SUPPRESS)

    parser.add_argument('--swa_step_size', type=int, default=param.swa_step_size,
                        help="set lr_step_size for steplr scheduler")
                    
    parser.add_argument('--swa_start_epoch', type=int, default=param.swa_start_epoch,
                        help="set swa start_epoch")

    parser.add_argument('--swa_ema_decay', type=float, default=param.swa_ema_decay,
                        help="Set the swa_ema_decay, default: %(default)s")

    parser.add_argument('--exclude_training_samples', type=str, default="_20",
                        help="Define training samples to be excluded")

    parser.add_argument('--mini_epochs', type=int, default=1,
                        help="Number of mini-epochs per epoch")

    parser.add_argument('--trainBatchSize', type=int, default=param.trainBatchSize,
                        help="Train batch size")

    parser.add_argument('--u_ratio', type=int, default=param.u_ratio,
                        help="Unlabeled ratio")

    parser.add_argument('--tau', type=float, default=param.tau,
                        help="Threshold for confidence")

    parser.add_argument('--mu', type=float, default=param.mu,
                        help="mu for unlabeled loss")                                                

    parser.add_argument('--l2RegularizationLambda', type=float, default=param.l2RegularizationLambda,
                        help="l2RegularizationLambda")                                                

    parser.add_argument('--RELATIVE_THRESHOLD', type=str2bool, default=param.RELATIVE_THRESHOLD,
                        help=SUPPRESS)
    
    parser.add_argument('--UNIFY_MASK', type=str2bool, default=param.UNIFY_MASK,
                        help=SUPPRESS)

    parser.add_argument('--FIXED_mu', type=str2bool, default=param.FIXED_mu,
                        help=SUPPRESS)
    
    parser.add_argument('--USE_DIST_ALIGN', type=str2bool, default=param.USE_DIST_ALIGN,
                        help=SUPPRESS)


    # Internal process control
    ## In pileup training mode or not
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Add indel length for training and calling, default true for full alignment
    parser.add_argument('--add_indel_length', type=str2bool, default=True,
                        help=SUPPRESS)

    # mutually-incompatible validation options
    vgrp = parser.add_mutually_exclusive_group()
    vgrp.add_argument('--random_validation', action='store_true',
                        help="Use random sample of dataset for validation, default: %(default)s")

    vgrp.add_argument('--validation_fn', type=str, default=None,
                        help="Binary tensor input for use in validation: %(default)s")


    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
