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

# for focal loss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import tqdm

logging.basicConfig(format='%(message)s', level=logging.INFO)
tables.set_blosc_max_threads(512)
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

class FocalLoss(nn.Module):
    def __init__(self, label_shape_cum, task, effective_label_num=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, input, target):
        input =  torch.clamp(input, 1e-9, 1-1e-9)
        cross_entropy = -target * torch.log(input)
        cross_entropy = cross_entropy.sum(-1)  # [N, label_channel]
        loss = ((1-input)**self.gamma) * target
        loss = loss.sum(-1)
        loss = loss * cross_entropy 
        return loss.mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self,):
        super(CrossEntropyLoss, self).__init__()
    def forward(self, input, target):
        input =  torch.clamp(input, 1e-9, 1-1e-9)
        cross_entropy = -target * torch.log(input)
        loss = cross_entropy.sum(-1)  # [N, label_channel]
        return loss.mean()


def validate(model,optimizer, loss_func, device, ochk_prefix, epoch, step, best_val_loss, val_loader, target_val_loader, history_writer ):
    with torch.no_grad(): 
        model.eval()
        running_vloss = 0.0
        pbar = tqdm.tqdm ( val_loader)
        for i, (inputs, labels) in enumerate(pbar, 0):

            inputs = inputs.squeeze()
            inputs = inputs.to(device)
            for j in range(len(labels)):
                labels[j] = labels[j].squeeze().to(device)
            outputs = model(inputs)
            # 
            loss = loss_func[0]( outputs[0], labels[0] ) + loss_func[1]( outputs[1], labels[1] ) + loss_func[2]( outputs[2], labels[2] ) + loss_func[3]( outputs[3], labels[3] )
            running_vloss += loss.item()
        epoch_loss_source = running_vloss / (i + 1)
        
        if epoch_loss_source < best_val_loss:
            best_val_loss = epoch_loss_source
            torch.save(  {
                        'epoch': epoch,
                        'steps': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        } , ochk_prefix+ ".val.best.pt")
            print(f'Source Validation for Epoch {epoch}| loss: {epoch_loss_source:.3f} | best_loss {best_val_loss:.3f} NEW BEST!!')
        else:
            print(f'Source Validation for Epoch {epoch}| loss: {epoch_loss_source:.3f} | best_loss {best_val_loss:.3f}')
    
    test_epoch_loss = 0
    if target_val_loader:
        with torch.no_grad(): 
            model.eval()
            running_vloss = 0.0
            pbar = tqdm.tqdm ( target_val_loader)
            for i, (inputs, labels) in enumerate(pbar, 0):

                inputs = inputs.squeeze()
                inputs = inputs.to(device)
                for j in range(len(labels)):
                    labels[j] = labels[j].squeeze().to(device)
                outputs = model(inputs)
                loss = loss_func[0]( outputs[0], labels[0] ) + loss_func[1]( outputs[1], labels[1] ) + loss_func[2]( outputs[2], labels[2] ) + loss_func[3]( outputs[3], labels[3] )
                running_vloss += loss.item()
            test_epoch_loss = running_vloss / (i + 1)
            print(f'Target Validation for Epoch {epoch}| loss: {test_epoch_loss:.3f}')
    history_writer.write('[Validation-Epoch {}/Step: {}] Best loss: {:.6f}; source loss: {:.6f}; Target domain loss: {:.6f};\n'.format(epoch+1, i, best_val_loss, epoch_loss_source, test_epoch_loss))
    history_writer.flush()
    return best_val_loss

def train_model(args):
    pileup = args.pileup
    ochk_prefix = args.ochk_prefix if args.ochk_prefix is not None else ""
    os.makedirs(os.path.dirname(ochk_prefix), exist_ok=True)
    if pileup:
        import params.param_rundvc as param
        model = model_path.Clair3_P( add_indel_length=args.add_indel_length)
    else:
        import params.param_rundvc as param
        model = model_path.Clair3( [1,1,1], platform= args.platform, add_indel_length=args.add_indel_length)
        
    label_shape = param.label_shape
    label_shape_cum = param.label_shape_cum
    batch_size, chunk_size = param.trainBatchSize, param.chunk_size
    assert batch_size % chunk_size == 0
    chunks_per_batch = batch_size // chunk_size
    random.seed(param.RANDOM_SEED)
    np.random.seed(param.RANDOM_SEED)
    learning_rate = args.learning_rate if args.learning_rate else param.initialLearningRate
    max_epoch = args.maxEpoch if args.maxEpoch else param.maxEpoch
    task_num = 4 if args.add_indel_length else 2
    mini_epochs = args.mini_epochs
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        
    from sequence_dataset_DA import Get_Dataloader_augmentation, Get_Dataloader_augmentation_pileup
    
    target_val_loader = None

    if pileup:
        train_loader, val_loader = Get_Dataloader_augmentation_pileup(param, args.bin_fn, 4, validation_fn = args.validation_fn,
            device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
            random_validation=args.random_validation, add_indel_length= args.add_indel_length, augmentation=0)

        if args.SSDA is not None:

            # last part of data is loaded for validation
            _, target_val_loader = Get_Dataloader_augmentation_pileup(param, args.ul_bin_fn, 4, validation_fn = args.validation_fn,
                device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
                random_validation=True, add_indel_length= args.add_indel_length, augmentation=0)

            if args.SSDA:
                param.trainingDatasetPercentage =  args.SSDA
                # unlabeled makes data loader iterate multiple times
                target_loader, _ = Get_Dataloader_augmentation_pileup(param, args.ul_bin_fn, 4, unlabeled=True, validation_fn = args.validation_fn,
                    batch_size= param.trainBatchSize_tl, device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
                    random_validation=True, add_indel_length= args.add_indel_length, augmentation=0)

    else:
        train_loader, val_loader = Get_Dataloader_augmentation(param, args.bin_fn, 6, validation_fn = args.validation_fn,
            device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
            random_validation=args.random_validation, add_indel_length= args.add_indel_length, augmentation=0)
        
        if args.SSDA is not None:
            # last part of data is loaded for validation
            _, target_val_loader = Get_Dataloader_augmentation(param, args.ul_bin_fn, 6, validation_fn = args.validation_fn,
                 device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
                random_validation=True, add_indel_length= args.add_indel_length, augmentation=0)
            if args.SSDA:
                param.trainingDatasetPercentage =  args.SSDA
                # unlabeled makes data loader iterate multiple times
                target_loader, _ = Get_Dataloader_augmentation(param, args.ul_bin_fn, 6, unlabeled=True, validation_fn = args.validation_fn,
                    batch_size= param.trainBatchSize_tl, device=device, platform= args.platform, exclude_training_samples=args.exclude_training_samples, 
                    random_validation=True, add_indel_length= args.add_indel_length, augmentation=0)

    if args.iteration is None:
        args.iteration = len(train_loader)
    total_chunks = len(train_loader) * chunks_per_batch
    logging.info("[INFO] The size of train dataset: {}".format(total_chunks * chunk_size))
    logging.info("[INFO] The size of validation dataset: {}".format(len(val_loader)*chunks_per_batch * chunk_size))
    logging.info("[INFO] The training batch size: {}".format(batch_size))
    logging.info("[INFO] The training learning_rate: {}".format(learning_rate))
    # logging.info("[INFO] Total training steps: {}".format(total_steps))
    logging.info("[INFO] 1-epoch training steps: {}".format( args.iteration ))
    logging.info("[INFO] Maximum training epoch: {}".format(max_epoch))
    if args.SSDA is not None:
        if args.SSDA:
            logging.info("[INFO-SSDA] Unlabeled data percentage: {} Num: {}".format(args.SSDA, len(target_loader)/ param.UNLABELED_SCALER *  param.trainBatchSize_tl))
        else:
            logging.info("[INFO-SSDA] Unlabeled data percentage: {} Num: {}".format(args.SSDA, 0))
    logging.info("[INFO] ochk_prefix: {}".format(ochk_prefix))
    logging.info("[INFO] Start training...")

    # loss_func = [FocalLoss(label_shape_cum, task, None) for task in range(task_num)]
    loss_func = [CrossEntropyLoss() for task in range(task_num)]

    from rangerlars import RangerLars
    # optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay = param.l2RegularizationLambda )
    # optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay = param.l2RegularizationLambda )
    optimizer = RangerLars(model.parameters(), lr=1e-4, weight_decay = param.l2RegularizationLambda)

    start_epoch = 0
    if args.chkpnt_fn is not None:
        print("Loaded Pretrained Model")
        checkpoint = torch.load(args.chkpnt_fn)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.cuda()    

    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    w_idx = 0
    
    import time
    history_writer = open( ochk_prefix+"Train_log.txt","w")
    
    history_writer.write( args.bin_fn+"2"+args.ul_bin_fn )
    history_writer.flush()

    best_accuracy = float('inf')
    best_val_loss = float('inf')

    validate_interval = args.iteration // param.evalInterval

    for epoch in range(start_epoch+1, max_epoch):
        running_loss = 0.0
        loss_batch = 100
        model.train()
        
        train_data = iter(train_loader)
        if args.SSDA:
            target_data = iter(target_loader)

        pbar = tqdm.tqdm (range(args.iteration))
        for step in pbar:

            try:
                (inputs, labels) = next(train_data)
            except:
                train_data = iter(train_loader)
                (inputs, labels) = next(train_data)
            inputs = inputs.squeeze()
            for j in range(len(labels)):
                labels[j] = labels[j].squeeze().to(device)

            num_target_labeled = 0
            if args.SSDA:
                try:
                    (inputs_t, labels_t) = next(target_data)
                except:
                    target_data = iter(target_loader)
                    (inputs_t, labels_t) = next(target_data)

                inputs_t = inputs_t.squeeze()
                num_target_labeled = inputs_t.size(0)
                for j in range(len(labels_t)):
                    labels_t[j] = labels_t[j].squeeze().to(device)
                    labels[j] = torch.cat([labels[j], labels_t[j] ],0 )
                inputs = torch.cat([inputs, inputs_t],0)

            optimizer.zero_grad()

            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if num_target_labeled:
                # labeled source
                loss = loss_func[0]( outputs[0][:-num_target_labeled], labels[0][:-num_target_labeled] ) + loss_func[1]( outputs[1][:-num_target_labeled], labels[1][:-num_target_labeled] )\
                        + loss_func[2]( outputs[2][:-num_target_labeled], labels[2][:-num_target_labeled] ) + loss_func[3]( outputs[3][:-num_target_labeled], labels[3][:-num_target_labeled] )
                # labeled target
                loss += loss_func[0]( outputs[0][-num_target_labeled:], labels[0][-num_target_labeled:] ) + loss_func[1]( outputs[1][-num_target_labeled:], labels[1][-num_target_labeled:] )\
                        + loss_func[2]( outputs[2][-num_target_labeled:], labels[2][-num_target_labeled:] ) + loss_func[3]( outputs[3][-num_target_labeled:], labels[3][-num_target_labeled:] ) 
            else:
                loss = loss_func[0]( outputs[0], labels[0] ) + loss_func[1]( outputs[1], labels[1] )\
                        + loss_func[2]( outputs[2], labels[2] ) + loss_func[3]( outputs[3], labels[3] )
            
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"epoch": epoch, "loss": f"{loss.item():.3f}" 
                 })
            
            
            if (step % validate_interval) == (validate_interval-1) :
                best_val_loss = validate(model, optimizer, loss_func, device, ochk_prefix,  
                                        epoch, step, best_val_loss, val_loader, target_val_loader
                                        , history_writer )
                model.train()

            if step % 50 == 0:
                w_idx += 1
                writer.add_scalars('[Train] Loss', {"Task 1":loss_func[0]( outputs[0], labels[0] ).item(),
                                    "Task 2":loss_func[1]( outputs[1], labels[1] ).item(),
                                    "Task 3":loss_func[2]( outputs[2], labels[2] ).item(),
                                    "Task 4":loss_func[3]( outputs[3], labels[3] ).item()}, w_idx)

        torch.save({
                    'epoch': epoch,
                    'steps': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, ochk_prefix+ f".{epoch:02d}.pt")

        

    history_writer.close()
    writer.close()


def main():
    parser = ArgumentParser(description="Train a Clair3 model")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--bin_fn', type=str, default="", required=True,
                        help="Binary tensor input generated by Tensor2Bin.py, support multiple bin readers using pytables")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model to resume training or for fine-tuning")

    parser.add_argument('--ochk_prefix', type=str, default=None, required=True,
                        help="Prefix for model output after each epoch")

    parser.add_argument('--iteration', type=int, default=None,
                        help="Number of iterations per epoch")

    # semisupervised domain adaptation, load target data and use the value
    parser.add_argument('--SSDA',  type=float, default=None,
                        help=SUPPRESS)

    parser.add_argument('--ul_bin_fn', type=str, default="", 
                        help="Unlabeled Binary tensor input generated by Tensor2Bin.py, support multiple bin readers using pytables")

    # options for advanced users
    parser.add_argument('--maxEpoch', type=int, default=None,
                        help="Maximum number of training epochs")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Set the initial learning rate, default: %(default)s")


    parser.add_argument('--exclude_training_samples', type=str, default="_20,_21,_22",
                        help="Define training samples to be excluded")

    parser.add_argument('--mini_epochs', type=int, default=1,
                        help="Number of mini-epochs per epoch")

    # Internal process control
    ## In pileup training mode or not
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Add indel length for training and calling, default true for full alignment
    parser.add_argument('--add_indel_length', type=str2bool, default=False,
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
