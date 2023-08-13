import logging
import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tables
import os
import sys
from itertools import accumulate
from shared.utils import str2bool


import clair3.model_Pytorch as model_path
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

def train_model(args):
    pileup = args.pileup
    if pileup:
        import shared.param_p as param
        # model = model_path.Clair3_P()
    else:
        import shared.param_f as param
        # model = model_path.Clair3_F(add_indel_length=add_indel_length)

    label_shape = param.label_shape
    label_shape_cum = param.label_shape_cum
    batch_size, chunk_size = param.trainBatchSize, param.chunk_size
    assert batch_size % chunk_size == 0
    chunks_per_batch = batch_size // chunk_size
    random.seed(param.RANDOM_SEED)
    np.random.seed(param.RANDOM_SEED)
    task_num = 4 if args.add_indel_length else 2
    
    device = "cpu"
        
    from clair3.sequence_dataset import Get_Dataloader, Get_Dataloader_augmentation
    if args.platform == "ilmn":
        # val_loader, _ = Get_Dataloader(param, args.bin_fn, 12, pin_memory=False, shuffle=False)
        val_loader, _ = Get_Dataloader_augmentation(param, args.bin_fn, 12,augmentation=0, pin_memory=False, platform= args.platform, shuffle=False)
    elif args.platform == "ont":
        val_loader, _ = Get_Dataloader_augmentation(param, args.bin_fn, 12, pin_memory=False, platform= args.platform, shuffle=False)
    # youngmok: important for multi-threaded data loading with tables library
    total_chunks = len(val_loader) * chunks_per_batch
    

    model = model_path.RUNDVC( [1,1,1], add_indel_length=args.add_indel_length)

    loss_func = [FocalLoss(label_shape_cum, task, None) for task in range(task_num)]

    from torchmetrics import F1Score
    f1_scores = [F1Score(num_classes=label_shape[i], average='micro') for i in range(4)]


    if args.chkpnt_fn is not None:
        print("Loaded Pretrained Model "+ args.chkpnt_fn)
        checkpoint = torch.load(args.chkpnt_fn, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Give pretrained model as input!")
        exit(1)
    logging.info("[INFO] The size of dataset: {}".format(total_chunks * chunk_size))
    logging.info("[INFO] The training batch size: {}".format(batch_size))
    logging.info("[INFO] Start Eval...")
    device = torch.device('cpu')

    from clair3.plot_tensor import plot_FA
    from clair3.task.gt21 import (
        GT21_Type, gt21_enum_from_label,
        HOMO_SNP_GT21, HOMO_SNP_LABELS,
        HETERO_SNP_GT21, HETERO_SNP_LABELS, GT21_LABELS, partial_label_from, mix_two_partial_labels
    )
    from torchsummary import summary
    # summary(model, (8,55,33)) # summary of the model
    list_of_errors = []
    list_of_errors_labels = []
    list_of_errors_preds = []
    import time
    start = time.time()

    true_gt = { GT21_LABELS[k]: 0 for k in range(21) }
    true_zy = { k: 0 for k in range(3) }


    def check_result(input,label):
        i=0
        index_of_wrong = torch.argmax(input[i], dim=1) != torch.argmax(label[i], dim=1)
        for i in range(1,4):
            index_of_wrong = index_of_wrong | ( torch.argmax(input[i], dim=1) != torch.argmax(label[i], dim=1))

        index_of_wrong = index_of_wrong & ( (torch.argmax(input[0], dim=1) >= 10) | (torch.argmax(label[0], dim=1) >=10) )
        
        return index_of_wrong

    total_preds = 0
    total_errors = 0

    stat = { GT21_LABELS[k]: 0 for k in range(21) }
    
    with torch.no_grad(): 
        model.eval()
        running_vloss = 0.0
        for i, (inputs, labels) in enumerate(val_loader, 0):
            inputs = inputs.squeeze().to(device)
            for j in range(len(labels)):
                labels[j] = labels[j].squeeze().to(device)

            outputs = model(inputs)

            predictions = torch.argmax(outputs[0], dim=1)
            true = torch.argmax(labels[0], dim=1)
            for tt in true:
                true_gt[GT21_LABELS[int(tt)]] += 1
                

            wrong_indels = check_result(outputs,labels)
            total_indels = (predictions >= 10) | (true >=10)

            for lab in predictions[total_indels]:
                stat[GT21_LABELS[lab]] += 1

            total_preds +=  len( inputs[ total_indels  ] )
            total_errors += len(inputs[ wrong_indels ] )
            print ( "Total indel Wrong: {}/ indel Total: {} / True indel: {}".format(len(inputs[ wrong_indels ] ), len( inputs[ total_indels  ] ), len(inputs[true>=10]) ) )
            print( "INDEL Acc: {} total_indels:{} total_indel_err:{} Datas: {} Number of wrong ".format(1.0 - total_errors/float(total_preds), total_preds, total_errors, len(labels[0])) ,len(inputs[wrong_indels]) )
            PRINT_ALL = True
            if PRINT_ALL:
                for idx, err in enumerate(inputs[total_indels]):
                    err = np.expand_dims(err.permute(1,2,0), axis=0)*100
                    plot_FA("error_cases/ALL_b{}_{}_True_{}_{}_{}_{}_Pred_{}_{}_{}_{}.png".format(i,idx, GT21_LABELS[ true[total_indels][idx] ], np.argmax(labels[1][total_indels][idx] ), np.argmax(labels[2][total_indels][idx] ), np.argmax(labels[3][total_indels][idx] ),\
                         GT21_LABELS[ predictions[total_indels][idx] ], np.argmax(outputs[1][total_indels][idx] ), np.argmax(outputs[2][total_indels][idx]), np.argmax(outputs[3][total_indels][idx]) ), err)

            if len(inputs[wrong_indels]) > 0:
                for idx, err in enumerate(inputs[wrong_indels]):
                    err = np.expand_dims(err.permute(1,2,0), axis=0)*100
                    plot_FA("error_cases/Input_b{}_{}_True_{}_{}_{}_{}_Pred_{}_{}_{}_{}.png".format(i,idx, GT21_LABELS[ true[wrong_indels][idx] ], np.argmax(labels[1][wrong_indels][idx] ), np.argmax(labels[2][wrong_indels][idx] ), np.argmax(labels[3][wrong_indels][idx] ),\
                         GT21_LABELS[ predictions[wrong_indels][idx] ], np.argmax(outputs[1][wrong_indels][idx] ), np.argmax(outputs[2][wrong_indels][idx]), np.argmax(outputs[3][wrong_indels][idx]) ), err)

                list_of_errors_labels.extend(true[wrong_indels])
                list_of_errors_preds.extend(predictions[wrong_indels])
            # print(stat)
            # Youngmok: should concat outputs, average f1 scores?
            loss = loss_func[0]( outputs[0], labels[0] ) + loss_func[1]( outputs[1], labels[1] ) + loss_func[2]( outputs[2], labels[2] ) + loss_func[3]( outputs[3], labels[3] )
            running_vloss += loss
            # print(f'Validation {int(time.time() - start )} loss: {loss.item():.3f} loss_1: {loss_func[0]( outputs[0], labels[0] ).item():.3f} loss_2: {loss_func[1]( outputs[1], labels[1] ).item():.3f} loss_3: {loss_func[2]( outputs[2], labels[2] ).item():.3f} loss_4: {loss_func[3]( outputs[3], labels[3] ).item():.3f} ')
            # print('Validation [F1-score] Task 1:{:.4f} Task 2:{:.4f} Task 3:{:.4f} Task 4:{:.4f}'.format( f1_scores[0](outputs[0], labels[0].int()).item(), f1_scores[1](outputs[1], labels[1].int()).item() ,f1_scores[2](outputs[2], labels[2].int()).item(), f1_scores[3](outputs[3], labels[3].int()).item() ) )

        avg_loss = running_vloss / (i + 1)
    
    
    # statistic of errors
    stat = { GT21_LABELS[k]: 0 for k in range(21) }
    for err in list_of_errors_labels:
        stat[GT21_LABELS[int(err)]] += 1
    print ( "statistics of True labels:", true_gt)
    print ( "statistics of wrong labels:", stat)
    
    # os.system("mkdir -p error_cases")
    # for idx,err in enumerate(list_of_errors):
    #     err = np.expand_dims(err.permute(1,2,0), axis=0)*100
    #     plot_FA("error_cases/Input_{}_Truth_{}_Pred_{}.png".format(idx, GT21_LABELS[ list_of_errors_labels[idx] ], GT21_LABELS[ list_of_errors_preds[idx] ]), err)


def main():
    parser = ArgumentParser(description="Eval a Clair3 model")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--bin_fn', type=str, default="", required=True,
                        help="Binary tensor input generated by Tensor2Bin.py, support multiple bin readers using pytables")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model to resume training or for fine-tuning")

    parser.add_argument('--exclude_training_samples', type=str, default=None,
                        help="Define training samples to be excluded")

    # Internal process control
    ## In pileup training mode or not
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Add indel length for training and calling, default true for full alignment
    parser.add_argument('--add_indel_length', type=str2bool, default=False,
                        help=SUPPRESS)

    # mutually-incompatible validation options
    vgrp = parser.add_mutually_exclusive_group()

    vgrp.add_argument('--validation_fn', type=str, default=None,
                        help="Binary tensor input for use in validation: %(default)s")


    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
