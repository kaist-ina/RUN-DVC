import logging
import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow_addons as tfa
import tensorflow as tf
import tables
import os
import sys
from itertools import accumulate

import clair3.model as model_path
from shared.utils import str2bool

logging.basicConfig(format='%(message)s', level=logging.INFO)
tables.set_blosc_max_threads(512)
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '8'


def get_label_task(label, label_shape_cum, task):
    if task == 0:
        return label[:label_shape_cum[task]]
    elif task == len(label_shape_cum) - 1:
        return label[label_shape_cum[task - 1]:]
    else:
        return label[label_shape_cum[task - 1]:label_shape_cum[task]]


def cal_class_weight(samples_per_cls, no_of_classes, beta=0.999):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    cls_weights = (1.0 - beta) / np.array(effective_num)
    cls_weights = cls_weights / np.sum(cls_weights) * no_of_classes
    return cls_weights


class FocalLoss(tf.keras.losses.Loss):
    """
    updated version of focal loss function, for multi class classification, we remove alpha parameter, which the loss
    more stable, and add gradient clipping to avoid gradient explosion and precision overflow.
    """

    def __init__(self, label_shape_cum, task, effective_label_num=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.cls_weights = None

        if effective_label_num is not None:
            print("Effective label actviated in focalloss")
            task_label_num = get_label_task(effective_label_num, label_shape_cum, task)
            cls_weights = cal_class_weight(task_label_num, len(task_label_num))
            cls_weights = tf.constant(cls_weights, dtype=tf.float32)
            cls_weights = tf.expand_dims(cls_weights, axis=0)
            self.cls_weights = cls_weights

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-9, clip_value_max=1 - 1e-9)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = ((1 - y_pred) ** self.gamma) * y_true
        FCLoss = cross_entropy * weight
        if self.cls_weights is not None:
            FCLoss = FCLoss * self.cls_weights
        reduce_fl = tf.reduce_sum(FCLoss, axis=-1)
        return reduce_fl


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, data, chunk_list, param, tensor_shape, mini_epochs=1, add_indel_length=False, validation=False):
        self.data = data
        self.chunk_list = chunk_list
        self.batch_size = param.trainBatchSize
        self.chunk_size = param.chunk_size
        self.chunks_per_batch = self.batch_size // self.chunk_size
        self.label_shape_cum = param.label_shape_cum[0:4 if add_indel_length else 2]
        self.mini_epochs = mini_epochs
        self.mini_epochs_count = -1
        self.validation = validation
        self.position_matrix = np.empty([self.batch_size] + tensor_shape, np.int32)
        self.label = np.empty((self.batch_size, param.label_size), np.float32)
        self.random_offset = 0
        self.on_epoch_end()

    def __len__(self):
        return int((len(self.chunk_list) // self.chunks_per_batch) // self.mini_epochs)

    def __getitem__(self, index):
        mini_epoch_offset = self.mini_epochs_count * self.__len__()
        chunk_batch_list = self.chunk_list[(mini_epoch_offset + index) * self.chunks_per_batch:(mini_epoch_offset + index + 1) * self.chunks_per_batch]
        for chunk_idx, (bin_id, chunk_id) in enumerate(chunk_batch_list):
            start_pos = self.random_offset + chunk_id * self.chunk_size
            self.position_matrix[chunk_idx * self.chunk_size:(chunk_idx + 1) * self.chunk_size] = \
                self.data[bin_id].root.position_matrix[start_pos:start_pos + self.chunk_size]
                
            # if not self.validation:
            #     for i in range(chunk_idx * self.chunk_size,(chunk_idx + 1) * self.chunk_size):
            #         np.random.shuffle(self.position_matrix[i])

            self.label[chunk_idx * self.chunk_size:(chunk_idx + 1) * self.chunk_size] = \
                self.data[bin_id].root.label[start_pos:start_pos + self.chunk_size]
        # batch, 89 x 33 x 8
        return self.position_matrix, tuple(
                np.split(self.label, self.label_shape_cum, axis=1)[:len(self.label_shape_cum)]
            )

    def on_epoch_end(self):
        self.mini_epochs_count += 1
        if (self.mini_epochs_count % self.mini_epochs) == 0:
            self.mini_epochs_count = 0
            if not self.validation:
                self.random_offset = np.random.randint(0, self.chunk_size)
                np.random.shuffle(self.chunk_list)


def get_chunk_list(chunk_offset):
    """
    get chunk list for training and validation data. we will randomly split training and validation dataset,
    all training data is directly acquired from various tensor bin files.

    """
    all_shuffle_chunk_list = []
    training_chunk_list, validation_chunk_list = [], []
    for bin_idx, chunk_num in enumerate(chunk_offset):
        current_chunk_list = [(bin_idx, chunk_idx) for chunk_idx in range(chunk_num)]
        all_shuffle_chunk_list += current_chunk_list
    return np.array(all_shuffle_chunk_list)


def exist_file_prefix(exclude_training_samples, f):
    for prefix in exclude_training_samples:
        if prefix in f:
            return True
    return False


def train_model(args):
    platform = args.platform
    pileup = args.pileup
    add_indel_length = args.add_indel_length
    exclude_training_samples = args.exclude_training_samples
    exclude_training_samples = set(exclude_training_samples.split(',')) if exclude_training_samples else set()
    add_validation_dataset = args.random_validation or (args.validation_fn is not None)
    validation_fn = args.validation_fn
    if pileup:
        import shared.param_p as param
        model = model_path.Clair3_P()
    else:
        import shared.param_f as param
        model = model_path.Clair3_F(add_indel_length=add_indel_length)

    tensor_shape = param.ont_input_shape if platform == 'ont' else param.input_shape
    label_shape = param.label_shape
    label_shape_cum = param.label_shape_cum
    batch_size, chunk_size = param.trainBatchSize, param.chunk_size
    assert batch_size % chunk_size == 0
    chunks_per_batch = batch_size // chunk_size
    param.RANDOM_SEED= 123123
    random.seed(param.RANDOM_SEED)
    np.random.seed(param.RANDOM_SEED)
    learning_rate = args.learning_rate if args.learning_rate else param.initialLearningRate
    max_epoch = args.maxEpoch if args.maxEpoch else param.maxEpoch
    task_num = 4 if add_indel_length else 2
    mini_epochs = args.mini_epochs

    def populate_dataset_table(file_list, file_path):
        chunk_offset = np.zeros(len(file_list), dtype=int)
        table_dataset_list = []
        for bin_idx, bin_file in enumerate(file_list):
            table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
            table_dataset_list.append(table_dataset)
            chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
            chunk_offset[bin_idx] = chunk_num
        return table_dataset_list, chunk_offset

    bin_list = os.listdir(args.bin_fn)
    # default we exclude sample hg003 and all chr20 for training
    bin_list = [f for f in bin_list if '_20_' not in f and not exist_file_prefix(exclude_training_samples, f)]
    logging.info("[INFO] total {} training bin files: {}".format(len(bin_list), ','.join(bin_list)))

    effective_label_num = None
    
    table_dataset_list, chunk_offset = populate_dataset_table(bin_list, args.bin_fn)
        
    train_shuffle_chunk_list = get_chunk_list(chunk_offset)
    train_chunk_num = len(train_shuffle_chunk_list)

    train_seq = DataSequence(table_dataset_list, train_shuffle_chunk_list, param, tensor_shape,
        mini_epochs=mini_epochs, add_indel_length=add_indel_length)

    # evaluation metrics 
    loss_func = [FocalLoss(label_shape_cum, task, effective_label_num) for task in range(task_num)]
    loss_task = {"output_{}".format(task+1 ): loss_func[task] for task in range(task_num)}
    metrics = {"output_{}".format(task+1): tfa.metrics.F1Score(num_classes=label_shape[task], average='micro') for
               task in range(task_num)}
    total_chunks = int(sum(chunk_offset))

    # Use first 20 element to initialize tensorflow model using graph mode
    output = model(np.array(table_dataset_list[0].root.position_matrix[:20]))

    logging.info("[INFO] The size of dataset: {}".format(total_chunks * chunk_size))
    logging.info("[INFO] Start Evaluation...")

    if args.chkpnt_fn is not None:
        model.load_weights(args.chkpnt_fn)
        logging.info("[INFO] Starting from model {}".format(args.chkpnt_fn))
    else:
        print("Give checkpoint file for evalution.")
        assert(0)

    # eval
    from clair3.plot_tensor import plot_FA
    from clair3.task.gt21 import (
        GT21_Type, gt21_enum_from_label,
        HOMO_SNP_GT21, HOMO_SNP_LABELS,
        HETERO_SNP_GT21, HETERO_SNP_LABELS, GT21_LABELS, partial_label_from, mix_two_partial_labels
    )
    list_of_errors = []
    list_of_errors_labels = []
    list_of_errors_preds = []
    true_gt = { GT21_LABELS[k]: 0 for k in range(21) }
    true_zy = { k: 0 for k in range(3) }
    for i, (inputs, labels) in enumerate(train_seq, 0):
        outputs = model(inputs)

        predictions = tf.argmax(outputs[0], axis=1)
        true = tf.argmax(labels[0], axis=1)
        index_of_wrong = predictions != true
        print( "Number of wrong " ,len(inputs[index_of_wrong]) )
        if len(inputs[index_of_wrong]) > 0:
                for idx, err in enumerate(inputs[index_of_wrong]):
                    # err = np.expand_dims(err.permute(1,2,0), axis=0)*100
                    err = np.expand_dims(err, axis=0)
                    plot_FA("error_cases/Input_b{}_{}_Truth_{}_Pred_{}.png".format(i,idx, GT21_LABELS[ true[index_of_wrong][idx] ], GT21_LABELS[ predictions[index_of_wrong][idx] ]), err)

                # list_of_errors.extend(inputs[index_of_wrong])
                list_of_errors_labels.extend(true[index_of_wrong])
                list_of_errors_preds.extend(predictions[index_of_wrong])

    stat = { GT21_LABELS[k]: 0 for k in range(21) }
    for err in list_of_errors_labels:
        stat[GT21_LABELS[int(err)]] += 1
    print ( "statistics of True labels:", true_gt)
    print ( "statistics of wrong labels:", stat)
    for table_dataset in table_dataset_list:
        table_dataset.close()

    if validation_fn:
        for table_dataset in validate_table_dataset_list:
            table_dataset.close()




def main():
    parser = ArgumentParser(description="Train a Clair3 model")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--bin_fn', type=str, default="", required=True,
                        help="Binary tensor input generated by Tensor2Bin.py, support multiple bin readers using pytables")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model to resume training or for fine-tuning")

    # options for advanced users
    parser.add_argument('--maxEpoch', type=int, default=None,
                        help="Maximum number of training epochs")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Set the initial learning rate, default: %(default)s")


    parser.add_argument('--exclude_training_samples', type=str, default=None,
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
