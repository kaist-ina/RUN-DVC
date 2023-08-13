import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import tables
tables.set_blosc_max_threads(512)

from clair3.utils import setup_environment

def populate_dataset_table(file_list, file_path):
    chunk_offset = np.zeros(len(file_list), dtype=int)
    table_dataset_list = []
    for bin_idx, bin_file in enumerate(file_list):
        table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
        table_dataset_list.append(table_dataset)
        chunk_num = (len(table_dataset.root.label) - 10) // 20
        chunk_offset[bin_idx] = chunk_num
    return table_dataset_list, chunk_offset

def populate_single_dataset_table( file_path):
    chunk_offset = np.zeros(1, dtype=int)
    table_dataset_list = []
    
    table_dataset = tables.open_file(file_path, 'r')
    table_dataset_list.append(table_dataset)
    chunk_num = (len(table_dataset.root.label) - 10) // 20
    chunk_offset[0] = chunk_num
    return table_dataset_list, chunk_offset

def plot_tensor(ofn, XArray):
    plot = plt.figure(figsize=(15, 8))

    plot_min = -30
    plot_max = 30
    # plot_arr = ["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"]

    plt.subplot(1, 1, 1)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 18, 1))
    plt.imshow(XArray.T, vmin=-100, vmax=plot_max, interpolation="nearest", cmap=plt.cm.hot)
    plt.colorbar()

    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)

def plot_FA(ofn, XArray):
    plot = plt.figure(figsize=(15, 8))

    plot_min = -100
    plot_max = 100

    PLOT_NAMES=["Reference Bases", "Alernative Bases", "Strandinfo", "Map Qual", "Base Qual", "Candidate Proportion", "Insertion", "Phase"]

    for i in range(XArray.shape[3]):
        # print(np.max(XArray[0,:,:,i], axis=0 ))
        plt.subplot(2, 4, i+1)
        plt.gca().set_title(PLOT_NAMES[i])
        plt.xticks(np.arange(0, XArray.shape[2], 5))
        # plt.yticks(np.arange(0, 8, 1), plot_arr)
        plt.imshow(XArray[0, :, :, i], vmin=plot_min, vmax=plot_max, interpolation="nearest", cmap=plt.cm.hot)
        plt.colorbar()

    # plot_arr = ["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"]
    # plt.subplot(4, 2, 1)
    # plt.xticks(np.arange(0, 33, 1))
    # plt.yticks(np.arange(0, 8, 1), plot_arr)
    # plt.imshow(XArray[0, :, :, 0].transpose(), vmin=0, vmax=plot_max, interpolation="nearest", cmap=plt.cm.hot)
    # plt.colorbar()

    # plt.subplot(4, 2, 2)
    # plt.xticks(np.arange(0, 33, 1))
    # plt.yticks(np.arange(0, 8, 1), plot_arr)
    # plt.imshow(XArray[0, :, :, 1].transpose(), vmin=plot_min, vmax=plot_max, interpolation="nearest", cmap=plt.cm.bwr)
    # plt.colorbar()

    plot.savefig(ofn, dpi=300, transparent=False, bbox_inches='tight')
    plt.close(plot)

def get_chunk_list(chunk_offset, train_chunk_num, chunks_per_batch=10, training_dataset_percentage=None):
    """
    get chunk list for training and validation data. we will randomly split training and validation dataset,
    all training data is directly acquired from various tensor bin files.

    """
    need_split_validation_data = training_dataset_percentage is not None
    all_shuffle_chunk_list = []
    training_chunk_list, validation_chunk_list = [], []
    for bin_idx, chunk_num in enumerate(chunk_offset):
        current_chunk_list = [(bin_idx, chunk_idx) for chunk_idx in range(chunk_num)]
        all_shuffle_chunk_list += current_chunk_list
        if need_split_validation_data:
            buffer_chunk_num = chunks_per_batch
            if chunk_num < buffer_chunk_num:
                training_chunk_list += [(bin_idx, chunk_idx) for chunk_idx in range(chunk_num)]
                continue

            training_chunk_num = int((chunk_num - buffer_chunk_num) * training_dataset_percentage)
            validation_chunk_num = int(chunk_num - buffer_chunk_num - training_chunk_num)
            if training_chunk_num > 0:
                training_chunk_list += current_chunk_list[:training_chunk_num]
            if validation_chunk_num > 0:
                validation_chunk_list += current_chunk_list[-validation_chunk_num:]

    if need_split_validation_data:
        return np.array(training_chunk_list), np.array(validation_chunk_list)

    return np.array(all_shuffle_chunk_list[:train_chunk_num]), np.array(all_shuffle_chunk_list[train_chunk_num:])

def getFAinput(data ,chunk_list, tensor_shape, index):
    chunk_batch_list = chunk_list[index]
    start_pos = chunk_batch_list[1] * 10
    bin_id = chunk_batch_list[0]
    position_matrix = np.empty([10] + tensor_shape, np.int32)
    position_matrix = data[bin_id].root.position_matrix[start_pos:start_pos + 10]
    # batch, 89 x 33 x 8
    if 0:
        for i in range(10):
            np.random.shuffle(position_matrix[i])
            #position_matrix[i] = np.random.permutation(position_matrix[i].reshape(position_matrix.shape[1:]))
    return position_matrix

def getSingleFAinput(data , tensor_shape, index):
    position_matrix = np.empty([1] + tensor_shape, np.int32)

    for idx, pos_info in enumerate(data[0].root.position):
        # if "I" in str(data[0].root.alt_info[idx]):
        #     break
        if str(index) in str(pos_info).split(":")[1]:
            break
    print( len(data[0].root.position) , len(data[0].root.label[idx]) ,index, data[0].root.position[idx] , data[0].root.alt_info[idx])
    # position_matrix = data[0].root.position_matrix[idx]
    # batch, 89 x 33 x 8
    if 0:
        np.random.shuffle(position_matrix)
        #position_matrix[i] = np.random.permutation(position_matrix[i].reshape(position_matrix.shape[1:]))

    if 0: # shift view augmentation
        # shift view is for ssl only, do not need label of variant length
        shift = 15
        
        input_size = (len(data[0].root.label[idx]) - 24)//2

        variant_length = max (0, np.argmax(data[0].root.label[idx][24:24+input_size]) - (input_size-1)//2, np.argmax(data[0].root.label[idx][24+input_size:]) -(input_size-1)//2 )
        print(variant_length)
        new_label = np.empty([1] + [90], np.int32)
        new_label[0][:24] = data[0].root.label[idx] [:24]

        pad = ((len(data[0].root.label[idx]) - 24)//2 - 33) //2
        assert(abs(shift) <= abs(pad))
        new_label[0][24:24+33] = data[0].root.label[idx][24+pad -shift: 24+65-pad-shift]
        print(len(data[0].root.label[idx]))
        new_label[0][24+33: ] = data[0].root.label[idx][24+65+pad-shift: 154-pad-shift]
        
        print(np.argmax(new_label[0][:21]), np.argmax(new_label[0][21:24]), np.argmax(new_label[0][24:57]), np.argmax(new_label[0][57:]), pad)

        position_matrix = data[0].root.position_matrix[:,:,pad-shift:-pad-shift,:]
    else:
        position_matrix = data[0].root.position_matrix[idx]

    
    return position_matrix

def create_png(args):
    # Use sequence_dataset_DA in RUNDVC folder instead
    assert(0)


def ParseArgs():
    parser = ArgumentParser(
        description="Visualize tensors and hidden layers in PNG")

    parser.add_argument('--array_fn', type=str, default="vartensors",
                        help="Array input")

    parser.add_argument('--bin_fn', type=str, default="tensor_bin",
                        help="Array input")

    parser.add_argument('--name', type=str, default=None,
                        help="output name")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = ParseArgs()
    setup_environment()
    create_png(args)


if __name__ == "__main__":
    main()
