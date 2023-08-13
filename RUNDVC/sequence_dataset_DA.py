from copy import deepcopy
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
import torch
import tables
import os
import numpy as np
import logging

tables.set_blosc_max_threads(512)
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

from argparse import ArgumentParser
import sys 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from augmentations  import *

import random


def get_dataset_size(bin_fn, exclude_training_samples="Nonerundvc"):

    def populate_dataset_table(file_list, file_path):
        total = 0
        for bin_idx, bin_file in enumerate(file_list):
            try:
                table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
            except Exception as e:
                print("Error while opening bin files!", e)
                exit(1)
            total += len(table_dataset.root.label)
            table_dataset.close()
        return total

    bin_list = os.listdir(bin_fn)
    # default we exclude sample hg003 and all chr20 for training
    bin_list = [f for f in bin_list if  not exist_file_suffix(exclude_training_samples, f)]
    # logging.info("[INFO] total {} training bin files: ".format(len(bin_list), ))

    
    return populate_dataset_table(bin_list, bin_fn)


class SequenceDataset(Dataset):
    def __init__(self, param, bin_fn, bin_fn_tl=None, tl_size=None, seed=1, platform="ilmn", shuffle=True, validation_fn=None, exclude_training_samples=None,random_validation=False, device="cpu", batch_size=None, mini_epochs=1, add_indel_length=False, validation=False, augmentation=0, unlabeled=False):
        self.device = device
        self.param = param
        self.unlabeled = unlabeled
        if batch_size is None:
            self.batch_size = param.trainBatchSize # 2000
        else:
            self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = param.chunk_size # 200
        self.random_validation = random_validation
        self.exclude_training_samples = exclude_training_samples
        self.validation_fn = validation_fn
        self.platform = platform
        self.bin_fn = bin_fn
        self.bin_fn_tl = bin_fn_tl
        self.tl_size = tl_size
        self.add_indel_length = add_indel_length
        assert self.batch_size % self.chunk_size == 0
        self.normalize_input = np.float32(param.NORMALIZE_NUM)
        self.normalize_input_pileup = np.float32(param.NORMALIZE_NUM) # np.float32( 89 if self.platform == "ont" else 55 )
        self.random_offset = 0
        self.augmentation = augmentation
        self.validation = validation
        self.mini_epochs = mini_epochs
        self.mini_epochs_count = -1
        self.seed = seed # random seed for dataset shuffling
        self.get_chunk_length(self.param, self.bin_fn, bin_fn_tl=self.bin_fn_tl, tl_size=self.tl_size, platform=self.platform, validation_fn=self.validation_fn, 
                exclude_training_samples=self.exclude_training_samples,random_validation=self.random_validation, 
                validation=self.validation)
    def get_chunk_length(self, param, bin_fn,bin_fn_tl=None, tl_size=None, platform="ilmn", validation_fn=None, exclude_training_samples=None,
        random_validation=False, add_indel_length=False, batch_size=None, validation=False):
        exclude_training_samples = set(exclude_training_samples.split(',')) if exclude_training_samples else set()
        add_validation_dataset = random_validation or (validation_fn is not None)
        batch_size, chunk_size = param.trainBatchSize, param.chunk_size
        assert batch_size % chunk_size == 0
        chunks_per_batch = batch_size // chunk_size
        def populate_dataset_table(file_list, file_path):
            batch_size, chunk_size = param.trainBatchSize, param.chunk_size
            chunk_offset = np.zeros(len(file_list), dtype=int)
            chunk_length = 0
            for bin_idx, bin_file in enumerate(file_list):
                try:
                    table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
                except Exception as e:
                    print("Error while opening bin files!", e)
                    exit(1)
                chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
                table_dataset.close()
                chunk_offset[bin_idx] = chunk_num
            return chunk_offset

         
        def populate_tl_dataset_table(file_list, file_path, tl_file_list, tl_file_path):
            batch_size, chunk_size = param.trainBatchSize, param.chunk_size
            chunk_offset = np.zeros(len(file_list) + len(tl_file_list), dtype=int)
            for bin_idx, bin_file in enumerate(file_list):
                try:
                    table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
                except Exception as e:
                    print("Error while opening bin files!", e)
                    exit(1)
                chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
                table_dataset.close()
                chunk_offset[bin_idx] = chunk_num
            for bin_idx_2, bin_file in enumerate(tl_file_list):
                try:
                    table_dataset = tables.open_file(os.path.join(tl_file_path, bin_file), 'r')
                except Exception as e:
                    print("Error while opening bin files!", e)
                    exit(1)
                chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
                table_dataset.close()
                chunk_offset[bin_idx_2+bin_idx+1] = chunk_num
            
            return chunk_offset
            
        bin_list = os.listdir(bin_fn)
        # default we exclude sample hg003 and all chr20 for training
        bin_list = [f for f in bin_list if  not exist_file_suffix(exclude_training_samples, f)]
        logging.info("[INFO] total {} training bin files: ".format(len(bin_list), ))
        num_source_labeled_files = len(bin_list)

        if bin_fn_tl:
            bin_tl_list = os.listdir(bin_fn_tl)
            bin_tl_list = [f for f in bin_tl_list if  not exist_file_suffix(exclude_training_samples, f)]
            num_target_labeled_files = len(bin_tl_list)
            chunk_offset = populate_tl_dataset_table(bin_list, bin_fn, bin_tl_list, bin_fn_tl)
        else:
            chunk_offset = populate_dataset_table(bin_list, bin_fn)

        if validation_fn:
            val_list = os.listdir(validation_fn)
            logging.info("[INFO] total {} validation bin files: {}".format(len(val_list), ','.join(val_list)))
            validate_chunk_offset = populate_dataset_table(val_list, validation_fn)
            train_chunk_num = int(sum(chunk_offset))
            train_shuffle_chunk_list, _ = get_chunk_list(chunk_offset, train_chunk_num)
            validate_chunk_num = int(sum(validate_chunk_offset))
            validate_shuffle_chunk_list, _ = get_chunk_list(validate_chunk_offset, validate_chunk_num)
            total_chunks = train_chunk_num + validate_chunk_num
        else:
            total_chunks = int(sum(chunk_offset))
            training_dataset_percentage = param.trainingDatasetPercentage if add_validation_dataset else None
            if add_validation_dataset:
                total_batches = total_chunks // chunks_per_batch
                validate_chunk_num = int(max(1., np.floor(total_batches * (1 - training_dataset_percentage))) * chunks_per_batch)
                train_chunk_num = int(total_chunks - validate_chunk_num)
            else:
                train_chunk_num = total_chunks
            train_shuffle_chunk_list, validate_shuffle_chunk_list = get_chunk_list(chunk_offset, train_chunk_num, chunks_per_batch, training_dataset_percentage)
            if tl_size:
                train_tl = [ i_ for i_ in train_shuffle_chunk_list if i_[0] >= num_source_labeled_files ][:int(tl_size//chunk_size)]
                train_shuffle_chunk_list = [i_ for i_ in train_shuffle_chunk_list if i_[0] < num_source_labeled_files]
                train_shuffle_chunk_list.extend(train_tl)

                validate_shuffle_chunk_list = [i_ for i_ in validate_shuffle_chunk_list if i_[0] < num_source_labeled_files]
                
        if not validation:
            self.chunk_list = list(train_shuffle_chunk_list)
        else:
            self.chunk_list = list(validate_shuffle_chunk_list)
        
        return
    
    def open_file(self, param, bin_fn, bin_fn_tl=None, tl_size=None,platform="ilmn", validation_fn=None, exclude_training_samples=None,
        random_validation=False, add_indel_length=False, batch_size=None, validation=False):
        exclude_training_samples = set(exclude_training_samples.split(',')) if exclude_training_samples else set()
        add_validation_dataset = random_validation or (validation_fn is not None)

        batch_size, chunk_size = param.trainBatchSize, param.chunk_size
        assert batch_size % chunk_size == 0
        chunks_per_batch = batch_size // chunk_size

        def populate_dataset_table(file_list, file_path):
            batch_size, chunk_size = param.trainBatchSize, param.chunk_size
            chunk_offset = np.zeros(len(file_list), dtype=int)
            table_dataset_list = []
            for bin_idx, bin_file in enumerate(file_list):
                try:
                    table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
                except Exception as e:
                    print("Error while opening bin files!", e)
                    exit(1)
                table_dataset_list.append(table_dataset)
                chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
                chunk_offset[bin_idx] = chunk_num
            return table_dataset_list, chunk_offset
        
        def populate_tl_dataset_table(file_list, file_path, tl_file_list, tl_file_path):
            batch_size, chunk_size = param.trainBatchSize, param.chunk_size
            chunk_offset = np.zeros(len(file_list) + len(tl_file_list), dtype=int)
            table_dataset_list = []
            for bin_idx, bin_file in enumerate(file_list):
                try:
                    table_dataset = tables.open_file(os.path.join(file_path, bin_file), 'r')
                except Exception as e:
                    print("Error while opening bin files!", e)
                    exit(1)
                table_dataset_list.append(table_dataset)
                chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
                chunk_offset[bin_idx] = chunk_num
            for bin_idx_2, bin_file in enumerate(tl_file_list):
                try:
                    table_dataset = tables.open_file(os.path.join(tl_file_path, bin_file), 'r')
                except Exception as e:
                    print("Error while opening bin files!", e)
                    exit(1)
                table_dataset_list.append(table_dataset)
                chunk_num = (len(table_dataset.root.label) - batch_size) // chunk_size
                chunk_offset[bin_idx_2+bin_idx+1] = chunk_num
            
            return table_dataset_list, chunk_offset

        bin_list = os.listdir(bin_fn)
        # default we exclude sample hg003 and all chr20 for training
        bin_list = [f for f in bin_list if  not exist_file_suffix(exclude_training_samples, f)]
        # logging.info("[INFO] total {} training bin files: ".format(len(bin_list), ))
        num_source_labeled_files = len(bin_list)

        if bin_fn_tl:
            bin_tl_list = os.listdir(bin_fn_tl)
            bin_tl_list = [f for f in bin_tl_list if  not exist_file_suffix(exclude_training_samples, f)]
            num_target_labeled_files = len(bin_tl_list)
            table_dataset_list, chunk_offset = populate_tl_dataset_table(bin_list, bin_fn, bin_tl_list, bin_fn_tl)
        else:
            table_dataset_list, chunk_offset = populate_dataset_table(bin_list, bin_fn)

        if validation_fn:
            val_list = os.listdir(validation_fn)
            # logging.info("[INFO] total {} validation bin files: {}".format(len(val_list), ','.join(val_list)))
            validate_table_dataset_list, validate_chunk_offset = populate_dataset_table(val_list, validation_fn)

            train_chunk_num = int(sum(chunk_offset))
            train_shuffle_chunk_list, _ = get_chunk_list(chunk_offset, train_chunk_num)

            validate_chunk_num = int(sum(validate_chunk_offset))
            validate_shuffle_chunk_list, _ = get_chunk_list(validate_chunk_offset, validate_chunk_num)
            total_chunks = train_chunk_num + validate_chunk_num
        else:
            total_chunks = int(sum(chunk_offset))
            training_dataset_percentage = param.trainingDatasetPercentage if add_validation_dataset else None
            if add_validation_dataset:
                total_batches = total_chunks // chunks_per_batch
                validate_chunk_num = int(max(1., np.floor(total_batches * (1 - training_dataset_percentage))) * chunks_per_batch)
                # +++++++++++++**----
                # +:training *:buffer -:validation
                # distribute one batch data as buffer for each bin file, avoiding shifting training data to validation data
                train_chunk_num = int(total_chunks - validate_chunk_num)
            else:
                train_chunk_num = total_chunks
            train_shuffle_chunk_list, validate_shuffle_chunk_list = get_chunk_list(chunk_offset, train_chunk_num, chunks_per_batch, training_dataset_percentage)
            
            if tl_size:
                # train_tl = [ i_ for i_ in train_shuffle_chunk_list if i_[0] >= num_source_labeled_files ][:int(tl_size//chunk_size)]
                # train_shuffle_chunk_list = [i_ for i_ in train_shuffle_chunk_list if i_[0] < num_source_labeled_files]
                # train_shuffle_chunk_list.extend(train_tl)
                validate_shuffle_chunk_list = [i_ for i_ in validate_shuffle_chunk_list if i_[0] < num_source_labeled_files]

        tensor_shape = param.ont_input_shape if platform == 'ont' else param.input_shape
        if not validation:
            self.data = table_dataset_list
            self.chunk_list = list(train_shuffle_chunk_list)
        else:
            if validation_fn: 
                self.data = validate_table_dataset_list
            else: # random_validation is given
                self.data = table_dataset_list

            if len(validate_shuffle_chunk_list) == 0:
                print("Please set random_validation on or give validation_fn as input argument")
                exit(1)
            self.chunk_list = list(validate_shuffle_chunk_list)
        
        label_shape = self.data[0].root.label[0].shape
        self.label_size = label_shape[0] #param.label_size
        self.label_shape = [21,3, (self.label_size-24)//2, (self.label_size-24)//2]
        from itertools import accumulate
        self.label_shape_cum = list(accumulate(self.label_shape))[0:4 if add_indel_length else 2]
        # youngmok: pad is the amount of pad to remove to make input to 55x33 shape
        self.pad = ((len(self.data[0].root.label[0]) - 24)//2 - 33) //2

        position_matrix_shape = self.data[0].root.position_matrix[0].shape
        self.position_matrix_size = list(position_matrix_shape)#tensor_shape
        if self.shuffle:
            self.num_source_labeled_files = num_source_labeled_files
            self.on_epoch_end()
        else:
            if self.tl_size:
                train_tl = [ i_ for i_ in self.chunk_list if i_[0] >= num_source_labeled_files ][:int(self.tl_size//self.chunk_size)]
                self.chunk_list = [i_ for i_ in self.chunk_list if i_[0] < num_source_labeled_files]
                self.chunk_list.extend(train_tl)
                
        return

    def __len__(self):

        return int(len(self.chunk_list) * self.chunk_size )

    def __getitem__(self, index):
        if not hasattr(self, 'data'):
            self.open_file(self.param, self.bin_fn, platform=self.platform, validation_fn=self.validation_fn, 
                exclude_training_samples=self.exclude_training_samples,random_validation=self.random_validation, 
                add_indel_length = self.add_indel_length,validation=self.validation)

        c_idx = index // self.chunk_size
        c_e_idx = index % self.chunk_size

        bin_id, chunk_id = self.chunk_list[c_idx]
        position_matrix = np.empty( self.position_matrix_size, np.float32)
        label = np.empty( self.label_size, np.float32)
        position_matrix = self.data[bin_id].root.position_matrix[chunk_id*self.chunk_size + c_e_idx]/ self.normalize_input
        position_matrix = torch.from_numpy(position_matrix)
        # print(position_matrix.shape, index,c_idx)
        position_matrix = position_matrix.permute(2,0,1)
        
        label = self.data[bin_id].root.label[chunk_id*self.chunk_size + c_e_idx]
        label = torch.from_numpy(label)
        
        return position_matrix, label

    def on_epoch_end(self):
        self.mini_epochs_count += 1
        if (self.mini_epochs_count % self.mini_epochs) == 0:
            self.mini_epochs_count = 0
            if not self.validation:
                # Youngmok: shuffle the list once at open file
                np.random.seed(self.seed)
                # self.random_offset = np.random.randint(0, self.chunk_size)
                if self.tl_size:
                    train_tl = [ i_ for i_ in self.chunk_list if i_[0] >= self.num_source_labeled_files ]
                    np.random.shuffle(train_tl)
                    train_tl = train_tl[:int(self.tl_size//self.chunk_size)]
                    self.chunk_list = [i_ for i_ in self.chunk_list if i_[0] < self.num_source_labeled_files]
                    self.chunk_list.extend(train_tl)
                    np.random.shuffle(self.chunk_list)
                else:
                    np.random.shuffle(self.chunk_list)

from collections import defaultdict



def prepare_data(position_matrix, label, shift1, platform, pad, label_shape, label_shape_cum, normalize_input):
    
    # strandinfo to 100 and 50, as it was
    mask = position_matrix[:,:,2] > 0
    position_matrix[:,:,2][mask] = 50
    
    mask = position_matrix[:,:,2] < 0
    position_matrix[:,:,2][mask] = -50 

    #normalize
    position_matrix = position_matrix / normalize_input

    # remove phase channel in ilmn dataset
    if platform == "ilmn":
        position_matrix = np.delete(position_matrix,[7],axis=2) # erase phase channel
    
    # resize label if needed
    if len(label) == 154:
        label = np.split(label, label_shape_cum, 0 )
        if -pad == shift1:
            position_matrix = position_matrix[:,pad-shift1:,:]
            del_sums = np.sum(label[2][:pad-shift1])

            label[2] = label[2][ pad-shift1:]
            
            if del_sums != 0:
                label[2][0] = 1

            del_sums = np.sum(label[3][:pad-shift1])

            label[3] = label[3][ pad-shift1:]

            if del_sums != 0:
                label[3][0] = 1

        else:          
            position_matrix = position_matrix[:,pad-shift1:-pad-shift1,:]
            assert(pad- shift1 >= 0)
            assert(pad + shift1 > 0)
            
            ins_sums = np.sum(label[2][-pad-shift1:])
            del_sums = np.sum(label[2][:pad-shift1])

            label[2] = label[2][ pad-shift1:-pad-shift1]
            
            if ins_sums != 0:
                label[2][-1] = 1
            if del_sums != 0:
                label[2][0] = 1

            ins_sums = np.sum(label[3][-pad-shift1:])
            del_sums = np.sum(label[3][:pad-shift1])

            label[3] = label[3][ pad-shift1:-pad-shift1]

            if ins_sums != 0:
                label[3][-1] = 1
            if del_sums != 0:
                label[3][0] = 1

        label = label[:4]

    elif len(label) == 4:
        pass
    else:
        assert(shift1 == 0)
        label = torch.from_numpy(label)
        label = torch.split(label, label_shape, 0 )
    # position_matrix = np.delete(position_matrix,[5,7],axis=2) # erase phase channel
    position_matrix = torch.from_numpy(position_matrix)#.to(self.device)
    position_matrix = position_matrix.permute(2,0,1)
    return position_matrix, label


class SequenceDatasetWithAugmentation(SequenceDataset):
    def __len__(self):
        return int(len(self.chunk_list) * self.chunk_size  * (self.param.UNLABELED_SCALER if self.unlabeled and not self.validation else 1) )

    def __getitem__(self, index):
        if not hasattr(self, 'data'):
            self.open_file(self.param, self.bin_fn, self.bin_fn_tl, tl_size=self.tl_size, platform=self.platform, validation_fn=self.validation_fn, 
                exclude_training_samples=self.exclude_training_samples,random_validation=self.random_validation, 
                add_indel_length = self.add_indel_length, validation=self.validation)
            self.randaug = RandAugment(self.param.randaug_num,self.param.randaug_intensity, self.platform)
        index = index % (int(len(self.chunk_list) * self.chunk_size ))
        
        c_idx = index // self.chunk_size
        c_e_idx = index % self.chunk_size

        bin_id, chunk_id = self.chunk_list[c_idx]
        idx = chunk_id*self.chunk_size + c_e_idx
        position_matrix = np.zeros( self.position_matrix_size, np.float32)
        position_matrix = self.data[bin_id].root.position_matrix[idx].copy()

        label = np.empty( self.label_size, np.float32)
        label = self.data[bin_id].root.label[idx]

        input_depth = 89 if self.platform == "ont" else 55

        shift1 = 0

        if self.augmentation == 0 or self.validation:
            # position_matrix = self.data[bin_id].root.position_matrix[idx]
            pass
        
        elif self.augmentation == 1 :
            # weak augmentation
            
            if random.random() < 0.5:
                position_matrix, label = row_drop( position_matrix , label,\
                                        self.data[bin_id].root.alt_info[idx], 3, shuffle=False, v_shift=True)
            
        elif self.augmentation == 2 :
            
            position_matrix, label = row_drop( position_matrix , label,\
                                            self.data[bin_id].root.alt_info[idx], 7, shuffle=False, v_shift=True)

            if self.param.randaug_num > 0:
                position_matrix, label = self.randaug(position_matrix , label, self.data[bin_id].root.alt_info[idx])
            

        position_matrix, label = prepare_data(position_matrix, label, shift1, self.platform, self.pad, self.label_shape, self.label_shape_cum, self.normalize_input)
        
        return position_matrix, label

def exist_file_suffix(exclude_training_samples, f):
    for prefix in exclude_training_samples:
        if f.endswith(prefix):
            return True
    return False

def exist_file_prefix(exclude_training_samples, f):
    for prefix in exclude_training_samples:
        if prefix in f:
            return True
    return False

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
        return training_chunk_list, validation_chunk_list

    return all_shuffle_chunk_list[:train_chunk_num],all_shuffle_chunk_list[train_chunk_num:]


def Get_Dataloader_augmentation(param, bin_fn, num_workers, batch_size=2000 , unlabeled=False, shuffle=True, device="cuda", platform="ilmn", pin_memory=True, validation_fn=None, exclude_training_samples=None, random_validation=False, add_indel_length=True, mini_epochs=1, augmentation=2):

    '''
    bin_fn: directory where tensor bin files are in
    shuffle: shuffle training data
    num_workers: worker num for dataloader


    Youngmok:
    
    Data is loaded using pytables library.

    Datas are segmented into chunks. Each chunk contains (train data file index, index of chunk), chunk-size is 200.

    Train/Validation is determined by dividing the chunks
    '''
    # youngmok: use SequenceDatasetBatch, which is faster by loading chunk at a single iteration
    train_seq_weak = SequenceDatasetWithAugmentation(param,bin_fn, unlabeled=unlabeled,  shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=augmentation)

    if not unlabeled:
        train_seq_strong = SequenceDatasetWithAugmentation(param,bin_fn, unlabeled=unlabeled,  platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
            validation= True, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=augmentation)
    else:
        train_seq_strong = SequenceDatasetWithAugmentation(param,bin_fn, unlabeled=unlabeled, shuffle=shuffle,  platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
            validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=2)

    NUM_WORKERS = num_workers
    train_loader = DataLoader(train_seq_weak,  batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False,pin_memory = pin_memory)

    train_loader_strong = DataLoader(dataset=train_seq_strong, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory = pin_memory)

    return train_loader, train_loader_strong

def Get_Dataloader_DA(param, bin_fn, bin_fn_ul, num_workers, bin_fn_tl=None, tl_size=None, seed=1, batch_size=1000 ,shuffle=True, device="cuda", platform="ilmn", pin_memory=True, validation_fn=None, exclude_training_samples=None, random_validation=False, add_indel_length=True, mini_epochs=1, augmentation=2):

    '''
    bin_fn: directory where tensor bin files are in
    shuffle: shuffle training data
    num_workers: worker num for dataloader

    Youngmok:
    
    Data is loaded using pytables library.

    Datas are segmented into chunks. Each chunk contains (train data file index, index of chunk), chunk-size is 200.

    Train/Validation is determined by dividing the chunks
    '''

    NUM_WORKERS = num_workers
    target_batch_size = batch_size * param.u_ratio
    # youngmok: use SequenceDatasetBatch, which is faster by loading chunk at a single iteration
    source_dataset_train_weak  = SequenceDatasetWithAugmentation(param,bin_fn,bin_fn_tl=bin_fn_tl, tl_size=tl_size, seed=seed,  shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=1)
    source_dataset_train_strong  = SequenceDatasetWithAugmentation(param,bin_fn,bin_fn_tl=bin_fn_tl, tl_size=tl_size, seed=seed, shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=2)
    source_dataset_train_test  = SequenceDatasetWithAugmentation(param,bin_fn,bin_fn_tl=bin_fn_tl, tl_size=tl_size, seed=seed, shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= True, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=0)

    source_dataloader_train_weak = DataLoader(source_dataset_train_weak,  batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False,pin_memory = pin_memory)
    source_dataloader_train_strong = DataLoader(source_dataset_train_strong,  batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False,pin_memory = pin_memory)
    source_dataloader_test = DataLoader(source_dataset_train_test,  batch_size=batch_size, num_workers=max(1,NUM_WORKERS-3), shuffle=False,pin_memory = pin_memory)

    target_dataset_train_weak  = SequenceDatasetWithAugmentation(param,bin_fn_ul, unlabeled=True, seed=seed, shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=1)
    target_dataset_train_strong  = SequenceDatasetWithAugmentation(param,bin_fn_ul,unlabeled=True, seed=seed, shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=2)
    target_dataset_train_test  = SequenceDatasetWithAugmentation(param,bin_fn_ul,unlabeled=True,  seed=seed,shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= True, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=0)

    target_dataloader_train_weak  = DataLoader(target_dataset_train_weak,  batch_size=target_batch_size, num_workers=NUM_WORKERS, shuffle=False,pin_memory = pin_memory)
    target_dataloader_train_strong = DataLoader(target_dataset_train_strong,  batch_size=target_batch_size, num_workers=NUM_WORKERS, shuffle=False,pin_memory = pin_memory)
    target_dataloader_test = DataLoader(target_dataset_train_test,  batch_size=target_batch_size, num_workers=max(1,NUM_WORKERS-3), shuffle=False,pin_memory = pin_memory)


    return (source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test), (target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test)

def prepare_data_pileup(position_matrix, label, platform, pad, label_shape, label_shape_cum, normalize_input):
    #normalize
    position_matrix = position_matrix / normalize_input
    
    # resize label if needed
    shift1=0
    if len(label) == 154:
        label = np.split(label, label_shape_cum, 0 )
        position_matrix = position_matrix[pad-shift1:-pad-shift1,:]
        assert(pad- shift1 >= 0)
        assert(pad + shift1 > 0)
        
        ins_sums = np.sum(label[2][-pad-shift1:])
        del_sums = np.sum(label[2][:pad-shift1])

        label[2] = label[2][ pad-shift1:-pad-shift1]
        
        if ins_sums != 0:
            label[2][-1] = 1
        if del_sums != 0:
            label[2][0] = 1

        ins_sums = np.sum(label[3][-pad-shift1:])
        del_sums = np.sum(label[3][:pad-shift1])

        label[3] = label[3][ pad-shift1:-pad-shift1]

        if ins_sums != 0:
            label[3][-1] = 1
        if del_sums != 0:
            label[3][0] = 1

        label = label[:4]

    elif len(label) == 4:
        pass
    else:
        assert(shift1 == 0)
        label = torch.from_numpy(label)
        label = torch.split(label, label_shape, 0 )
    # position_matrix = np.delete(position_matrix,[5,7],axis=2) # erase phase channel
    position_matrix = torch.from_numpy(position_matrix).type(torch.FloatTensor)#.to(self.device)
    return position_matrix, label

class SequenceDatasetWithAugmentation_pileup(SequenceDataset):
    def __len__(self):
        return int(len(self.chunk_list) * self.chunk_size  * (self.param.UNLABELED_SCALER if self.unlabeled and not self.validation else 1) )
    
    def __getitem__(self, index):
        if not hasattr(self, 'data'):
            self.open_file(self.param, self.bin_fn, platform=self.platform, validation_fn=self.validation_fn, 
                exclude_training_samples=self.exclude_training_samples,random_validation=self.random_validation, 
                add_indel_length = self.add_indel_length, validation=self.validation)
            # self.existkey = {}
        c_idx = index // self.chunk_size
        c_e_idx = index % self.chunk_size

        bin_id, chunk_id = self.chunk_list[c_idx]
        position_matrix = np.zeros( self.position_matrix_size, np.float32)
        position_matrix = self.data[bin_id].root.position_matrix[chunk_id*self.chunk_size + c_e_idx]

        label = np.empty( self.label_size, np.float32)
        label = self.data[bin_id].root.label[chunk_id*self.chunk_size + c_e_idx]
        # print(position_matrix.shape)

        position_matrix, label = prepare_data_pileup(position_matrix, label, self.platform, self.pad, self.label_shape, self.label_shape_cum, self.normalize_input_pileup)
        
        return position_matrix, label

def Get_Dataloader_augmentation_pileup(param, bin_fn, num_workers, unlabeled=False, batch_size=2000 ,shuffle=True, device="cuda", platform="ilmn", pin_memory=True, validation_fn=None, exclude_training_samples=None, random_validation=False, add_indel_length=True, mini_epochs=1, augmentation=2):

    '''
    bin_fn: directory where tensor bin files are in
    shuffle: shuffle training data
    num_workers: worker num for dataloader


    Youngmok:
    
    Data is loaded using pytables library.

    Datas are segmented into chunks. Each chunk contains (train data file index, index of chunk), chunk-size is 200.

    Train/Validation is determined by dividing the chunks
    '''
    # youngmok: use SequenceDatasetBatch, which is faster by loading chunk at a single iteration
    train_seq = SequenceDatasetWithAugmentation_pileup(param,bin_fn,unlabeled=unlabeled, batch_size=batch_size, shuffle=shuffle, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= False, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=augmentation)

    val_seq = SequenceDatasetWithAugmentation_pileup(param,bin_fn, batch_size=batch_size, platform=platform, validation_fn=validation_fn, exclude_training_samples=exclude_training_samples, random_validation=random_validation,
        validation= True, mini_epochs=mini_epochs, add_indel_length=add_indel_length, augmentation=augmentation)

    NUM_WORKERS = num_workers

    train_loader = DataLoader(train_seq,  batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False,pin_memory = pin_memory)

    val_loader = DataLoader(dataset=val_seq,  shuffle=False, num_workers=NUM_WORKERS,batch_size=batch_size,pin_memory = pin_memory)

    return train_loader, val_loader


def plot_FA(ofn, XArray):
    plot = plt.figure(figsize=(10, 8))

    plot.subplots_adjust(wspace=0.2)
    
    plot_min = -75
    plot_max = 75

    PLOT_NAMES=["Reference Bases", "Mutations", "Strand", "Mapping Quality", "Base Quality", "Target Mutation", "Insertion Bases", "Phasing"]

    for i in range(XArray.shape[3]):
        # print(np.max(XArray[0,:,:,i], axis=0 ))
        plt.subplot(2, 4, i+1)
        plt.gca().set_title(PLOT_NAMES[i])
        plt.xticks(np.arange(0, XArray.shape[2], 8))
        # plt.yticks(np.arange(0, 8, 1), plot_arr)
        plt.imshow(XArray[0, :, :, i], vmin=plot_min, vmax=plot_max, interpolation="none", cmap=plt.cm.hot)
        # if i+1 % 4 == 0:
        #     plt.colorbar(orientation="vertical")

    plot.savefig(ofn, dpi=300, transparent=False, bbox_inches='tight')
    plt.close(plot)

GT21_LABELS = [
    'AA',
    'AC',
    'AG',
    'AT',
    'CC',
    'CG',
    'CT',
    'GG',
    'GT',
    'TT',
    'DelDel',
    'ADel',
    'CDel',
    'GDel',
    'TDel',
    'InsIns',
    'AIns',
    'CIns',
    'GIns',
    'TIns',
    'InsDel'
]
import tqdm
def validation_DA(args):
    print("Validate data from DA")
    import params.param_rundvc as param
    tensor_shape = param.input_shape
    device = "cpu"
    
    data  = Get_Dataloader_DA(param, args.bin_fn, args.ul_bin_fn, 8, platform=args.platform,batch_size= 1000, device=device, pin_memory=False, shuffle=False)
    source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
    target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]
    

    dataset = zip(source_dataloader_train_weak, source_dataloader_train_strong, target_dataloader_train_weak, target_dataloader_train_strong)

    # this is where the unsupervised learning comes in, as such, we're not interested in labels
    total_data_len = min(len(source_dataloader_train_weak), len(source_dataloader_train_strong), len(target_dataloader_train_weak), len(target_dataloader_train_strong))
    pbar = tqdm.tqdm(dataset)
    i = 0
    for (data_source_weak, label_source), (data_source_strong, label_source_strong), (data_target_weak, labels_target), (data_target_strong, _) in pbar:
        i += 1
        for idx, FAI in enumerate(zip(data_source_weak, data_source_strong, data_target_weak, data_target_strong )):
            
            FAI = list(FAI)            
            for iidx,II in enumerate(FAI):
                FAI[iidx] = np.expand_dims(FAI[iidx].permute(1,2,0), axis=0) * 100
            # print(idx, label[0][idx], label[1][idx], label[2][idx])

            target_label = label_source_strong
            gt21 = np.argmax(target_label[0][idx])
            zyg = np.argmax(target_label[1][idx])
            vl1 = np.argmax(target_label[2][idx])
            vl2 = np.argmax(target_label[3][idx])
            if np.sum(FAI[0]) <= 0.01:
                print(i , idx, vl1, vl2, GT21_LABELS[gt21])
                plot_FA(args.name+"/{}_{}_S_gt_{}_zg_{}_vl_{}_{}.jpg".format(i, idx, GT21_LABELS[gt21], zyg, vl1, vl2), FAI[1])
                
                target_label = label_source
                gt21 = np.argmax(target_label[0][idx])
                zyg = np.argmax(target_label[1][idx])
                vl1 = np.argmax(target_label[2][idx])
                vl2 = np.argmax(target_label[3][idx])
                plot_FA(args.name+"/{}_{}_W_gt_{}_zg_{}_vl_{}_{}.jpg".format(i, idx, GT21_LABELS[gt21], zyg, vl1, vl2), FAI[0])
            pbar.set_postfix( {"Total Iters": total_data_len, "epoch": i})
            
    return 


def create_png_SSDA(args):
    print("Plot Input for DA")
    import params.param_rundvc  as param
    tensor_shape = param.input_shape
    device = "cpu"
    AUGMENTATION = 0
    param.trainingDatasetPercentage =  0.3
    source_dataloader_train_weak, source_dataloader_train_strong  = Get_Dataloader_augmentation(param, 
            args.bin_fn, 8,  unlabeled=True, platform=args.platform, random_validation=True,
            batch_size= 3400, device=device, pin_memory=False, augmentation= AUGMENTATION)

    dataset = zip(source_dataloader_train_weak, source_dataloader_train_strong)

    # this is where the unsupervised learning comes in, as such, we're not interested in labels
    i = 0
    for (data_source_weak, label_source), (data_source_strong, label_source_strong) in dataset:
        i += 1
        for idx, FAI in enumerate(zip(data_source_weak, data_source_strong )):
            # assert( not (label_source != label_source_strong).any() )
            if idx != 1:
                continue
            FAI = list(FAI)            
            for iidx,II in enumerate(FAI):
                FAI[iidx] = np.expand_dims(FAI[iidx].permute(1,2,0), axis=0) * 100
            # print(idx, label[0][idx], label[1][idx], label[2][idx])

            target_label = label_source_strong
            gt21 = np.argmax(target_label[0][idx])
            zyg = np.argmax(target_label[1][idx])
            vl1 = np.argmax(target_label[2][idx])
            vl2 = np.argmax(target_label[3][idx])

            if gt21 >= 0:
                print(i , idx, vl1, vl2, GT21_LABELS[gt21])
                plot_FA(args.name+"/{}_{}_S_gt_{}_zg_{}_vl_{}_{}.jpg".format(i, idx, GT21_LABELS[gt21], zyg, vl1, vl2), FAI[1])
                
                target_label = label_source
                gt21 = np.argmax(target_label[0][idx])
                zyg = np.argmax(target_label[1][idx])
                vl1 = np.argmax(target_label[2][idx])
                vl2 = np.argmax(target_label[3][idx])
                plot_FA(args.name+"/{}_{}_W_gt_{}_zg_{}_vl_{}_{}.jpg".format(i, idx, GT21_LABELS[gt21], zyg, vl1, vl2), FAI[0])



def create_png_DA(args):
    print("Plot Input for DA")
    import params.param_rundvc  as param
    tensor_shape = param.input_shape
    device = "cpu"
    
    data  = Get_Dataloader_DA(param, args.bin_fn, args.ul_bin_fn, 8, platform=args.platform,batch_size= 1000, device=device, pin_memory=False, shuffle=False)
    source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
    target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]

    dataset = zip(source_dataloader_train_weak, source_dataloader_train_strong, target_dataloader_train_weak, target_dataloader_train_strong)

    # this is where the unsupervised learning comes in, as such, we're not interested in labels
    i = 0
    for (data_source_weak, label_source), (data_source_strong, label_source_strong), (data_target_weak, labels_target), (data_target_strong, _) in dataset:
        i += 1
        for idx, FAI in enumerate(zip(data_source_weak, data_source_strong, data_target_weak, data_target_strong )):
            
            FAI = list(FAI)            
            for iidx,II in enumerate(FAI):
                FAI[iidx] = np.expand_dims(FAI[iidx].permute(1,2,0), axis=0) * 100
            # print(idx, label[0][idx], label[1][idx], label[2][idx])

            target_label = label_source_strong
            gt21 = np.argmax(target_label[0][idx])
            zyg = np.argmax(target_label[1][idx])
            vl1 = np.argmax(target_label[2][idx])
            vl2 = np.argmax(target_label[3][idx])

            if gt21 >= 10:
                print(i , idx, vl1, vl2, GT21_LABELS[gt21])
                plot_FA(args.name+"/{}_{}_S_gt_{}_zg_{}_vl_{}_{}.jpg".format(i, idx, GT21_LABELS[gt21], zyg, vl1, vl2), FAI[1])
                
                target_label = label_source
                gt21 = np.argmax(target_label[0][idx])
                zyg = np.argmax(target_label[1][idx])
                vl1 = np.argmax(target_label[2][idx])
                vl2 = np.argmax(target_label[3][idx])
                plot_FA(args.name+"/{}_{}_W_gt_{}_zg_{}_vl_{}_{}.jpg".format(i, idx, GT21_LABELS[gt21], zyg, vl1, vl2), FAI[0])
            # plot_FA(args.name+"/B{}_{}_source_weak.png".format(i, idx), FAI[0])
            # plot_FA(args.name+"/B{}_{}_source_strong.png".format(i, idx), FAI[1])
            # plot_FA(args.name+"/B{}_{}_target_weak.png".format(i, idx), FAI[2])
            # plot_FA(args.name+"/B{}_{}_target_strong.png".format(i, idx), FAI[3])
            # print("Processed {}_{}".format(i,idx) , GT21_LABELS[gt21], zyg, vl1, vl2)
    return 

def create_png_pileup(args):
    print("Plot Input for pileup")
    import params.param_rundvc  as param
    tensor_shape = param.input_shape
    device = "cpu"
    
    train_loader, val_loader  = Get_Dataloader_augmentation_pileup(param, args.bin_fn, 8, platform=args.platform,batch_size= 1000, device=device, pin_memory=False, shuffle=False)
    plot_arr = ('A', 'C', 'G', 'T', 'I', 'I1', 'D', 'D1', '*', 'a', 'c', 'g','t', 'i', 'i1','d', 'd1','#')
    for ii, (data, label) in enumerate(train_loader):
        data = data * 100
        for idx, elem in enumerate(data):
            target_label = label
            gt21 = np.argmax(target_label[0][idx])
            zyg = np.argmax(target_label[1][idx])
            vl1 = np.argmax(target_label[2][idx])
            vl2 = np.argmax(target_label[3][idx])

            plot = plt.figure(figsize=(10, 8))

            plot_min = -55
            plot_max = 55

            # plt.plot(1, 1, 0)
            plt.gca().set_title("Pileup")
            plt.xticks(np.arange(0, elem.shape[0], 8))
            plt.yticks(np.arange(0, 18, 1), plot_arr)
            plt.imshow(elem.T, vmin=plot_min, vmax=plot_max, interpolation="none", cmap=plt.cm.hot)
            plt.colorbar()

            plot.savefig(args.name+"/{}_{}_GT_{}_ZY_{}_L1_{}_L2_{}.jpg".format(ii,idx, GT21_LABELS[gt21], zyg, vl1, vl2), dpi=300, transparent=False, bbox_inches='tight')
            plt.close(plot)


def create_png_swav(args):
    print("Plot Input for SWAV")
    import params.param_rundvc  as param
    tensor_shape = param.input_shape
    device = "cpu"

    data = Get_Dataloader_SWAV( param, args.bin_fn, args.ul_bin_fn, 8, global_view=2, local_view=0, platform=args.platform, batch_size= 1000, device=device, pin_memory=False, shuffle=False)
    source_dataloader_train_weak, source_dataloader_test = data[0]
    target_dataloader_train_weak, target_dataloader_test = data[1]

    dataset = zip(source_dataloader_train_weak, target_dataloader_train_weak)

    # this is where the unsupervised learning comes in, as such, we're not interested in labels
    batch_idx = 0
    for (data_source_weak, label_source), (data_target_weak, labels_target) in dataset:
        batch_idx += 1
        for idx in range(len(data_source_weak[0])):
            for g_i in range( len(data_source_weak) ):
                FAI = data_source_weak[g_i][idx]
                FAI = np.expand_dims(FAI.permute(1,2,0), axis=0) * 100
                # print(idx, label[0][idx], label[1][idx], label[2][idx])

                target_label = label_source[g_i]
                gt21 = np.argmax(target_label[0][idx])
                zyg = np.argmax(target_label[1][idx])
                vl1 = np.argmax(target_label[2][idx])
                vl2 = np.argmax(target_label[3][idx])

                if gt21 >= 10:
                    print(batch_idx , idx, vl1, vl2, GT21_LABELS[gt21])
                    plot_FA(args.name+"/SWAV_{}_{}_globview_{}_gt_{}_zg_{}_vl_{}_{}.jpg".format(batch_idx, idx, g_i,GT21_LABELS[gt21], zyg, vl1, vl2), FAI)
                    
    return 


def ParseArgs():
    parser = ArgumentParser(
        description="Visualize tensors and hidden layers in PNG")


    parser.add_argument('--bin_fn', type=str, default="tensor_bin",
                        help="Array input")

    parser.add_argument('--ul_bin_fn', type=str, default="tensor_bin",
                        help="Array input")

    parser.add_argument('--check', action='store_true',
                        help="check number of bin_fn")

    parser.add_argument('--platform', type=str, default="ilmn",
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
    if args.check:
        size = get_dataset_size(args.bin_fn)
        print("Dataset size of",args.bin_fn,"is {}".format(size))
        return 0 
    create_png_SSDA(args)
    # create_png_DA(args)
    # create_png_pileup(args)
    # create_png_swav(args)
    # validation_DA(args)


if __name__ == "__main__":
    main()
