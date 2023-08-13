# Code Adapted from https://github.com/4uiiurz1/pytorch-auto-augment/blob/master/auto_augment.py


import random
import numpy as np
from collections import defaultdict
from itertools import accumulate
import sys

label_shape_cum = list(accumulate([21, 3, 65, 65]))[0:4 ]


class RandAugment:
    def __init__(self, n, m, platform):
        self.n = n
        self.m = m      # [0, 30]
        assert(m <= 30 and m >= 0)
        if platform=="ilmn":
            self.augment_list = [
                (random_invert, 0, 10),
                (random_noise, 0, 10),
                (point_mutation, 0, 10),
                (indel_mutation, 0, 10),
                (random_candprop_drop, 0, 10),
                (random_mapqual_max, 0, 10),
                ("pass", 0, 10),
                (row_drop_no_overlap, 0, 10),
            ]
        else:
            self.augment_list = [
                (random_invert, 0, 10),
                (random_noise, 0, 10),
                (point_mutation, 0, 10),
                (indel_mutation, 0, 10),
                (random_candprop_drop, 0, 10),
                (random_mapqual_max, 0, 10),
                (random_hap_drop, 0, 10),
                (row_drop_no_overlap, 0, 10),
                ("pass", 0, 10),
            ]

    def __call__(self, img, label, alt_info):
        ops = set(random.choices(self.augment_list, k=self.n))
        for op, minval, maxval in ops:
            if op == "pass":
                continue
            val = int( (float(self.m) / 30) * float(maxval - minval) + minval )
            img, label = op(img, label, alt_info, val)

        return img, label


def apply_policy(img, label, alt_info, policy):
    if random.random() < policy[1]:
        img, label = operations[policy[0]](img, label, alt_info, policy[2])
    if random.random() < policy[4]:
        img, label = operations[policy[3]](img, label, alt_info, policy[5])

    return img, label

def row_drop(img, label, alt_info, magnitude, shuffle=False, v_shift=True ):
    '''
    drops row, adjust the AF value. Also, shuffle rows without changing the shape
    row shuffle, horizontal shift is also managed here.
    '''
    assert(img.shape[1]==65 or img.shape[1]==33)
    img = np.array(img)
    MAX_DROP = 0.8
    magnitudes = np.linspace(0, MAX_DROP, 11)
    # portion = np.random.rand()*magnitudes[magnitude]
    portion = magnitudes[magnitude]
    center = 32 if img.shape[1]==65 else 16
    position_matrix = np.zeros( img.shape,  np.float32)

    input_depth = img.shape[0]
    # depth is depth at the variant position, not the whole number of rows that are not empty
    depth = min(input_depth,int(str(alt_info[0]).split('-', maxsplit=1)[0][2:]  ) )
    # See clair3/utils.py write_table_dict function for below code
    # prefix_padding_depth should be determined with total number of reads
    padding_depth = input_depth - depth
    prefix_padding_depth = int(padding_depth / 2)
    suffix_padding_depth = padding_depth - int(padding_depth / 2)
    if depth <= 8: # min coverage is 8
        portion = 0
    drop_rows = set(np.random.choice(range(prefix_padding_depth, input_depth-suffix_padding_depth), int(depth*portion), replace=False))
    
    shift = 0
    

    cp = len(drop_rows)//2  #(input_depth - depth + len(drop_rows) )//2
    prefix = -1
    suffix = -1
    assert( prefix_padding_depth+depth <= input_depth)
    af_dict_drop = defaultdict(lambda : 0)
    
    if v_shift:
        # shift vertical, 
        shift_size = (input_depth + len(drop_rows) - depth) // 4

        if shift_size != 0:
            shift = np.random.randint(-shift_size, shift_size)
        else:
            shift = 0
        cp += shift
        shift += len(drop_rows)//2

    for rr in range(input_depth):
        # horizontal shift or row dropped, keep the allele frequency
        if (cp < 0  or cp >= input_depth) or rr in drop_rows:
            af_val = img[rr,center,5]
            if af_val !=0:
                af_dict_drop[ af_val ] += 1
            if cp < 0 :
                cp += 1
        else:
            position_matrix[cp,:,:] = (img[ rr,:,:])
            if (abs(position_matrix[cp,0,3]) >= 0.01 and  abs(position_matrix[cp,-1,3]) >= 0.01):
                suffix = cp
                if prefix == -1:
                    prefix = cp
            cp += 1

        
        

    # iterate through all af_val and change the value
    # see CreateTensorFullAlignment/generate_tensor function for af_val calculation
    for rr in range(input_depth):
        af_val = position_matrix[rr, center, 5]
        if af_val  != 0:
            revised_af_val = af_val * float(depth) / 100.0
            if af_val in af_dict_drop:
                revised_af_val = revised_af_val - af_dict_drop[af_val] 
            revised_af_val = revised_af_val / float(depth - len(drop_rows) ) * 100.0
            position_matrix[rr, center, 5] = min( int(revised_af_val), 100)
            # print( af_val , revised_af_val, "Drop num:" , af_dict_drop[af_val], "Remain num:" , af_dict[af_val] ,"Ori Depth:" ,  depth, "Total row drop:", len(drop_rows) )
    if shuffle:
        np.random.shuffle(position_matrix[prefix:suffix,:,:])

    return position_matrix, label

def row_drop_ns(img, label, alt_info, magnitude):
    '''
    drops row, adjust the AF value. Also, shuffle rows without changing the shape
    '''
    assert(img.shape[1]==65 or img.shape[1]==33)
    img = np.array(img)
    MAX_DROP = 0.7
    magnitudes = np.linspace(0, MAX_DROP, 11)
    # portion = np.random.rand()*magnitudes[magnitude]
    # min coverage 8
    portion = magnitudes[magnitude]
    
    center = 32 if img.shape[1]==65 else 16
    position_matrix = np.zeros( img.shape,  np.float32)

    input_depth = img.shape[0]
    # depth is depth at the variant position, not the whole number of rows that are not empty
    depth = min(input_depth,int(str(alt_info[0]).split('-', maxsplit=1)[0][2:]  ) )
    # See clair3/utils.py write_table_dict function for below code
    # prefix_padding_depth should be determined with total number of reads
    padding_depth = input_depth - depth
    prefix_padding_depth = int(padding_depth / 2)
    suffix_padding_depth = padding_depth - int(padding_depth / 2)
    if depth <= 8: # min coverage is 8
        portion = 0
    drop_rows = set(np.random.choice(range(prefix_padding_depth, input_depth-suffix_padding_depth), int(depth*portion), replace=False))

    cp = len(drop_rows)//2  #(input_depth - depth + len(drop_rows) )//2
    assert( prefix_padding_depth+depth <= input_depth)
    af_dict_drop = defaultdict(lambda : 0)
    # af_dict = defaultdict(lambda : 0)
    for rr in range(input_depth):
        if rr in drop_rows:
            af_val = img[rr,center,5]
            if af_val !=0:
                af_dict_drop[ af_val ] += 1
        else:
            # af_val = img[rr,center,5]
            # if af_val !=0:
            #     af_dict[af_val] += 1
            position_matrix[cp,:,:] = img[ rr,:,:]
            cp += 1

    # iterate through all af_val and change the value
    # see CreateTensorFullAlignment/generate_tensor function for af_val calculation
    for rr in range(input_depth):
        af_val = position_matrix[rr, center, 5]
        if af_val  != 0:
            revised_af_val = af_val * float(depth) / 100.0
            if af_val in af_dict_drop:
                revised_af_val = revised_af_val - af_dict_drop[af_val] 
            revised_af_val = revised_af_val / float(depth - len(drop_rows) ) * 100.0
            position_matrix[rr, center, 5] = min( int(revised_af_val), 100)
            # print( af_val , revised_af_val, "Drop num:" , af_dict_drop[af_val], "Remain num:" , af_dict[af_val] ,"Ori Depth:" ,  depth, "Total row drop:", len(drop_rows) )


    return position_matrix, label


def shuffle(img, label, alt_info, magnitude=0):
    '''
    shuffle reads
    '''
    assert(img.shape[1]==65 or img.shape[1]==33)
    img = np.array(img)
    # portion = np.random.rand()*magnitudes[magnitude]

    input_depth = img.shape[0]
    suffix = -1
    prefix = -1
    # af_dict = defaultdict(lambda : 0)
    for rr in range(input_depth):
        if (abs(img[rr,0,3]) >= 0.01 and  abs(img[rr,-1,3]) >= 0.01):
                suffix = rr
                if prefix == -1:
                    prefix = rr
        else:
            if abs(img[rr,-1,3]) <= 0.01 and abs(img[rr,0,3]) >= 0.01:
                prefix = -1
    if prefix != -1:
        np.random.shuffle(img[prefix:suffix,:,:])
    return img, label


def row_drop_no_overlap(img, label, alt_info, magnitude=0):
    '''
    drops read that has no overlap to the candidate mutation position.
    '''
    assert(img.shape[1]==65 or img.shape[1]==33)
    img = np.array(img)
    MAX_DROP = 0.7
    # portion = np.random.rand()*magnitudes[magnitude]
    center = 32 if img.shape[1]==65 else 16
    position_matrix = np.zeros( img.shape,  np.float32)

    input_depth = img.shape[0]
    depth = min(input_depth,int(str(alt_info[0]).split('-', maxsplit=1)[0][2:]  ) )

    cp = 0
    num_center_overlap = sum(img[ :,center,3] == 0)
    cp = (input_depth - depth) // 2
    # af_dict = defaultdict(lambda : 0)
    for rr in range(input_depth):
        
        if img[ rr,center,3] == 0:
            pass
        else:
            position_matrix[cp,:,:] = img[ rr,:,:]
            cp += 1

    return position_matrix, label

def random_candprop_drop(img, label, alt_info, magnitude=0):
    '''
    ranmdomly set max in map quality cause it is not necessary for finding mutations
    dropping it randomly would help robustness or adaptation
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 1, 11)
    
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img

    position_matrix[:,:,5] = 0

    return position_matrix, label


def random_mapqual_max(img, label, alt_info, magnitude=0):
    '''
    ranmdomly set max in map quality cause it is not necessary for finding mutations
    dropping it randomly would help robustness or adaptation
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 1, 11)
    
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img

    mask = position_matrix[:,:,3] > 0 
    position_matrix[:,:,3][mask] = 100

    return position_matrix, label

HAP_TYPE = dict(zip((1, 0, 2), (-50, 20, 50))) 
# PM_DICT = { 25: [50,75,100], 50:[25,75,100], 75:[25,50,100], 100:[25,50,75]}
# INDEL_DICT = [-50, -100] # -50 insertion -100 deletion

PM_DICT = { 75: [50,-75,-50], 50:[-50,75,-75], -75:[-50,50,75], -50:[50,-75,75]}
INDEL_DICT = [-25, 25] # -50 insertion -100 deletion

def random_hap_drop(img, label, alt_info, magnitude=0):
    '''
    ranmdomly drop haplotype info cause it is not necessary for finding mutations
    dropping it randomly would help robustness or adaptation
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 1, 11)
    
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img

    mask = position_matrix[:,:,7] != 0 
    position_matrix[:,:,7][mask] = HAP_TYPE[0]

    return position_matrix, label



def random_invert(img, label, alt_info, magnitude):
    '''
    Reverse-complement reference bases randomly, except the variant position
    * make sure alternative bases are different from reference bases
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 1, 11)
    
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img

    num_col = position_matrix.shape[1]
    num_max = num_col // 2
    invert_num = int( magnitudes[magnitude] * num_max / np.max(magnitudes) )
    invert_cand = np.delete(np.arange(num_col), num_col//2) # exclude variant position, centor of col
    invert_list = np.random.choice(invert_cand, invert_num, replace=False)


    lenth = len(invert_list)
    # invert columns selected
    for idx_ in range(lenth):
        col = invert_list[idx_]
        # position_matrix[:,col,0] = (img[:,col,0] + 50) 
        mask = position_matrix[:,col,0] != 0
        # print("before",mask, position_matrix[:,col,0])
        if mask.any():
            position_matrix[:,col,0][mask] = PM_DICT[int(position_matrix[:,col,0][mask][0]) ] [ (idx_+col)%3 ]
            mask = (position_matrix[:,col,0] != 0) & (position_matrix[:,col,0] == position_matrix[:,col,1] )
            if mask.any():
                position_matrix[:,col,1][mask] = PM_DICT[position_matrix[:,col,1][mask][0] ] [ (lenth+idx_+col)%3 ]

    return position_matrix, label

def random_noise(img, label, alt_info, magnitude):
    '''
    add noise to  base qual
    '''
    img = np.array(img)
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img
    shape = list(img.shape)
    num_size = shape[0] * shape[1]
    num_noise_max = num_size //2
    magnitudes = np.linspace(0, num_noise_max, 11)
    x = list( np.random.randint(0,num_size, size= int(magnitudes[magnitude]) ) )

    apply_channel = [4] # Base qual
    shape[2] = len(apply_channel)

    noise =  np.random.normal(loc=5, scale=20, size=shape)
    noise[ noise < 0] = 0

    for idx, i in enumerate( apply_channel ):
        mask = position_matrix[:,:,i] == 0

        for loc in x:
            row = loc % shape[0]
            col = loc // shape[0]
            position_matrix[row,col,i] -= noise[row,col,idx] 
        
        position_matrix[:,:,i][mask] = 0
        mask = position_matrix[:,:,i] < 0
        position_matrix[:,:,i][mask] = 0


    return position_matrix, label    




def point_mutation(img, label, alt_info, magnitude):
    '''
    add point mutation with low base quality to alternative bases
    '''
    img = np.array(img)
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img
    shape = list(img.shape)
    num_size = shape[0] * shape[1]
    magnitudes = np.linspace(0, num_size // 50, 11)
    x = list( np.random.randint(0,num_size, size= int(magnitudes[magnitude]) ) )
    input_depth = img.shape[0]
    depth = min(input_depth,int(str(alt_info[0]).split('-', maxsplit=1)[0][2:]  ) )
    center = 32 if img.shape[1]==65 else 16
    apply_channel = [1] # alternative bases
    shape[2] = 1

    noise =  np.random.normal(loc=20, scale=20, size=len(x))
    noise[ noise < 0] = 0

    for idx, ch in enumerate( apply_channel ):

        for ii, loc in enumerate(x):
            row = loc % shape[0]
            col = loc // shape[0]
            if  abs(position_matrix[row,col,0]) > 26 and position_matrix[row,col,ch] >= 0:
                # random select base by hash, 
                position_matrix[row,col,ch] = PM_DICT[position_matrix[row,col,0]][(row+col+ii)%3]
                # also modify the base qual
                position_matrix[row,col,4] -= noise[ii] 

                if col == center:
                    
                    mask = (position_matrix[:,col,3] != 0)
                    depth = sum(mask)

                    mask = (position_matrix[:,col,1] == position_matrix[row,col,ch])
                    count = sum(mask)

                    position_matrix[:,col,5][mask] = 100 * count/depth
                
        mask = position_matrix[:,:,4] < 0
        position_matrix[:,:,4][mask] = 0

        ## if normalize required
        # p_min = position_matrix[:,:,i].min() 
        # p_max = position_matrix[:,:,i].max()  
        # position_matrix[:,:,i] = 100* (position_matrix[:,:,i] - p_min  )/ ( p_max - p_min )
        # position_matrix[:,:,i][mask] = 0

    return position_matrix, label  


def indel_mutation(img, label, alt_info, magnitude):
    '''
    add indel mutation 
    '''
    img = np.array(img)
    position_matrix = np.zeros( img.shape, np.float32)
    position_matrix = img
    shape = list(img.shape)
    num_size = shape[0] * shape[1]
    magnitudes = np.linspace(0, num_size // 50, 11)
    x = list( np.random.randint(0,num_size, size= int(magnitudes[magnitude]) ) )
    ch = 1 # alternative base channel
    for ii, loc in enumerate(x):
        row = loc % shape[0]
        col = loc // shape[0]
        if  position_matrix[row,col,0] > 0 and position_matrix[row,col,ch] >= 0 and (col+1 == shape[1] or position_matrix[row,col+1,ch] >= 0 ):
            # random select base by hash, 
            position_matrix[row,col,ch] = INDEL_DICT[(row+col+ii)%2]
            if position_matrix[row,col,ch] == INDEL_DICT[0]:
                # insertion
                position_matrix[row,col,6] = position_matrix[row,col + ((row+ii)%2) ,0] 
            else:
                if col+1 != shape[1]:
                    position_matrix[row,col+1,:3] = 0
                    position_matrix[row,col+1,4] = 0
                    if shape[2] ==8:
                        position_matrix[row,col+1,7] = 0

    return position_matrix, label  

def validate(position_matrix):
    # no bases of reference and alternative bases should be same 
    mask = position_matrix[:,:,0] == position_matrix[:,:,1]
    mask &= position_matrix[:,:,0] != 0
    if  mask.any():
        print(position_matrix[:,:,0])
    assert( not mask.any() )
    # no bases should be under 0 in reference
    # if (position_matrix[:,:,0] < 0).any():
    # print(position_matrix[:,:,0])
    # assert( not (position_matrix[:,:,0] < 0).any() )

    for col in range(33-1):
        if ((position_matrix[:,col,2] ==50) & (position_matrix[:,col+1,2] == -50)).any():
            with np.printoptions(threshold=sys.maxsize, precision=3, suppress=True):
                print(position_matrix[:,:,2])
            assert(False)
        if ((position_matrix[:,col,2] ==-50) & (position_matrix[:,col+1,2] == 50)).any():
            assert(False)

    pass

'''****************************Not used, Not valid augmentations *************************************'''
def row_shuffle(img, label, alt_info, magnitude):
    '''
    complete random shuffle
    '''
    # img = np.array(img)
    magnitudes = np.linspace(0, 0.5, 11)
    input_depth = img.shape[0]
    np.random.shuffle(img)
    return img,label

def col_shift(img, label, alt_info, magnitude):
    '''
    shift the view, horizontal shifts
    should consider the label changes for variant length
    '''
    magnitudes = np.linspace(0, 14, 11)

    assert(img.shape[1] == 65)
    input_size = 65
    variant_length = max (0, abs(np.argmax(label[24:24+input_size]) - 32),
                            abs( np.argmax(label[24+input_size:]) - 32) )
    if variant_length >= 16:
        # just set to 4 (heruistic)
        shift1 = (np.random.randint(-14, 4)//2) *2
    else:
        shift1 = (np.random.randint(-14, 14-variant_length)//2) *2


    label = np.split(label, label_shape_cum, 0 )
    if -16 == shift1:
        img = img[:,16-shift1:,:]
        del_sums = np.sum(label[2][:16-shift1])

        label[2] = label[2][ 16-shift1:]
        
        if del_sums != 0:
            label[2][0] = 1

        del_sums = np.sum(label[3][:16-shift1])

        label[3] = label[3][ 16-shift1:]

        if del_sums != 0:
            label[3][0] = 1

    else:          
        img = img[:,16-shift1:-16-shift1,:]
        assert(16- shift1 >= 0)
        assert(16 + shift1 > 0)
        
        ins_sums = np.sum(label[2][-16-shift1:])
        del_sums = np.sum(label[2][:16-shift1])

        label[2] = label[2][ 16-shift1:-16-shift1]
        
        if ins_sums != 0:
            label[2][-1] = 1
        if del_sums != 0:
            label[2][0] = 1

        ins_sums = np.sum(label[3][-16-shift1:])
        del_sums = np.sum(label[3][:16-shift1])

        label[3] = label[3][ 16-shift1:-16-shift1]

        if ins_sums != 0:
            label[3][-1] = 1
        if del_sums != 0:
            label[3][0] = 1
    return img, label[:4]


def horizontalflip(img, label, magnitude):
    '''
    flip! should manage the indel correctly
    should consider that indel are realigned w.r.t left-end of ambiguous positions...
    maybe use this aug only when length of indel are 1 or same?? 

    **label changes** - only available for SNPs, flip changes label for all length indels
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 0.5, 11)
    position_matrix = np.zeros( img.shape, np.float32)
    assert(img.shape[1] == 65)
    input_size = 65
    
    if label[57] == 1 and  label[122] == 1:
        # we should change candidate position incase of deletion or insertion
        print( "hehe")
    else:
        return position_matrix, label


    position_matrix[:,:,:5] = img[:,:,:5]
    position_matrix = np.flip(position_matrix,1)
    position_matrix[:,:,5:] = img[:,:,5:]

    return position_matrix, label

def invert(img, label, magnitude):
    '''
    Reverse complement all bases
    **label changes** - so should not be used in augmentation 
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 0.5, 11)
    position_matrix = np.zeros( img.shape, np.float32)
    
    position_matrix = img
    position_matrix[:,:,0] = 125 - img[:,:,0]
    position_matrix[:,:,6] = 125 - img[:,:,6] 
    mask = position_matrix[:,:,0] > 101
    position_matrix[:,:,0][mask] = 0

    mask = position_matrix[:,:,6] > 101
    position_matrix[:,:,6][mask] = 0

    # should adapt genotype21 label

    assert(False) # not implemented

    return position_matrix, label

# /* Not Implemented */

def cutmix(img, label, magnitude):
    '''
    mix the reference bases from other without touching variants part
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 0.5, 11)
    position_matrix = np.zeros( img.shape, np.float32)

def mask(img, label, magnitude):
    '''
    mask except the variant parts 
    '''
    img = np.array(img)
    magnitudes = np.linspace(0, 0.5, 11)
