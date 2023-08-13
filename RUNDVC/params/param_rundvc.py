from itertools import accumulate

# Full alignment input feature list
channel = (
'reference_base', 'alternative_base', 'mapping_quality', 'base_quality', 'strand_info', 'variant_type', 'insert_base',
'phasing_info')  # phasing info if add_phasing
channel_size = len(channel)

matrix_depth_dict = {'ont': 89, 'hifi': 55, 'ilmn': 55}

input_flankingBaseNum = 16
input_no_of_positions = 2 * input_flankingBaseNum + 1
input_shape = [matrix_depth_dict['hifi'], input_no_of_positions, channel_size]
ont_input_shape = [matrix_depth_dict['ont'], input_no_of_positions, channel_size]
label_shape = [21, 3, input_no_of_positions, input_no_of_positions]
label_size = sum(label_shape)
label_shape_cum = list(accumulate(label_shape))
NORMALIZE_NUM = 100.0


# Training hyperparameters
chunk_size = 200
trainBatchSize = 1000
trainBatchSize_tl = 1000
predictBatchSize = 200
trainingDatasetPercentage = 0.95
maxEpoch = 40
UNLABELED_SCALER=1000
OPERATION_SEED = None
RANDOM_SEED = 1
evalInterval = 3
# sgd_nesterov , sgd , rmsprop , rangerlars
opt_name="radam" 
momentum=0.9

lr_min=1e-5
lr_gamma=0.97   #0.1    decrease lr_rate by factor of lr_gamma
lr_step_size=5   #30 decrease every step size epoch
lr_scheduler=""
patience=10

clip_grad_norm=0.5
mu = 0.05
initialLearningRate = 1e-4 # 3e-5
l2RegularizationLambda = 1e-5

swa_step_size=32
swa_start_epoch=0
swa_ema_decay = 0.99

randaug_num=2
randaug_intensity=15

UNIFY_MASK = True
FIXED_mu = True
RELATIVE_THRESHOLD = False
USE_DIST_ALIGN = False


u_ratio = 1  # labeled unlabeled ratio
use_scheduler = False
tau=0.9