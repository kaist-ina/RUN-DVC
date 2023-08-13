# # Setup variables

RUNDMC="../run_rundvc.sh"

# Get variables from other env file
source ./env.sh
PILEUP_MODEL_PATH="/data/models/pileup_novafree.pt"
# CHR=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
CHR=( 20 21 22 )
THREADS=90

DEBUG="0"

if [ "${DEBUG}" = "1" ]; 
then 
    ALL_UNPHASED_BAM_FILE_PATH=(
    "/data/HG001.novaseq.pcr-plus.30x.dedup.grch38.bam"
    )

    # Each line represents a sample, a sample can be specified multiple times to allow downsampling
    ALL_SAMPLE=(
    'hg003_novaseq_plus'
    )

    # A downsampling numerator (1000 as denominator) for each sample in ALL_SAMPLE, 1000 means no downsampling, 800 means 80% (800/1000)
    DEPTHS=(
    30
    )

    # Reference genome file for each sample
    ALL_REFERENCE_FILE_PATH=(
    "/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
    )

    # just used for defining region to sample (to avoid NN in reference)
    ALL_BED_FILE_PATH=(
    "/data/data_HG00X/HG002_GRCh37_1_22_v4.2.1_benchmark.bed"
    )
fi

# Chromosome prefix ("chr" if chromosome names have the "chr" prefix)
CHR_PREFIX="chr"

MIN_AF=0.08

THREADS_LOW=$((${THREADS}*3/4))
if [[ ${THREADS_LOW} < 1 ]]; then THREADS_LOW=1; fi
# The number of chucks to be divided into for parallel processing
CHUNK_LIST=`seq 1 ${chunk_num}`

MAXIMUM_NON_VARIANT_RATIO=1

# Temporary working directory
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
TENSOR_CANDIDATE_PATH="${DATASET_FOLDER_PATH}/tensor_can_ul"
TENSOR_CANDIDATE_PATH_PILEUP="${DATASET_FOLDER_PATH}/tensor_can_pileup_ul"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/ul_bins"
BINS_PILEUP_FOLDER_PATH="${DATASET_FOLDER_PATH}/ul_bins_pileup"
SPLIT_BED_PATH="${DATASET_FOLDER_PATH}/split_beds"
VAR_OUTPUT_PATH="${DATASET_FOLDER_PATH}/var"
PILEUP_OUTPUT_PATH="${OUTPUT_DIR}/pileup_output"
UNIFIED_VCF_PATH="${OUTPUT_DIR}/unified_vcf"
UNIFIED_VCF_PATH_P="${OUTPUT_DIR}/unified_vcf_P"
UL_VCF_OUTPUT_PATH="${OUTPUT_DIR}/unlabeled_vcf"

mkdir -p ${UNIFIED_VCF_PATH}
mkdir -p ${UNIFIED_VCF_PATH_P}
mkdir -p ${DATASET_FOLDER_PATH}
mkdir -p ${TENSOR_CANDIDATE_PATH}
mkdir -p ${TENSOR_CANDIDATE_PATH_PILEUP}
mkdir -p ${BINS_FOLDER_PATH}
mkdir -p ${BINS_PILEUP_FOLDER_PATH}
mkdir -p ${SPLIT_BED_PATH}
mkdir -p ${VAR_OUTPUT_PATH}
mkdir -p ${PILEUP_OUTPUT_PATH}
mkdir -p ${UL_VCF_OUTPUT_PATH}
