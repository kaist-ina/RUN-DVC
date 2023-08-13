# Setup variables
CLAIR3="../clair3.py"                                     # clair3.py
PYPY=`which pypy3`                                 # e.g. pypy3
WHATSHAP=`which whatshap`                         # e.g. whatshap
PARALLEL=`which parallel`                         # e.g. parallel
SAMTOOLS=`which samtools`                         # e.g. samtools
TABIX=`which tabix`    
PYTHON3=`which python`

# Input parameters
PLATFORM="ilmn"                       # e.g. {ont, hifi, ilmn}
OUTPUT_DIR="/ssd4/data_novaseq_pcrdata"					       # e.g. output

mkdir -p ${OUTPUT_DIR}


ALL_UNPHASED_BAM_FILE_PATH=(
"/data/HG001.novaseq.pcr-free.30x.dedup.grch38.bam"
"/data/HG001.novaseq.pcr-plus.30x.dedup.grch38.bam"
"/data/HG002.novaseq.pcr-free.30x.dedup.grch38.bam"
"/data/HG002.novaseq.pcr-plus.30x.dedup.grch38.bam"
"/data/HG003.novaseq.pcr-free.30x.dedup.grch38.bam"
"/data/HG003.novaseq.pcr-plus.30x.dedup.grch38.bam"
)

# Each line represents a sample, a sample can be specified multiple times to allow downsampling
ALL_SAMPLE=(
'hg001_novaseq_pcrfree'
'hg001_novaseq_pcrplus'
'hg002_novaseq_pcrfree'
'hg002_novaseq_pcrplus'
'hg003_novaseq_pcrfree'
'hg003_novaseq_pcrplus'
)

# A downsampling numerator (1000 as denominator) for each sample in ALL_SAMPLE, 1000 means no downsampling, 800 means 80% (800/1000)
DEPTHS=(
30
30
30
30
30
30
)

# Reference genome file for each sample
ALL_REFERENCE_FILE_PATH=(
"/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
"/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
"/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
"/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
"/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
"/data/human_ref/hg38/Homo_sapiens_assembly38.fasta"
)

# High-confident BED region file for each sample
ALL_BED_FILE_PATH=(
"/data/data_HG00X/HG001_GRCh38_1_22_v4.2.1_benchmark.bed"
"/data/data_HG00X/HG001_GRCh38_1_22_v4.2.1_benchmark.bed"
"/data/data_HG00X/HG002_GRCh38_1_22_v4.2.1_benchmark.bed"
"/data/data_HG00X/HG002_GRCh38_1_22_v4.2.1_benchmark.bed"
"/data/data_HG00X/HG003_GRCh38_1_22_v4.2.1_benchmark.bed"
"/data/data_HG00X/HG003_GRCh38_1_22_v4.2.1_benchmark.bed"
)

# GIAB truth VCF files (without representation unification) for each sample
TRUTH_VCF_FILE_PATH=(
"/data/data_HG00X/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
"/data/data_HG00X/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
"/data/data_HG00X/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
"/data/data_HG00X/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
"/data/data_HG00X/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
"/data/data_HG00X/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
)


# Chromosome prefix ("chr" if chromosome names have the "chr" prefix)
CHR_PREFIX="chr"

MIN_AF=0.08


# array of chromosomes (do not include "chr"-prefix)
CHR=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
# CHR=( 21 )
# Number of threads to be used
THREADS=96
THREADS_LOW=$((${THREADS}*3/4))
if [[ ${THREADS_LOW} < 1 ]]; then THREADS_LOW=1; fi
# The number of chucks to be divided into for parallel processing
chunk_num=96
CHUNK_LIST=`seq 1 ${chunk_num}`

MAXIMUM_NON_VARIANT_RATIO=1

# Temporary working directory
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
TENSOR_CANDIDATE_PATH="${DATASET_FOLDER_PATH}/tensor_can"
TENSOR_CANDIDATE_PATH_PILEUP="${DATASET_FOLDER_PATH}/tensor_can_pileup"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins"
BINS_PILEUP_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins_pileup"
CANDIDATE_DETAILS_PATH="${DATASET_FOLDER_PATH}/candidate_details"
CANDIDATE_BED_PATH="${DATASET_FOLDER_PATH}/candidate_bed"
SPLIT_BED_PATH="${DATASET_FOLDER_PATH}/split_beds"
VAR_OUTPUT_PATH="${DATASET_FOLDER_PATH}/var"
PILEUP_OUTPUT_PATH="${OUTPUT_DIR}/pileup_output"
UNPHASED_TRUTH_VCF_PATH="${OUTPUT_DIR}/unphased_truth_vcf"
PHASE_VAR_PATH="${OUTPUT_DIR}/phased_var"
PHASE_VCF_PATH="${OUTPUT_DIR}/phased_vcf"
PHASE_BAM_PATH="${OUTPUT_DIR}/phased_bam"
PHASE_BAM_PATH_P="${OUTPUT_DIR}/phased_bam_P"
UNIFIED_VCF_PATH="${OUTPUT_DIR}/unified_vcf"
UNIFIED_VCF_PATH_P="${OUTPUT_DIR}/unified_vcf_P"
VCF_OUTPUT_PATH="${OUTPUT_DIR}/vcf_output"
VCF_OUTPUT_PATH_PILEUP="${OUTPUT_DIR}/vcf_output_P"

mkdir -p ${VCF_OUTPUT_PATH}
mkdir -p ${VCF_OUTPUT_PATH_PILEUP}
mkdir -p ${UNIFIED_VCF_PATH}
mkdir -p ${UNIFIED_VCF_PATH_P}
mkdir -p ${DATASET_FOLDER_PATH}
mkdir -p ${TENSOR_CANDIDATE_PATH}
mkdir -p ${TENSOR_CANDIDATE_PATH_PILEUP}
mkdir -p ${BINS_FOLDER_PATH}
mkdir -p ${BINS_PILEUP_FOLDER_PATH}
mkdir -p ${CANDIDATE_DETAILS_PATH}
mkdir -p ${SPLIT_BED_PATH}
mkdir -p ${VAR_OUTPUT_PATH}
mkdir -p ${CANDIDATE_BED_PATH}
mkdir -p ${PILEUP_OUTPUT_PATH}
mkdir -p ${UNPHASED_TRUTH_VCF_PATH}
mkdir -p ${PHASE_VAR_PATH}
mkdir -p ${PHASE_VCF_PATH}
mkdir -p ${PHASE_BAM_PATH}
mkdir -p ${PHASE_BAM_PATH_P}