#!/bin/bash
SCRIPT_NAME=$(basename "$0")
Usage="Usage: ./${SCRIPT_NAME} --bam_fn=BAM --ref_fn=REF --output=OUTPUT_DIR --threads=THREADS --platform=PLATFORM --pileup_model=PILEUP_MODEL [--fa_model=FULLALIGNMENT_MODEL] [--bed_fn=BED] [options]"

set -e
ARGS=`getopt -o b:f:t:m:p:o:r::c::s::h::g \
-l bam_fn:,ref_fn:,threads:,platform:,output:,\
bed_fn::,vcf_fn::,ctg_name::,sample_name::,help::,qual::,samtools::,python::,pypy::,parallel::,whatshap::,chunk_num::,chunk_size::,var_pct_full::,var_pct_phasing::,\
min_mq::,min_coverage::,min_contig_size::,snp_min_af::,indel_min_af::,ref_pct_full::,pileup_only::,fast_mode::,gvcf::,print_ref_calls::,haploid_precise::,haploid_sensitive::,include_all_ctgs::,\
no_phasing_for_fa::,pileup_model::,fa_model::,call_snp_only::,remove_intermediate_dir::,enable_phasing::,enable_long_indel::,use_gpu::,longphase_for_phasing::,longphase:: -n 'run_rundvc.sh' -- "$@"`

if [ $? != 0 ] ; then echo"No input. Terminating...">&2 ; exit 1 ; fi
eval set -- "${ARGS}"

while true; do
   case "$1" in
    -b|--bam_fn ) BAM_FILE_PATH="$2"; shift 2 ;;
    -f|--ref_fn ) REFERENCE_FILE_PATH="$2"; shift 2 ;;
    -t|--threads ) THREADS="$2"; shift 2 ;;
    -p|--platform ) PLATFORM="$2"; shift 2 ;;
    -o|--output ) OUTPUT_FOLDER="$2"; shift 2 ;;
    --bed_fn ) BED_FILE_PATH="$2"; shift 2 ;;
    --vcf_fn ) VCF_FILE_PATH="$2"; shift 2 ;;
    --ctg_name ) CONTIGS="$2"; shift 2 ;;
    --sample_name ) SAMPLE="$2"; shift 2 ;;
    --chunk_num ) CHUNK_NUM="$2"; shift 2 ;;
    --chunk_size ) CHUNK_SIZE="$2"; shift 2 ;;
    --qual ) QUAL="$2"; shift 2 ;;
    --samtools ) SAMTOOLS="$2"; shift 2 ;;
    --python ) PYTHON="$2"; shift 2 ;;
    --pypy ) PYPY="$2"; shift 2 ;;
    --parallel ) PARALLEL="$2"; shift 2 ;;
    --whatshap ) WHATSHAP="$2"; shift 2 ;;
    --longphase ) LONGPHASE="$2"; shift 2 ;;
    --var_pct_full ) PRO="$2"; shift 2 ;;
    --ref_pct_full ) REF_PRO="$2"; shift 2 ;;
    --var_pct_phasing ) PHASING_PCT="$2"; shift 2 ;;
    --pileup_only ) PILEUP_ONLY="$2"; shift 2 ;;
    --fast_mode ) FAST_MODE="$2"; shift 2 ;;
    --call_snp_only ) SNP_ONLY="$2"; shift 2 ;;
    --print_ref_calls ) SHOW_REF="$2"; shift 2 ;;
    --gvcf ) GVCF="$2"; shift 2 ;;
    --snp_min_af ) SNP_AF="$2"; shift 2 ;;
    --indel_min_af ) INDEL_AF="$2"; shift 2 ;;
    --min_mq ) MIN_MQ="$2"; shift 2 ;;
    --min_coverage ) MIN_COV="$2"; shift 2 ;;
    --min_contig_size ) MIN_CONTIG_SIZE="$2"; shift 2 ;;
    --pileup_model ) PILEUP_PREFIX="$2"; shift 2 ;;
    --fa_model ) FA_PREFIX="$2"; shift 2 ;;
    --haploid_precise ) HAP_PRE="$2"; shift 2 ;;
    --haploid_sensitive ) HAP_SEN="$2"; shift 2 ;;
    --include_all_ctgs ) INCLUDE_ALL_CTGS="$2"; shift 2 ;;
    --no_phasing_for_fa ) NO_PHASING="$2"; shift 2 ;;
    --remove_intermediate_dir ) RM_TMP_DIR="$2"; shift 2 ;;
    --enable_phasing ) ENABLE_PHASING="$2"; shift 2 ;;
    --enable_long_indel ) ENABLE_LONG_INDEL="$2"; shift 2 ;;
    --use_gpu ) USE_GPU="$2"; shift 2 ;;
    --longphase_for_phasing ) USE_LONGPHASE="$2"; shift 2 ;;

    -- ) shift; break; ;;
    -h|--help ) print_help_messages; break ;;
    * ) print_help_messages; exit 0 ;;
   esac
done


SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
CLAIR3="${SHELL_FOLDER}/../clair3.py"

if [ ${BED_FILE_PATH} = "EMPTY" ] ; then BED_FILE_PATH= ; fi
RETRIES=4


rm -rf ${OUTPUT_FOLDER}/tmp

LOG_PATH="${OUTPUT_FOLDER}/log"
TMP_FILE_PATH="${OUTPUT_FOLDER}/tmp"
SPLIT_BED_PATH="${TMP_FILE_PATH}/split_beds"
PILEUP_VCF_PATH="${TMP_FILE_PATH}/pileup_output"
GVCF_TMP_PATH="${TMP_FILE_PATH}/gvcf_tmp_output"
PHASE_OUTPUT_PATH="${TMP_FILE_PATH}/phase_output"
FULL_ALIGNMENT_OUTPUT_PATH="${TMP_FILE_PATH}/full_alignment_output"
PHASE_VCF_PATH="${PHASE_OUTPUT_PATH}/phase_vcf"
PHASE_BAM_PATH="${PHASE_OUTPUT_PATH}/phase_bam"
CANDIDATE_BED_PATH="${FULL_ALIGNMENT_OUTPUT_PATH}/candidate_bed"
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1


CHUNK_LIST=`seq 1 ${CHUNK_NUM}`

echo $''
echo "[INFO] Check environment variables"
# --vcf_fn ${VCF_FILE_PATH} \ vcf is from source for counting number of mutations.
${PYTHON} ${CLAIR3} CheckEnvs \
    --bam_fn ${BAM_FILE_PATH} \
    --bed_fn ${BED_FILE_PATH} \
    --output_fn_prefix ${OUTPUT_FOLDER} \
    --ref_fn ${REFERENCE_FILE_PATH} \
    --ctg_name ${CONTIGS} \
    --chunk_num ${CHUNK_NUM} \
    --chunk_size ${CHUNK_SIZE} \
    --include_all_ctgs ${INCLUDE_ALL_CTGS} \
    --threads ${THREADS} \
    --python ${PYTHON} \
    --pypy ${PYPY} \
    --samtools ${SAMTOOLS} \
    --whatshap ${WHATSHAP} \
    --parallel ${PARALLEL} \
    --qual ${QUAL} \
    --sampleName ${SAMPLE} \
    --var_pct_full ${PRO} \
    --ref_pct_full ${REF_PRO} \
    --snp_min_af ${SNP_AF} \
    --indel_min_af ${INDEL_AF} \
    --min_contig_size ${MIN_CONTIG_SIZE}

if [ "$(uname)" = "Darwin" ];
then
    mapfile -t CHR < "${OUTPUT_FOLDER}/tmp/CONTIGS"
else
    readarray -t CHR < "${OUTPUT_FOLDER}/tmp/CONTIGS"
fi

if [ ${#CHR[@]} -eq 0 ]; then echo "[INFO] Exit in environment checking"; exit 0; fi

THREADS_LOW=$((${THREADS}*3/4))
LONGPHASE_THREADS=$((${THREADS}*1/2))
if [[ ${THREADS_LOW} < 1 ]]; then THREADS_LOW=1; fi
if [[ ${LONGPHASE_THREADS} < 1 ]]; then LONGPHASE_THREADS=1; fi
if [ "${PLATFORM}" = "ont" ]; then LP_PLATFORM="ont"; else LP_PLATFORM="pb"; fi

cd ${OUTPUT_FOLDER}
# Pileup calling
#-----------------------------------------------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""

#-----------------------------------------------------------------------------------------------------------------------

if [ "${PLATFORM}" = "ilmn" ];
then
    echo "[INFO] 2/6 No phasing"
    ${PARALLEL} -j${THREADS} ln -sf ${BAM_FILE_PATH} ${PHASE_BAM_PATH}/{1}.bam ::: ${CHR[@]}
    if [ -f ${BAM_FILE_PATH}.bai ]; then ${PARALLEL} --retries ${RETRIES} -j${THREADS} ln -sf ${BAM_FILE_PATH}.bai ${PHASE_BAM_PATH}/{1}.bam.bai ::: ${CHR[@]}; fi
    if [ -f ${BAM_FILE_PATH%.*}.bai ]; then ${PARALLEL} --retries ${RETRIES} -j${THREADS} ln -sf ${BAM_FILE_PATH%.*}.bai ${PHASE_BAM_PATH}/{1}.bam.bai ::: ${CHR[@]}; fi
    
    echo "[INFO] 3/6 SplitExtendBed"
    ${PARALLEL} --joblog ${LOG_PATH}/parallel_3_split_extend_bed.log -j${THREADS} \
    "${PYPY} ${CLAIR3} SplitExtendBed \
        --bed_fn ${BED_FILE_PATH} \
        --output_fn ${SPLIT_BED_PATH}/{1} \
        --ctgName {1}" ::: ${CHR[@]}

    # Pileup calling for scoring
    echo $''
    echo "[INFO] 4/6 Call variants using pileup model"

    time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_4_call_var_bam_full_alignment.log -j ${THREADS_LOW} \
    "${PYTHON} ${CLAIR3} CallVarBam \
        --chkpnt_fn ${PILEUP_PREFIX} \
        --bam_fn ${PHASE_BAM_PATH}/{1}.bam \
        --call_fn ${PILEUP_VCF_PATH}/pileup_{1}_{2}.vcf \
        --sampleName ${SAMPLE} \
        --extend_bed ${SPLIT_BED_PATH}/{1} \
        --bed_fn ${BED_FILE_PATH} \
        --ref_fn ${REFERENCE_FILE_PATH} \
        --ctgName {1} \
        --chunk_id {2} \
        --chunk_num ${CHUNK_NUM} \
        --add_indel_length \
        --gvcf ${GVCF} \
        --minMQ ${MIN_MQ} \
        --minCoverage ${MIN_COV} \
        --enable_long_indel ${ENABLE_LONG_INDEL} \
        --python ${PYTHON} \
        --pypy ${PYPY} \
        --temp_file_dir ${GVCF_TMP_PATH} \
        --samtools ${SAMTOOLS} \
        --platform ${PLATFORM} \
        --rundmc \
        --pileup" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]} |& tee ${LOG_PATH}/6_call_var_bam_full_alignment.log

    # # RNN mutation calling
    # time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_4_call_var_bam_full_alignment.log -j ${THREADS_LOW} \
    # "${PYTHON} ${CLAIR3} CallVarBam \
    #     --chkpnt_fn ${MODEL_PATH} \
    #     --bam_fn ${PHASE_BAM_PATH}/{1}.bam \
    #     --call_fn ${PILEUP_VCF_PATH}/pileup_{1}_{2}.vcf \
    #     --sampleName ${SAMPLE} \
    #     --extend_bed ${SPLIT_BED_PATH}/{1} \
    #     --bed_fn ${BED_FILE_PATH} \
    #     --ref_fn ${REFERENCE_FILE_PATH} \
    #     --ctgName {1} \
    #     --chunk_id {2} \
    #     --chunk_num ${CHUNK_NUM} \
    #     --add_indel_length \
    #     --gvcf ${GVCF} \
    #     --minMQ ${MIN_MQ} \
    #     --minCoverage ${MIN_COV} \
    #     --enable_long_indel ${ENABLE_LONG_INDEL} \
    #     --python ${PYTHON} \
    #     --pypy ${PYPY} \
    #     --temp_file_dir ${GVCF_TMP_PATH} \
    #     --samtools ${SAMTOOLS} \
    #     --platform ${PLATFORM} \
    #     --pileup" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]} |& tee ${LOG_PATH}/6_call_var_bam_full_alignment.log

    # ### CNN based calling for mutation calling
    # time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_4_call_var_bam_full_alignment.log -j ${THREADS_LOW} \
    # "${PYTHON} ${CLAIR3} CallVarBam \
    #     --chkpnt_fn ${MODEL_PATH} \
    #     --bam_fn ${PHASE_BAM_PATH}/{1}.bam \
    #     --call_fn ${PILEUP_VCF_PATH}/pileup_{1}_{2}.vcf \
    #     --sampleName ${SAMPLE} \
    #     --extend_bed ${SPLIT_BED_PATH}/{1} \
    #     --bed_fn ${BED_FILE_PATH} \
    #     --ref_fn ${REFERENCE_FILE_PATH} \
    #     --ctgName {1} \
    #     --chunk_id {2} \
    #     --chunk_num ${CHUNK_NUM} \
    #     --add_indel_length \
    #     --gvcf ${GVCF} \
    #     --minMQ ${MIN_MQ} \
    #     --minCoverage ${MIN_COV} \
    #     --enable_long_indel ${ENABLE_LONG_INDEL} \
    #     --python ${PYTHON} \
    #     --pypy ${PYPY} \
    #     --temp_file_dir ${GVCF_TMP_PATH} \
    #     --samtools ${SAMTOOLS} \
    #     --platform ${PLATFORM}" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]} |& tee ${LOG_PATH}/6_call_var_bam_full_alignment.log

    echo $''
    echo "[INFO] 5/6 Aggregating and Sorting VCF"
    ${PYPY} ${CLAIR3} SortVcf \
        --input_dir ${PILEUP_VCF_PATH} \
        --vcf_fn_prefix "pileup_" \
        --output_fn ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf \
        --sampleName ${SAMPLE} \
        --ref_fn ${REFERENCE_FILE_PATH} \
        --contigs_fn ${TMP_FILE_PATH}/CONTIGS
        
else
    echo "[INFO] 2/6 Call variants using pileup model"
    time ${PARALLEL} --retries ${RETRIES} -C ' ' --joblog ${LOG_PATH}/parallel_1_call_var_bam_pileup.log -j ${THREADS_LOW} \
    "${PYTHON} ${CLAIR3} CallVarBam \
        --chkpnt_fn ${PILEUP_PREFIX} \
        --bam_fn ${BAM_FILE_PATH} \
        --call_fn ${PILEUP_VCF_PATH}/pileup_{1}_{2}.vcf \
        --sampleName ${SAMPLE} \
        --ref_fn ${REFERENCE_FILE_PATH} \
        --extend_bed ${SPLIT_BED_PATH}/{1} \
        --bed_fn ${BED_FILE_PATH} \
        --ctgName {1} \
        --chunk_id {2} \
        --chunk_num ${CHUNK_NUM} \
        --add_indel_length \
        --platform ${PLATFORM} \
        --snp_min_af ${SNP_AF} \
        --indel_min_af ${INDEL_AF} \
        --minMQ ${MIN_MQ} \
        --minCoverage ${MIN_COV} \
        --enable_long_indel ${ENABLE_LONG_INDEL} \
        --python ${PYTHON} \
        --pypy ${PYPY} \
        --samtools ${SAMTOOLS} \
        --temp_file_dir ${GVCF_TMP_PATH} \
        --pileup" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]} |& tee ${LOG_PATH}/1_call_var_bam_pileup.log

    # time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_4_call_var_bam_full_alignment.log -j ${THREADS_LOW} \
    # "${PYTHON} ${CLAIR3} CallVarBam \
    #     --chkpnt_fn ${MODEL_PATH} \
    #     --bam_fn ${BAM_FILE_PATH} \
    #     --call_fn ${PILEUP_VCF_PATH}/pileup_{1}_{2}.vcf \
    #     --sampleName ${SAMPLE} \
    #     --extend_bed ${SPLIT_BED_PATH}/{1} \
    #     --bed_fn ${BED_FILE_PATH} \
    #     --ref_fn ${REFERENCE_FILE_PATH} \
    #     --ctgName {1} \
    #     --chunk_id {2} \
    #     --chunk_num ${CHUNK_NUM} \
    #     --add_indel_length \
    #     --gvcf ${GVCF} \
    #     --minMQ ${MIN_MQ} \
    #     --minCoverage ${MIN_COV} \
    #     --enable_long_indel ${ENABLE_LONG_INDEL} \
    #     --python ${PYTHON} \
    #     --pypy ${PYPY} \
    #     --temp_file_dir ${GVCF_TMP_PATH} \
    #     --samtools ${SAMTOOLS} \
    #     --platform ${PLATFORM}" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]} |& tee ${LOG_PATH}/6_call_var_bam_full_alignment.log


    ${PYPY} ${CLAIR3} SortVcf \
        --input_dir ${PILEUP_VCF_PATH} \
        --vcf_fn_prefix "pileup" \
        --output_fn ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf \
        --sampleName ${SAMPLE} \
        --ref_fn ${REFERENCE_FILE_PATH} \
        --contigs_fn ${TMP_FILE_PATH}/CONTIGS

    # echo $''
    # echo "[INFO] 3/6 Select heterozygous SNP variants for Whatshap phasing and haplotagging"
    # gzip -fdc ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf.gz | ${PYPY} ${CLAIR3} SelectQual --phase --output_fn ${PHASE_VCF_PATH} --var_pct_phasing ${PHASING_PCT}
    # time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_2_select_hetero_snp.log -j${THREADS} \
    # "${PYPY} ${CLAIR3} SelectHetSnp \
    #     --vcf_fn ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf.gz \
    #     --split_folder ${PHASE_VCF_PATH} \
    #     --ctgName {1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} |& tee ${LOG_PATH}/2_select_hetero_snp.log
    
    # echo "[INFO] 4/6 Phase VCF file using Whatshap"
    # time ${PARALLEL}  --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_3_phase.log -j${THREADS} \
    # "${WHATSHAP} phase \
    #     --output ${PHASE_VCF_PATH}/phased_{1}.vcf.gz \
    #     --reference ${REFERENCE_FILE_PATH} \
    #     --chromosome {1} \
    #     --distrust-genotypes \
    #     --ignore-read-groups \
    #     ${PHASE_VCF_PATH}/{1}.vcf \
    #     ${BAM_FILE_PATH}" ::: ${CHR[@]} |& tee ${LOG_PATH}/3_phase.log
    
    # ${PARALLEL} -j${THREADS} tabix -f -p vcf ${PHASE_VCF_PATH}/phased_{}.vcf.gz ::: ${CHR[@]}

    # rm -rf ${PHASE_BAM_PATH}/*
    # echo $''
    # echo "[INFO] 5/6 Haplotag input BAM file using Whatshap"
    # time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_4_haplotag.log -j${THREADS} \
    # "${WHATSHAP} haplotag \
    #     --output ${PHASE_BAM_PATH}/{1}.bam \
    #     --reference ${REFERENCE_FILE_PATH} \
    #     --ignore-read-groups \
    #     --regions {1} \
    #     ${PHASE_VCF_PATH}/phased_{1}.vcf.gz \
    #     ${BAM_FILE_PATH}" ::: ${CHR[@]} |& tee ${LOG_PATH}/4_haplotag.log
    # ${PARALLEL} -j${THREADS} ${SAMTOOLS} index -@12 ${PHASE_BAM_PATH}/{1}.bam ::: ${CHR[@]}

fi

if [ ${PILEUP_ONLY} == True ]; then
    cp ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf.gz ${OUTPUT_FOLDER}/pileup.vcf.gz
    cp ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf.gz.tbi ${OUTPUT_FOLDER}/pileup.vcf.gz.tbi
    if [ "$( gzip -fdc ${OUTPUT_FOLDER}/pileup.vcf.gz | grep -v '#' | wc -l )" -eq 0 ]; then echo "[INFO] Exit in pileup variant calling"; exit 0; fi
    if [ ${RM_TMP_DIR} == True ]; then echo "[INFO] Removing intermediate files in ${OUTPUT_FOLDER}/tmp"; rm -rf ${OUTPUT_FOLDER}/tmp; fi
    echo "[INFO] Only call pileup output with --pileup_only, output file: ${OUTPUT_FOLDER}/pileup.vcf.gz"
    echo "[INFO] Finish calling!"
    exit 0;
fi


# echo $''
# echo "[INFO] 5/4 Select candidates for unlabeled data"
# time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_5_select_candidate.log -j${THREADS} \
# "${PYPY} ${CLAIR3} SelectUnlabeledCandidates \
#     --pileup_vcf_fn ${OUTPUT_FOLDER}/unlabeled_candidates.vcf.gz \
#     --split_folder ${CANDIDATE_BED_PATH} \
#     --ref_fn ${REFERENCE_FILE_PATH} \
#     --platform ${PLATFORM} \
#     --ctgName {1}" ::: ${CHR[@]}  |& tee ${LOG_PATH}/5_select_candidate.log
# if [  "${CONTIGS}" == "EMPTY" ];
# then
    # echo 'Contig not given.'
    echo "[INFO] 6/6 Selecting unlabeled data"
    ${PYPY} ${CLAIR3} SelectUnlabeledCandidates \
        --pileup_vcf_fn ${OUTPUT_FOLDER}/unlabeled_candidates_PreFilter.vcf.gz \
        --split_folder ${CANDIDATE_BED_PATH} \
        --vcf_fn ${VCF_FILE_PATH} \
        --ref_fn ${REFERENCE_FILE_PATH} \
        --platform ${PLATFORM} --eval_vcf |& tee ${LOG_PATH}/6_select_unlabeled_candidate.log
        

    grep -v "^#" ${CANDIDATE_BED_PATH}/ul_dataset | sort -k1,1V -k2n > ${OUTPUT_FOLDER}/ul_dataset.vcf
    mkdir -p eval_vcf/
    cp ${CANDIDATE_BED_PATH}/ul_dataset* ${OUTPUT_FOLDER}/eval_vcf
    cp ${CANDIDATE_BED_PATH}/ul_dataset.bed ${OUTPUT_FOLDER}/
# else
#     echo 'Contig given.'
#     time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_5_select_candidate.log -j${THREADS} \
#     "${PYPY} ${CLAIR3} SelectUnlabeledCandidates \
#         --pileup_vcf_fn ${OUTPUT_FOLDER}/unlabeled_candidates.vcf.gz \
#         --split_folder ${CANDIDATE_BED_PATH} \
#         --ref_fn ${REFERENCE_FILE_PATH} \
#         --platform ${PLATFORM} \
#         --ctgName {1}" ::: ${CHR[@]}  |& tee ${LOG_PATH}/5_select_candidate.log
# fi


# if [ ${ENABLE_PHASING} == True ]
# then
#     echo "[INFO] 7/7 Phasing VCF output in parallel using WhatsHap"
#     time ${PARALLEL} --retries ${RETRIES} --joblog ${LOG_PATH}/parallel_8_phase_vcf_output.log -j${THREADS} \
#     "${WHATSHAP} phase \
#         --output ${TMP_FILE_PATH}/merge_output/phased_merge_{1}.vcf \
#         --reference ${REFERENCE_FILE_PATH} \
#         --ignore-read-groups \
#         ${TMP_FILE_PATH}/merge_output/merge_{1}.vcf \
#         ${BAM_FILE_PATH}" ::: ${CHR[@]} |& tee ${LOG_PATH}/8_phase_vcf_output.log

#     ${PYPY} ${CLAIR3} SortVcf \
#         --input_dir ${TMP_FILE_PATH}/merge_output \
#         --vcf_fn_prefix "phased_merge" \
#         --output_fn ${OUTPUT_FOLDER}/phased_merge_output.vcf \
#         --sampleName ${SAMPLE} \
#         --ref_fn ${REFERENCE_FILE_PATH} \
#         --contigs_fn ${TMP_FILE_PATH}/CONTIGS
# fi

if [ ${RM_TMP_DIR} == True ]; then echo "[INFO] Removing intermediate files in ${OUTPUT_FOLDER}/tmp"; rm -rf ${OUTPUT_FOLDER}/tmp; fi

echo $''
echo "[INFO] Finish generating unlabeled dataset for RUNDMC"

if [ ${ENABLE_PHASING} == True ]; then echo "[INFO] Finish calling, phased output file: ${OUTPUT_FOLDER}/phased_merge_output.vcf.gz"; fi