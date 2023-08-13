#!/bin/bash

source env_ul.sh

echo "Running RUNDMC"
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/RUNDMC.log -j1 \
"${RUNDMC} \
  --bam_fn={4} \
  --bed_fn={3}  \
  --ref_fn={2} \
  --threads=${THREADS} \
  --chunk_num=${chunk_num} \
  --platform=${PLATFORM} \
  --pileup_model=${PILEUP_MODEL_PATH} \
  --no_phasing_for_fa \
  --output=${UL_VCF_OUTPUT_PATH}/{1}_{5}/"  ::: ${ALL_SAMPLE[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${DEPTHS[@]} 

${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/RUNDMC.log -j1 \
"${PYPY} ${CLAIR3} SelectUnlabeledCandidates \
    --pileup_vcf_fn ${UL_VCF_OUTPUT_PATH}/{1}_{2}/unlabeled_candidates_PreFilter.vcf.gz \
    --split_folder ${UL_VCF_OUTPUT_PATH}/{1}_{2}/ \
    --platform ${PLATFORM} \
    --ul_dataset_size 4200000 \
    --eval_vcf" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}

echo "Running SplitExtendBed"
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${CLAIR3} SplitExtendBed \
    --bed_fn ${UL_VCF_OUTPUT_PATH}/{2}_{3}/ul_dataset.bed \
    --output_fn ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 

# enable for realignment!

# assume realignment is done when making source dataset.
# echo "Running Realignment"
# ${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/realignreads.log -j${THREADS_LOW} \
# "${PYPY} ${CLAIR3} RealignReads \
#     --bam_fn {4} \
#     --ref_fn {5} \
#     --read_fn ${PHASE_BAM_PATH}/_{2}_{3}_{1} \
#     --chunk_id {6} \
#     --chunk_num ${chunk_num} \
#     --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
#     --ctgName ${CHR_PREFIX}{1} \
#     --samtools ${SAMTOOLS}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} ::: ${CHUNK_LIST[@]}

# echo "Running Indexing on Realigned BAMs"
# ### Index the phased bam files using samtools, for long reads, realigned reads have contig info at the end
# ${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/phasedbam_indexing.log -j ${THREADS} "if [ -f ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{4}_${chunk_num} ]; then ${SAMTOOLS} index -@12 ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{4}_${chunk_num}; fi" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} ::: ${CHUNK_LIST[@]}

# generate target unlabeled dataset with labels for evaluation.
echo "Running GetTruth for training data"
# Convert an unified VCF file into a simplified var file
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR3} GetTruth \
    --vcf_fn ${UNIFIED_VCF_PATH}/unified_{2}_{3}.vcf.gz \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 



echo "Running CreateTensor"
rm ${TENSOR_CANDIDATE_PATH}/*
# Create full-alignment tensors for model training, removed  --full_aln_regions ${CANDIDATE_BED_PATH}/{2}_{3}_{1}_{7} 
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/create_tensor_full_alignment.log -j${THREADS_LOW} \
"if [ -f ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{6}_${chunk_num} ]; then 
    ${PYPY} ${CLAIR3} CreateTrainingTensor \
    --bam_fn ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{6}_${chunk_num} \
    --ref_fn {5} \
    --vcf_fn ${UL_VCF_OUTPUT_PATH}/{2}_{3}/ul_dataset.vcf \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --bin_fn ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_{6} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --bed_fn ${UL_VCF_OUTPUT_PATH}/{2}_{3}/ul_dataset.bed \
    --platform ${PLATFORM} \
    --shuffle \
    --chunk_id {6} \
    --chunk_num ${chunk_num}; fi" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]}  ::: ${CHUNK_LIST[@]} 

echo "Running Mergebins"
# Merge compressed binaries
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/mergeBin.log -j${THREADS} \
"${PYTHON3} ${CLAIR3} MergeBin \
    ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_* \
    --platform ${PLATFORM}\
    --out_fn ${BINS_FOLDER_PATH}/bin_ul_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}