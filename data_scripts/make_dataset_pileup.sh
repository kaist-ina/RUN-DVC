#!/bin/bash

source env.sh


echo "Running SplitExtendBed"
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${CLAIR3} SplitExtendBed \
    --bed_fn {4} \
    --output_fn ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]}


${PARALLEL} -j${THREADS} ln -sf {4} ${PHASE_BAM_PATH_P}/_{2}_{3}_{1} ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]}
${PARALLEL} --retries ${RETRIES} -j${THREADS} ln -sf {4}.bai ${PHASE_BAM_PATH_P}/_{2}_{3}_{1}.bai ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]}

echo "Running GetTruth for repr unif."
# Convert an unified VCF file into a simplified var file
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR3} GetTruth \
    --vcf_fn {4} \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}  :::+ ${TRUTH_VCF_FILE_PATH[@]}

# Unification 
echo "Running Unification."
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/unify_repre.log -j${THREADS} \
"${PYPY} ${CLAIR3} UnifyRepresentation \
    --bam_fn ${PHASE_BAM_PATH_P}/_{2}_{3}_{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --ref_fn {5} \
    --bed_fn {4} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --output_vcf_fn ${VCF_OUTPUT_PATH_PILEUP}/vcf_{2}_{3}_{1}_{6} \
    --min_af ${MIN_AF} \
    --chunk_id {6} \
    --chunk_num ${chunk_num} \
    --platform ${PLATFORM} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} ::: ${CHUNK_LIST[@]} > ${DATASET_FOLDER_PATH}/RU.log

echo "Running Sorting VCF& merging."
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/softvcf.log -j${THREADS} \
"
cat ${VCF_OUTPUT_PATH_PILEUP}/vcf_{1}_{2}* | ${PYPY} ${CLAIR3} SortVcf --output_fn ${UNIFIED_VCF_PATH_P}/unified_{1}_{2}.vcf
" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 
# cat ${VCF_OUTPUT_PATH_PILEUP}/vcf_* | ${PYPY} ${CLAIR3} SortVcf --output_fn ${OUTPUT_DIR}/unified.vcf
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/bgzip.log -j${THREADS} \
"
bgzip -f ${UNIFIED_VCF_PATH_P}/unified_{1}_{2}.vcf
" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 
# bgzip -f ${OUTPUT_DIR}/unified.vcf

${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/tabix.log -j${THREADS} \
"
tabix -f -p vcf ${UNIFIED_VCF_PATH_P}/unified_{1}_{2}.vcf.gz
" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 


echo "Running GetTruth for training data"
# Convert an unified VCF file into a simplified var file
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR3} GetTruth \
    --vcf_fn ${UNIFIED_VCF_PATH_P}/unified_{2}_{3}.vcf.gz \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 

rm ${TENSOR_CANDIDATE_PATH_PILEUP}/*  ${BINS_PILEUP_FOLDER_PATH}/*

echo "Running CreateTensor"
# Create full-alignment tensors for model training, removed  --full_aln_regions ${CANDIDATE_BED_PATH}/{2}_{3}_{1}_{7} 
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/create_tensor_pileup.log -j${THREADS_LOW} \
"${PYPY} ${CLAIR3} CreateTrainingTensor \
    --bam_fn ${PHASE_BAM_PATH_P}/_{2}_{3}_{1} \
    --ref_fn {5} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --bin_fn ${TENSOR_CANDIDATE_PATH_PILEUP}/tensor_{2}_{3}_{1}_{7} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --bed_fn {6} \
    --pileup  \
    --platform ${PLATFORM} \
    --shuffle \
    --maximum_non_variant_ratio ${MAXIMUM_NON_VARIANT_RATIO} \
    --chunk_id {7} \
    --chunk_num ${chunk_num}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} ::: ${CHUNK_LIST[@]}

echo "Running Mergebins"
# Merge compressed binaries
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/mergeBin.log -j${THREADS} \
"${PYTHON3} ${CLAIR3} MergeBin \
    ${TENSOR_CANDIDATE_PATH_PILEUP}/tensor_{2}_{3}_{1}_* \
    --platform ${PLATFORM}\
    --pileup \
    --out_fn ${BINS_PILEUP_FOLDER_PATH}/bin_P_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}
