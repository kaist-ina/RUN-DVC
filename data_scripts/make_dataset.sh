#!/bin/bash

source env.sh


echo "Running SplitExtendBed"
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${CLAIR3} SplitExtendBed \
    --bed_fn {4} \
    --output_fn ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]}

echo "Running Realignment"
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/realignreads.log -j${THREADS_LOW} \
"${PYPY} ${CLAIR3} RealignReads \
    --bam_fn {4} \
    --ref_fn {5} \
    --read_fn ${PHASE_BAM_PATH}/_{2}_{3}_{1} \
    --chunk_id {6} \
    --chunk_num ${chunk_num} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} ::: ${CHUNK_LIST[@]}

echo "Running Indexing on Realigned BAMs"
### Index the phased bam files using samtools, for long reads, realigned reads have contig info at the end
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/phasedbam_indexing.log -j ${THREADS} "if [ -f ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{4}_${chunk_num} ]; then ${SAMTOOLS} index -@12 ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{4}_${chunk_num}; fi" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} ::: ${CHUNK_LIST[@]}

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
"if [ -f ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{6}_${chunk_num} ]; then 
    ${PYPY} ${CLAIR3} UnifyRepresentation \
    --bam_fn ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{6}_${chunk_num} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --ref_fn {5} \
    --bed_fn {4} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --output_vcf_fn ${VCF_OUTPUT_PATH}/vcf_{2}_{3}_{1}_{6} \
    --min_af ${MIN_AF} \
    --chunk_id {6} \
    --chunk_num ${chunk_num} \
    --platform ${PLATFORM} \
    --ctgName ${CHR_PREFIX}{1}; fi" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} ::: ${CHUNK_LIST[@]} > ${DATASET_FOLDER_PATH}/RU.log


echo "Running Sorting VCF& merging."
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/softvcf.log -j${THREADS} \
"
cat ${VCF_OUTPUT_PATH}/vcf_{1}_{2}* | ${PYPY} ${CLAIR3} SortVcf --output_fn ${UNIFIED_VCF_PATH}/unified_{1}_{2}.vcf
" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 
# cat ${VCF_OUTPUT_PATH}/vcf_* | ${PYPY} ${CLAIR3} SortVcf --output_fn ${OUTPUT_DIR}/unified.vcf
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/bgzip.log -j${THREADS} \
"
bgzip -f ${UNIFIED_VCF_PATH}/unified_{1}_{2}.vcf
" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 
# bgzip -f ${OUTPUT_DIR}/unified.vcf

${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/tabix.log -j${THREADS} \
"
tabix -f -p vcf ${UNIFIED_VCF_PATH}/unified_{1}_{2}.vcf.gz
" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 


echo "Running GetTruth for training data"
# Convert an unified VCF file into a simplified var file
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR3} GetTruth \
    --vcf_fn ${UNIFIED_VCF_PATH}/unified_{2}_{3}.vcf.gz \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} 


echo "Running CreateTensor"
# Create full-alignment tensors for model training, removed  --full_aln_regions ${CANDIDATE_BED_PATH}/{2}_{3}_{1}_{7} 
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/create_tensor_full_alignment.log -j${THREADS_LOW} \
"if [ -f ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{7}_${chunk_num} ]; then 
    ${PYPY} ${CLAIR3} CreateTrainingTensor \
    --bam_fn ${PHASE_BAM_PATH}/_{2}_{3}_{1}.{7}_${chunk_num} \
    --ref_fn {5} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --bin_fn ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_{7} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --bed_fn {6} \
    --phasing_info_in_bam \
    --platform ${PLATFORM} \
    --shuffle \
    --maximum_non_variant_ratio ${MAXIMUM_NON_VARIANT_RATIO} \
    --chunk_id {7} \
    --chunk_num ${chunk_num}; fi" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} ::: ${CHUNK_LIST[@]}

echo "Running Mergebins"
# Merge compressed binaries
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/mergeBin.log -j${THREADS} \
"${PYTHON3} ${CLAIR3} MergeBin \
    ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_* \
    --platform ${PLATFORM}\
    --out_fn ${BINS_FOLDER_PATH}/bin_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}