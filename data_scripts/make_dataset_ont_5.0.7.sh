#!/bin/bash

source env_ont_5.0.7.sh

# Remove the phasing information if the VCF input is already phased
${PARALLEL} -j${THREADS} "${WHATSHAP} unphase {3} > ${UNPHASED_TRUTH_VCF_PATH}/unphased_truth_{1}_{2}.vcf.gz" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${TRUTH_VCF_FILE_PATH[@]}

echo "Running Phase"
# WhatsHap phasing
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/phase.log -j${THREADS} \
"${WHATSHAP} phase \
    --output ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz \
    --reference {5} \
    --chromosome ${CHR_PREFIX}{1} \
    --ignore-read-groups \
    --distrust-genotypes \
    ${UNPHASED_TRUTH_VCF_PATH}/unphased_truth_{2}_{3}.vcf.gz \
    {4}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} 

# Index the phased VCF files using tabix, which is neccesary for read haplotagging
${PARALLEL} -j ${THREADS} ${TABIX} -p vcf ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}

echo "Running Haplotag"
# WhatsHap haplotaging
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/haplotag.log -j${THREADS} \
"${WHATSHAP} haplotag \
    --output ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam \
    --reference {5} \
    --regions ${CHR_PREFIX}{1} \
    --ignore-read-groups \
    ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz \
    {4}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} 

# Index the phased bam files using samtools
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/index.log -j ${THREADS} ${SAMTOOLS} index -@12 ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}



# Call variants and select candidate variants 
# Only select the candidates in the high-confident BED regions for model training (with --bed_fn)
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/RUNDMC.log -j1 \
"${RUNDMC} \
  --bam_fn={4} \
  --bed_fn={3}  \
  --ref_fn={2} \
  --threads=${THREADS} \
  --chunk_num=${chunk_num} \
  --platform=${PLATFORM} \
  --pileup_model=${PILEUP_MODEL_PATH} \
  --pileup_only \
  --output=${PILEUP_OUTPUT_PATH}/{1}_{5}/"  ::: ${ALL_SAMPLE[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${DEPTHS[@]} 



${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/select_pileup_candidates.log -j${THREADS} \
"${PYPY} ${CLAIR3} SelectHetSnp \
--alt_fn ${PILEUP_OUTPUT_PATH}/{2}_{3}/pileup.vcf.gz \
--split_folder ${CANDIDATE_BED_PATH} \
--sampleName {2} \
--depth {3} \
--ref_pct_full 0.15 \
--var_pct_full 1.0 \
--chunk_num ${chunk_num} \
--phasing_info_in_bam \
--phase \
--ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}


echo "Running SplitExtendBed"
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${CLAIR3} SplitExtendBed \
    --bed_fn {4} \
    --output_fn ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]}

echo "Running GetTruth for repr unif."
# Convert an unified VCF file into a simplified var file
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR3} GetTruth \
    --vcf_fn ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${PHASE_VAR_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}  

# Unification 
echo "Running Unification."
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/unify_repre.log -j${THREADS} \
"${PYPY} ${CLAIR3} UnifyRepresentation \
    --bam_fn ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam \
    --var_fn ${PHASE_VAR_PATH}/var_{2}_{3}_{1} \
    --ref_fn {5} \
    --bed_fn {4} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --output_vcf_fn ${VCF_OUTPUT_PATH}/vcf_{2}_{3}_{1}_{6} \
    --min_af ${MIN_AF} \
    --chunk_id {6} \
    --chunk_num ${chunk_num} \
    --platform ${PLATFORM} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} ::: ${CHUNK_LIST[@]} > ${DATASET_FOLDER_PATH}/RU.log

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
# :::+ ${UNIFIED_VCF_FILE_PATH[@]}



echo "Running CreateTensor"
# --add_no_phasing_data_training \
# Create full-alignment tensors for model training, removed  --full_aln_regions ${CANDIDATE_BED_PATH}/{2}_{3}_{1}_{7} 
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/create_tensor_full_alignment.log -j${THREADS_LOW} \
"${PYPY} ${CLAIR3} CreateTrainingTensor \
    --bam_fn ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam \
    --ref_fn {5} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --bin_fn ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_{7} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --full_aln_regions ${CANDIDATE_BED_PATH}/{2}_{3}_{1}_{7} 
    --bed_fn {6} \
    --phasing_info_in_bam \
    --allow_duplicate_chr_pos \
    --platform ${PLATFORM} \
    --shuffle \
    --maximum_non_variant_ratio ${MAXIMUM_NON_VARIANT_RATIO} \
    --chunk_id {7} \
    --chunk_num ${chunk_num}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} ::: ${CHUNK_LIST[@]}

echo "Running Mergebins"
# Merge compressed binaries
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/mergeBin.log -j${THREADS} \
"${PYTHON3} ${CLAIR3} MergeBin \
    ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_* \
    --platform ${PLATFORM}\
    --out_fn ${BINS_FOLDER_PATH}/bin_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}
