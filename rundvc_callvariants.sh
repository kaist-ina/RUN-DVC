# HG003 samples
BAM_FILE="/data/HG003.novaseq.pcr-free.30x.dedup.grch38.bam"
DATA_NAME="baseline_novaplus2novafree"
MODEL="/data/rundmc/baseline_hiseqX.pt"
./run_rundmc.sh \
  --rundvc_call_mut \
  --bam_fn=${BAM_FILE} \
  --bed_fn=/data/data_HG00X/HG003_GRCh38_1_22_v4.2.1_benchmark.bed  \
  --ref_fn=/data/human_ref/hg38/Homo_sapiens_assembly38.fasta \
  --threads=94 \
  --chunk_num=50 \
  --platform="ilmn" \
  --fa_model=${MODEL} \
  --no_phasing_for_fa \
  --output=/data/output/calls/rundmc_${DATA_NAME}




BAM_FILE="/data/HG003.novaseq.pcr-free.30x.dedup.grch38.bam"
DATA_NAME="baseline_novaplus2novafree"
# pileup model
MODEL="/data/rundmc/pileup.pt"
./run_rundmc.sh \
  --bam_fn=${BAM_FILE} \
  --bed_fn=/data/data_HG00X/HG003_GRCh38_1_22_v4.2.1_benchmark.bed  \
  --ref_fn=/data/human_ref/hg38/Homo_sapiens_assembly38.fasta \
  --threads=94 \
  --chunk_num=50 \
  --platform="ilmn" \
  --pileup_model=${MODEL} \
  --no_phasing_for_fa \
  --output=/data/output/calls/rundmc_${DATA_NAME}