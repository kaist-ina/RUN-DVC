import shlex
import math
import sys
import logging
import os

from argparse import ArgumentParser, SUPPRESS
from collections import defaultdict

from shared.intervaltree.intervaltree import IntervalTree
import shared.param_f as param
from shared.utils import subprocess_popen, IUPAC_base_to_num_dict as BASE2NUM, region_from, reference_sequence_from, str2bool, log_warning

import subprocess

from shared.interval_tree import bed_tree_from, is_region_in

logging.basicConfig(format='%(message)s', level=logging.INFO)


def gaussian_distribution(x, mu, sig=16):
    return math.exp(-math.pow(x - mu, 2.) / (2 * math.pow(sig, 2.)))


def discrete_gaussian_pro(entropy_windnow):
    gaussian_pro = [gaussian_distribution(index, entropy_windnow / 2, 1) for index in range(entropy_windnow)]
    return gaussian_pro


def calculate_sequence_entropy(sequence, entropy_window=None, kmer=5):
    """
    We use a kmer-based sequence entropy calculation to measure the complexity of a region.
    sequence: a chunked sequence around a candidate position, default no_of_positions = flankingBaseNum + 1 + flankingBaseNum
    entropy_window: a maximum entropy window for scanning, if the sequence is larger than the entropy window, a slide
    window would be adopted for measurement.
    kmer: default kmer size for sequence entropy calculation.
    """

    count_repeat_kmer_counts = [0] * (entropy_window + 2)
    count_repeat_kmer_counts[0] = entropy_window

    entropy = [0.0] * (entropy_window + 2)
    for i in range(1, entropy_window + 2):
        e = 1.0 / entropy_window * i
        entropy[i] = e * math.log(e)
    entropy_mul = -1 / math.log(entropy_window)
    entropy_kmer_space = 1 << (2 * kmer)

    kmer_hash_counts = [0] * entropy_kmer_space  # value should smaller than len(seq)
    mask = -1 if kmer > 15 else ~((-1) << (2 * kmer))
    kmer_suffix, kmer_prefix = 0, 0

    i = 0
    i2 = -entropy_window
    entropy_sum = 0.0
    all_entropy_sum = [0.0] * len(sequence)
    while (i2 < len(sequence)):

        if (i < len(sequence)):
            n = BASE2NUM[sequence[i]]
            kmer_suffix = ((kmer_suffix << 2) | n) & mask

            count_repeat_kmer_counts[kmer_hash_counts[kmer_suffix]] -= 1
            entropy_sum -= entropy[kmer_hash_counts[kmer_suffix]]
            kmer_hash_counts[kmer_suffix] += 1
            count_repeat_kmer_counts[kmer_hash_counts[kmer_suffix]] += 1
            entropy_sum += entropy[kmer_hash_counts[kmer_suffix]]

        if i2 >= 0 and i < len(sequence):
            n2 = BASE2NUM[sequence[i2]]
            kmer_prefix = ((kmer_prefix << 2) | n2) & mask  # add base info
            count_repeat_kmer_counts[kmer_hash_counts[kmer_prefix]] -= 1
            entropy_sum -= entropy[kmer_hash_counts[kmer_prefix]]
            kmer_hash_counts[kmer_prefix] -= 1
            count_repeat_kmer_counts[kmer_hash_counts[kmer_prefix]] += 1
            entropy_sum += entropy[kmer_hash_counts[kmer_prefix]]
            all_entropy_sum[i] = entropy_sum
        i += 1
        i2 += 1
    return entropy_sum * entropy_mul


def sqeuence_entropy_from(samtools_execute_command, fasta_file_path, contig_name, candidate_positions):
    """
    Calculate sequence entropy in a specific candidate windows, variants in low sequence entropy regions (low
    mappability regions, such as homopolymer, tandem repeat, segmental duplications regions) would more likely have
    more complex variants representation, which is beyond pileup calling. Hence, those candidate variants are re-called by
    full alignment calling.
    We use a kmer-based sequence entropy calculation to measure the complexity of a region, we would directly query the
    chunked reference sequence for sequence entropy calculation for each candidate variant.
    """

    ref_regions = []
    reference_start, reference_end = min(list(candidate_positions)) - param.no_of_positions, max(
        list(candidate_positions)) + param.no_of_positions + 1
    reference_start = 1 if reference_start < 1 else reference_start
    ref_regions.append(region_from(ctg_name=contig_name, ctg_start=reference_start, ctg_end=reference_end))
    reference_sequence = reference_sequence_from(
        samtools_execute_command=samtools_execute_command,
        fasta_file_path=fasta_file_path,
        regions=ref_regions
    )
    if reference_sequence is None or len(reference_sequence) == 0:
        sys.exit("[ERROR] Failed to load reference seqeunce from file ({}).".format(fasta_file_path))

    entropy_window = param.no_of_positions
    candidate_positions_entropy_list = []
    for pos in candidate_positions:
        ref_seq = reference_sequence[
                  pos - param.flankingBaseNum - reference_start: pos + param.flankingBaseNum + 1 - reference_start]
        sequence_entropy = calculate_sequence_entropy(sequence=ref_seq, entropy_window=entropy_window)
        candidate_positions_entropy_list.append((pos, sequence_entropy))

    return candidate_positions_entropy_list


def SelectCandidates(args):
    """
    Select low quality and low sequence entropy candidate variants for full aligement. False positive pileup variants
    and true variants missed by pileup calling would mostly have low quality score (reference quality score for missing
    variants), so only use a proportion of low quality variants for full alignment while maintain high quality pileup
    output, as full alignment calling is substantially slower than pileup calling.
    """

    phased_vcf_fn = args.phased_vcf_fn
    pileup_vcf_fn = args.pileup_vcf_fn
    source_vcf_fn = args.vcf_fn
    source_bin_fn = args.bin_fn
    var_pct_full = args.var_pct_full
    ref_pct_full = args.ref_pct_full
    seq_entropy_pro = args.seq_entropy_pro
    contig_name = args.ctgName.split(",") if args.ctgName else None
    phasing_window_size = param.phasing_window_size
    platform = args.platform
    split_bed_size = args.split_bed_size
    split_folder = args.split_folder
    extend_bp = param.extend_bp
    call_low_seq_entropy = args.call_low_seq_entropy
    phasing_info_in_bam = args.phasing_info_in_bam
    eval_vcf = args.eval_vcf
    random_select = args.random
    need_phasing_list = []
    need_phasing_set = set()
    ref_call_pos_list = []
    flankingBaseNum = param.flankingBaseNum
    qual_fn = args.qual_fn if args.qual_fn is not None else 'qual'
    fasta_file_path = args.ref_fn
    samtools_execute_command = args.samtools

    low_sequence_entropy_list = []

    

    # if source_bin_fn:
    #     bin_list = os.listdir(bin_fn)
    #     bin_list = [f for f in bin_list ]
    #     for bin_idx, bin_file in enumerate(bin_list):
    #         table_dataset = tables.open_file(os.path.join(bin_fn, bin_file), 'r')
            
    # vcf format
    unzip_process = subprocess_popen(shlex.split("gzip -fdc %s" % (pileup_vcf_fn)))
    header = []

    contig_dict = defaultdict(defaultdict)

    total_num = 0
    chr_dict = defaultdict()
    for row in unzip_process.stdout:
        if row[0] == '#':
            header.append(row)
            continue
        save_row = row
        columns = row.rstrip().split('\t')
        ctg_name = columns[0]
        if contig_name and not ctg_name in contig_name:
            continue
        total_num += 1
        pos = int(columns[1])
        ref_base = columns[3]
        alt_base = columns[4]
        qual = float(columns[5])

        contig_dict[ctg_name][int(pos)] = save_row

        # reference calling
        if columns[6] == "RefCall":
            ref_call_pos_list.append((pos, qual, ctg_name))
        else:
            need_phasing_list.append((pos, qual, ctg_name))
    source_total_num = 3700000
    # count source
    if source_vcf_fn:
        # vcf format
        unzip_process = subprocess_popen(shlex.split("gzip -fdc %s" % (source_vcf_fn)))
        source_total_num = 0
        for row in unzip_process.stdout:
            if row[0] == '#':
                continue
            columns = row.rstrip().split('\t')
            ctg_name = columns[0]
            # count number for truth labels only the chromosomes that exist in unlabeled vcf
            if contig_dict and not ctg_name in contig_dict:
                continue
            source_total_num += 1
        args.ul_dataset_size = source_total_num * args.NUM_UL_RATIO
        if source_total_num ==0:
            assert(False)
        print("Obtained variants {} from source VCF file:".format(source_total_num), source_vcf_fn)

    if contig_name:
        print("Total number of ref/var records for {}:".format(contig_name),total_num)
    else:
        print("Total number of ref/var records:",total_num)
    low_qual_ref_list = sorted(ref_call_pos_list, key=lambda x: x[1])
    low_qual_variant_list = sorted(need_phasing_list, key=lambda x: -x[1])
    if random_select:
        low_qual_variant_list.extend(low_qual_ref_list)
        import random
        random.shuffle(low_qual_variant_list )
        low_qual_variant_list = low_qual_variant_list[:args.ul_dataset_size]

    else:
        num_selected = args.ul_dataset_size
        # ul_dataset = low_qual_variant_list
        num_selected -= len(low_qual_variant_list)
        if num_selected >= 0:
            low_qual_variant_list.extend( low_qual_ref_list[:num_selected] )
        else:
            low_qual_variant_list = low_qual_variant_list[:args.ul_dataset_size]
    

    # if call_low_seq_entropy:
    #     candidate_positions = sorted(ref_call_pos_list, key=lambda x: x[1])[
    #                             :int((var_pct_full + seq_entropy_pro) * len(ref_call_pos_list))] + sorted(need_phasing_list,
    #                                                                                                     key=lambda x: x[
    #                                                                                                         1])[:int(
    #         (var_pct_full + seq_entropy_pro) * len(need_phasing_list))]
    #     candidate_positions = set([item[0] for item in candidate_positions])

    #     candidate_positions_entropy_list = sqeuence_entropy_from(samtools_execute_command=samtools_execute_command,
    #                                                                 fasta_file_path=fasta_file_path,
    #                                                                 contig_name=contig_name,
    #                                                                 candidate_positions=candidate_positions)

    #     low_sequence_entropy_list = sorted(candidate_positions_entropy_list, key=lambda x: x[1])[
    #                                 :int(seq_entropy_pro * len(candidate_positions_entropy_list))]

    # calling with phasing_info_in_bam: select low qual ref and low qual vairant for phasing calling
    def compress_index_vcf(input_vcf):
        # use bgzip to compress vcf -> vcf.gz
        # use tabix to index vcf.gz
        proc = subprocess.run('bgzip -f {}'.format(input_vcf), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc = subprocess.run('tabix -f -p vcf {}.gz'.format(input_vcf), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    need_phasing_row_list = set(low_qual_variant_list)
    print('[INFO] Total candidates need to be processed in : ', len(need_phasing_row_list))


    if eval_vcf:
        for contig_name in contig_dict.keys():
            wfs =  open(os.path.join(split_folder, 'ul_dataset'), 'w') 
            wfs5 =  open(os.path.join(split_folder, 'ul_dataset_0.5'), 'w') 
            wfs1 =  open(os.path.join(split_folder, 'ul_dataset_0.1'), 'w') 
            wfs0 =  open(os.path.join(split_folder, 'ul_dataset_0'), 'w') 
            # wfs.write("\t".join(header))
            break
        
        for i,pos in enumerate(need_phasing_row_list):
            wfs.write(str(contig_dict[pos[2]][int(pos[0])]))

            if i < source_total_num * 1.5:
                wfs5.write(str(contig_dict[pos[2]][int(pos[0])]))

            if i < source_total_num * 1.1:
                wfs1.write(str(contig_dict[pos[2]][int(pos[0])]))

            if i < source_total_num * 1:
                wfs0.write(str(contig_dict[pos[2]][int(pos[0])]))
        
        for contig_name in contig_dict.keys():
            wfs.close()
            wfs1.close()
            wfs5.close()
            wfs0.close()
            # compress_index_vcf(os.path.join(split_folder, 'ul_dataset'))
            break

    else:

        SINGLE_FILE = True
        if SINGLE_FILE:
            pass
        else:
            wfs = {}

        for contig_name in contig_dict.keys():
            if SINGLE_FILE:
                wfs =  open(os.path.join(split_folder, 'ul_dataset'), 'w') 
                # wfs.write("\t".join(header))
                break
            else:
                wfs[contig_name] =  open(os.path.join(split_folder, '{}'.format(contig_name)), 'w') 
                # wfs.write("\t".join(header))
        
        for pos in need_phasing_row_list:
            if SINGLE_FILE:
                wfs.write(str(contig_dict[pos[2]][int(pos[0])]))
            else:
                wfs[pos[2]].write(str(contig_dict[pos[2]][int(pos[0])]))
        
        for contig_name in contig_dict.keys():
            if SINGLE_FILE:
                wfs.close()
                # compress_index_vcf(os.path.join(split_folder, 'ul_dataset'))
                break
            else:
                wfs[contig_name].close()
                compress_index_vcf(os.path.join(split_folder, '{}'.format(contig_name)))

    # considering only chr1-22 for sorting to be correct
    chromosomes = sorted(list(contig_dict.keys()))
    
    expand_region_size = param.no_of_positions

    EXPAND = False
    wfs =  open(os.path.join(split_folder, 'ul_dataset.bed'), 'w') 
    for cc in chromosomes:
        pre_end, pre_start = -1, -1
        sorted_pos = sorted(contig_dict[cc].keys())
        for pos in sorted_pos:
            if not EXPAND:
                wfs.write("{}\t{}\t{}\n".format(cc,pos-1,pos+2))
            else:
                # youngmok: at first iteration, update pre_start
                if pre_start == -1:
                    pre_start = pos - expand_region_size
                    pre_end = pos+2 + expand_region_size
                    continue
                # youngmok: udpate pre_end to new pre_end based on ctg_end (integrate overlap regions), or add region to output
                if pre_end >= pos - expand_region_size:
                    pre_end = pos+2 + expand_region_size
                    continue
                else: # youngmok: 
                    wfs.write("{}\t{}\t{}\n".format(cc,pre_start,pre_end))
                    pre_start = pos - expand_region_size
                    pre_end = pos+2 + expand_region_size
    wfs.close()
            




def main():
    parser = ArgumentParser(description="Select candidates for unlabeled data")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--split_folder', type=str, default=None, required=True,
                        help="Path to directory that stores candidate region, required")

    parser.add_argument('--pileup_vcf_fn', type=str, default=None, required=True,
                        help="Input pileup pileup vcf, required")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference fasta file input, required")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Source domain VCF file input for choosing number of data, required")
    
    parser.add_argument('--bin_fn', type=str, default=None,
                        help="Source domain BED file, required")

    parser.add_argument('--ul_dataset_size', type=int, default=80000000,
                        help="Number of dataset to select. default: %(default)s")

    parser.add_argument('--NUM_UL_RATIO', type=int, default=2,
                        help="Number of dataset to select. default: %(default)s")

    parser.add_argument('--var_pct_full', type=float, default=0.3,
                        help="Specify an expected percentage of low quality 0/1 and 1/1 variants called in the pileup mode for full-alignment mode calling, default: %(default)f")

    parser.add_argument('--ref_pct_full', type=float, default=0.3,
                        help="Specify an expected percentage of low quality 0/0 variants called in the pileup mode for full-alignment mode calling, default: %(default)f")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of sequence to be processed")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', samtools version >= 1.10 is required, default: %(default)s")
    
    parser.add_argument('--eval_vcf', action='store_true',
                        help="DEBUG: evaluate unlabeled data selection with input vcf")

    parser.add_argument('--random', action='store_true',
                        help="DEBUG: evaluate random selection vcf")

    # options for advanced users
    parser.add_argument('--call_low_seq_entropy', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable full alignment calling on candidate variants with low sequence entropy")

    parser.add_argument('--seq_entropy_pro', type=float, default=0.05,
                        help="EXPERIMENTAL: Define the percentage of the candidate variants with the lowest sequence entropy for full alignment calling, default: %(default)f")

    parser.add_argument('--split_bed_size', type=int, default=10000,
                        help="EXPERIMENTAL: Define the candidate bed size for each split bed file. default: %(default)s")

    # options for debug purpose
    parser.add_argument('--phasing_info_in_bam', action='store_false',
                        help="DEBUG: Skip phasing and use the phasing info provided in the input BAM (HP tag), default: True")

    # options for internal process control
    ## Default chr prefix for contig name
    parser.add_argument('--chr_prefix', type=str, default='chr',
                        help=SUPPRESS)

    ## Input phased pileup vcf
    parser.add_argument('--phased_vcf_fn', type=str, default=None,
                        help=SUPPRESS)

    ## Output all alternative candidates path
    parser.add_argument('--all_alt_fn', type=str, default=None,
                        help=SUPPRESS)

    ## Input the file that contains the quality cut-off for selecting low-quality pileup calls for phasing and full-alignment calling
    parser.add_argument('--qual_fn', type=str, default=None,
                        help=SUPPRESS)

    args = parser.parse_args()

    SelectCandidates(args)


if __name__ == "__main__":
    main()
