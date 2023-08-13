import sys
import shlex
from argparse import ArgumentParser
import shared.param_p as param
from shared.utils import subprocess_popen

def split_extend_bed(args):

    """
    Split bed file regions according to the contig name and extend bed region with no_of_positions =
    flankingBaseNum + 1 + flankingBaseNum, which allow samtools mpileup submodule to scan the flanking windows.
    """

    bed_fn = args.bed_fn
    output_fn = args.output_fn
    contig_name = args.ctgName
    region_start = args.ctgStart
    region_end = args.ctgEnd
    expand_region_size = args.expand_region_size
    if bed_fn is None:
        return
    output = []
    unzip_process = subprocess_popen(shlex.split("gzip -fdc %s" % (bed_fn)))
    pre_end, pre_start = -1, -1

    for row in unzip_process.stdout:
        if row[0] == '#':
            continue
        columns = row.strip().split()
        ctg_name = columns[0]
        if contig_name != None and ctg_name != contig_name and "chr"+ctg_name != contig_name:
            continue
        ctg_start, ctg_end = int(columns[1]), int(columns[2])
        if region_start and ctg_end < region_start:
            continue
        if region_end and ctg_start > region_end:
            break
        # youngmok: at first iteration, update pre_start
        if pre_start == -1:
            pre_start = ctg_start - expand_region_size
            pre_end = ctg_end + expand_region_size
            continue
        # youngmok: udpate pre_end to new pre_end based on ctg_end (integrate overlap regions), or add region to output
        if pre_end >= ctg_start - expand_region_size:
            pre_end = ctg_end + expand_region_size
            continue
        else: # youngmok: 
            output.append(' '.join([contig_name, str(pre_start), str(pre_end)]))
            pre_start = ctg_start - expand_region_size
            pre_end = ctg_end + expand_region_size
    with open(output_fn, 'w') as output_file:
        output_file.write('\n'.join(output))

    unzip_process.stdout.close()
    unzip_process.wait()

def split_extend_bed_and_check_region_exists(args):

    """
    Split bed file regions according to the contig name and extend bed region with no_of_positions =
    flankingBaseNum + 1 + flankingBaseNum, which allow samtools mpileup submodule to scan the flanking windows.
    """

    bed_fn = args.bed_fn
    output_fn = args.output_fn
    contig_name = args.ctgName
    region_start = args.ctgStart
    region_end = args.ctgEnd
    expand_region_size = args.expand_region_size
    if bed_fn is None:
        return
    from shared.utils import file_path_from
    chunk_num = args.chunk_num
    chunk_id = args.chunk_id -1
    if chunk_id is not None:

        fai_fn = file_path_from(args.fasta_file_path, suffix=".fai", exit_on_not_found=True, sep='.')
        contig_length = 0
        with open(fai_fn, 'r') as fai_fp:
            for row in fai_fp:
                columns = row.strip().split("\t")
                ctg_name = columns[0]
                if ctg_name != contig_name and "chr"+ctg_name != contig_name:
                    continue
                contig_length = int(columns[1])
        chunk_size = contig_length // chunk_num + 1 if contig_length % chunk_num else contig_length // chunk_num
        chunk_ctg_start = chunk_size * chunk_id  # 0-base to 1-base
        chunk_ctg_end = chunk_ctg_start + chunk_size

    output = []
    unzip_process = subprocess_popen(shlex.split("gzip -fdc %s" % (bed_fn)))
    pre_end, pre_start = -1, -1

    for row in unzip_process.stdout:
        if row[0] == '#':
            continue
        columns = row.strip().split()
        ctg_name = columns[0]
        if contig_name != None and ctg_name != contig_name and "chr"+ctg_name != contig_name:
            continue
        ctg_start, ctg_end = int(columns[1]), int(columns[2])
        if ctg_start > chunk_ctg_start and ctg_end < chunk_ctg_end:
            print(columns, chunk_ctg_start , chunk_ctg_end)

        if region_start and ctg_end < region_start:
            continue
        if region_end and ctg_start > region_end:
            break
        # youngmok: at first iteration, update pre_start
        if pre_start == -1:
            pre_start = ctg_start - expand_region_size
            pre_end = ctg_end + expand_region_size
            continue
        # youngmok: udpate pre_end to new pre_end based on ctg_end (integrate overlap regions), or add region to output
        if pre_end >= ctg_start - expand_region_size:
            pre_end = ctg_end + expand_region_size
            continue
        else: # youngmok: 
            output.append(' '.join([contig_name, str(pre_start), str(pre_end)]))
            pre_start = ctg_start - expand_region_size
            pre_end = ctg_end + expand_region_size



    unzip_process.stdout.close()
    unzip_process.wait()

def main():
    parser = ArgumentParser(description="Extend bed region for pileup calling")

    parser.add_argument('--output_fn', type=str, default=None,
                        help="Path to directory that stores small bins, default: %(default)s)"
                        )
    parser.add_argument('--bed_fn', type=str, default=None,
                        help="Path of the output folder, default: %(default)s")

    parser.add_argument('--expand_region_size', type=int, default=param.no_of_positions,
                        help="Expand region size for each bed region, default: %(default)s")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed")
    
    parser.add_argument('--fasta_file_path', type=str, default=None,
                        help="The name of reference, default: %(default)s")

    ## The number of chucks to be divided into for parallel processing
    parser.add_argument('--chunk_num', type=int, default=None,
                        )

    ## The chuck ID to work on
    parser.add_argument('--chunk_id', type=int, default=None,
                        )
                        
    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)
    if args.fasta_file_path is not None:
        split_extend_bed_and_check_region_exists(args)
    else:
        split_extend_bed(args)


if __name__ == "__main__":
    main()