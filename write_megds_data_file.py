# Write data file with weights 
# default weights set to 1

import sys
import glob
from tqdm import tqdm

def main():

    inpath = '/lus/eagle/projects/RL-fold/gdharuman/Megatron-DeepSpeed/protein_gym/indels/DMS_ProteinGym_indels_multi_prop_fit_meg-ds_bin-idx/'
    outpath = '/lus/eagle/projects/RL-fold/gdharuman/Megatron-DeepSpeed/ultrafeedback_dataset/'

    # tag = "pref"
    tag = sys.argv[1]
    inpath += '*_'+tag+'_*.bin'
    print(f'inpath: {inpath}')

    nfiles = glob.glob(inpath)
    print(f'Number of files with the tag: {len(nfiles)}')

    lines = []
    # for nf in nfiles:
    for i in tqdm(range(len(nfiles))):
        lines.append('1.0 ' + nfiles[i].split('.bin')[0])
    
    # print(lines)

    with open(outpath+f'data_textseq_proteingym_indels_file_list_{tag[0]}.txt', 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

if __name__ == '__main__':
    main()
