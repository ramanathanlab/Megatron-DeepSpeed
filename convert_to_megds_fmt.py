# Convert ProteinGym text-sequences files to megatron-deepspeed format

import glob
import os
from tqdm import tqdm

def return_text_seq(jsonob):
    
    fit_text_seq = []
    unfit_text_seq = []
    fit_seq = []
    unfit_seq = []
    for i in tqdm(range(len(jsonob))):
        if jsonob['context'][i]['fitness'] == 'fit':
            fit_seq.append(jsonob['context'][i]['sequence'])
            fit_text_seq.append(jsonob['text'][i])
        elif jsonob['context'][i]['fitness'] == 'unfit':
            unfit_seq.append(jsonob['context'][i]['sequence'])
            unfit_text_seq.append(jsonob['text'][i])
    
    return fit_text_seq, unfit_text_seq, fit_seq, unfit_seq


def write_to_megds_fmt(outpath, fname, in_seqs, tag=None):
    
    # Convert to json for meg-ds
    exm_message_ch = in_seqs[0]
    message_text = ""
    message_text += exm_message_ch
    import json
    d0 = {"id": f"{0}", "text": message_text}
    st = json.dumps(d0)
    # st = f'{d0}'
    st
    # print(f'st[0]: {st}')

    for i in tqdm(range(1,len(in_seqs))):

        exm_message_ch = in_seqs[i]
        message_text = ""
        message_text += exm_message_ch
        di = {"id": f"{i}", "text": message_text}
        st = st + '\n' + json.dumps(di)

    fname += f'_{tag}.json'
    with open(os.path.join(outpath,fname), 'w') as f:
        f.write(st)
        
def convert_to_megds_fmt(inpath, outpath):
    
    import pandas as pd
    jsonobjf = pd.read_json(path_or_buf=inpath, lines=True)
    
    f_text_seq, uf_text_seq, f_seq, uf_seq = return_text_seq(jsonobjf)
    
    fname = inpath.split('/')[-1].split('.jsonl')[0]
    
    write_to_megds_fmt(outpath, fname, f_text_seq, tag='pref')
    write_to_megds_fmt(outpath, fname, uf_text_seq, tag='unpref')


def main():

    outpath = '/lus/eagle/projects/RL-fold/gdharuman/Megatron-DeepSpeed/protein_gym/substitutions/DMS_ProteinGym_substitutions_multi_prop_fit_meg-ds'
    nfiles = glob.glob('/lus/eagle/projects/RL-fold/gdharuman/Megatron-DeepSpeed/protein_gym/substitutions/DMS_ProteinGym_substitutions_multi_prop_fit/*.jsonl')
    print(f'Number of Substition files: {nfiles}')

    for nf in tqdm(nfiles):
        convert_to_megds_fmt(nf, outpath)

if __name__ == '__main__':
    main()