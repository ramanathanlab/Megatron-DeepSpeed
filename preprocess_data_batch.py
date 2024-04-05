# Preprocess batch of files to bin and idx format

def main():

    import subprocess
    import glob
    import os
    from tqdm import tqdm

    nfiles = glob.glob('./protein_gym/indels/DMS_ProteinGym_indels_multi_prop_fit_meg-ds/*.json')
    print(f'num files: {len(nfiles)}')

    for i in tqdm(range(len(nfiles))):
        sname = nfiles[i].split('/')[-1].split('.')[0]
        print(f'Input json filename: {sname}')
        cmd = f'python preprocess_data.py --input ./protein_gym/indels/DMS_ProteinGym_indels_multi_prop_fit_meg-ds/{sname}.json --output-prefix ./protein_gym/indels/DMS_ProteinGym_indels_multi_prop_fit_meg-ds_bin-idx/{sname} --tokenizer-type Llama2Tokenizer --tokenizer-model /lus/eagle/projects/datasets/dolma/utils/tokenizer.model --workers 16'
        returned_value = os.system(cmd)

if __name__ == '__main__':
    main()


# python preprocess_data.py --input ./protein_gym/indels/DMS_ProteinGym_indels_multi_prop_fit_meg-ds/HIS7_YEAST_Pokusaeva_2019_indels_multi_prop_fit_pref.json --output-prefix ./protein_gym/indels/DMS_ProteinGym_indels_multi_prop_fit_meg-ds_bin-idx/HIS7_YEAST_Pokusaeva_2019_indels_multi_prop_fit_pref --tokenizer-type Llama2Tokenizer --tokenizer-model /lus/eagle/projects/datasets/dolma/utils/tokenizer.model --workers 16


''