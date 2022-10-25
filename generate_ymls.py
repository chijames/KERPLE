import json
import os

os.makedirs('kerple_configs/exp_configs', exist_ok=True)
os.makedirs('kerple_configs/lengths', exist_ok=True)

pos_embs = ['kerple_power_ap_', 'kerple_log']
datasets = {"github":"data/github/github_text_document", "arxiv":"data/arxiv/arxiv_text_document", "openwebtext2":"data/openwebtext2/openwebtext2_text_document"}
seeds = list(range(1235, 1235+5))
lengths = [512, 1024, 2048, 4096, 8192, 16384]

if __name__ == '__main__':
    for length in lengths:
        with open('kerple_configs/lengths/length_{}.yml'.format(length), 'w') as outfile:
            json.dump({'seq-length-val':length}, outfile, indent=4)
    
    for seed in seeds:
        for dataset_name, dataset_path in datasets.items():
            for pos_emb in pos_embs:
                seed = str(seed)
                with open('kerple_configs/exp_configs/'+'_'.join([pos_emb, dataset_name, seed])+'.yml', 'w') as outfile:
                    json.dump({
                        'save': '_'.join([pos_emb, dataset_name, seed]) + '_checkpoints',
                        'load': '_'.join([pos_emb, dataset_name, seed]) + '_checkpoints',
                        'log-dir': '_'.join([pos_emb, dataset_name, seed]) + '_logs',
                        'seed': int(seed),
                        'data-path': dataset_path,
                        'pos-emb': pos_emb
                    }, outfile, indent=4)
                    
