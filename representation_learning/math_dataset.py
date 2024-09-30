import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MathDataset(Dataset):
    def __init__(self, tokenizer, args):
        # Load both dataset and cluster files
        with open(args.json_file_dataset, 'r') as f:
            self.data = json.load(f)
        with open(args.json_file_cluster_kc, 'r') as f:
            self.cluster_to_kcs = json.load(f)

        # Get kc to cluster mapping from cluster to kc
        self.kc_to_cluster = self.__get_kc_to_cluster__(self.cluster_to_kcs)

        # Save other inputs
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.max_length_kc = args.max_length_kc

    def __get_kc_to_cluster__(self, cluster_to_kcs):
        # Create kc_to_cluster mapping from cluster_to_kcs
        kc_to_cluster = {}
        for cluster_id, kcs in cluster_to_kcs.items():
            for kc in kcs:
                kc_to_cluster[kc] = cluster_id
        return kc_to_cluster

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        problem = self.data[str(idx)]
        # Include special tokens to differentiate segments
        question = "[Q] " + problem['question']
        steps = ["[S] " + step for step in problem['step_by_step_solution_list']]
        concepts = ["[KC] " + kc for kc in problem['knowledge_concepts_list']]
        mappings = problem['mapping_step_kc_gpt-4o'].split(', ')

        # Tokenize question, steps, and concepts separately
        question_enc = self.tokenizer(question, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        step_encs = [self.tokenizer(step, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt') for step in steps]
        concept_encs = [self.tokenizer(kc, max_length=self.max_length_kc, truncation=True, padding='max_length', return_tensors='pt') for kc in concepts]

        # Extract mappings into sorted tensor
        kc_step_pairs = []
        for map_item in mappings:
            step_idx, kc_idx = map(int, map_item.split('-'))
            kc_step_pairs.append((kc_idx - 1, step_idx - 1))  # Reverse to have KC first

        # Sort by kc_idx, then by step_idx
        kc_step_pairs.sort()

        # Convert list of tuples to tensor
        kc_step_pairs_tensor = torch.tensor(kc_step_pairs, dtype=torch.long)

        # Get KC's cluster id
        kc_cluster_ids = [int(self.kc_to_cluster[kc]) for kc in problem['knowledge_concepts_list']]
        kc_cluster_ids_tensors = torch.tensor(kc_cluster_ids, dtype=torch.long)

        return {
            'question_ids': question_enc['input_ids'].squeeze(0),
            'question_mask': question_enc['attention_mask'].squeeze(0),
            'step_ids': torch.cat([enc['input_ids'] for enc in step_encs], dim=0),
            'step_mask': torch.cat([enc['attention_mask'] for enc in step_encs], dim=0),
            'kc_ids': torch.cat([enc['input_ids'] for enc in concept_encs], dim=0),
            'kc_mask': torch.cat([enc['attention_mask'] for enc in concept_encs], dim=0),
            'kc_step_pairs': kc_step_pairs_tensor,
            'kc_cluster_ids': kc_cluster_ids_tensors
        }
