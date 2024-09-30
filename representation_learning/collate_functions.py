import torch
from collections import defaultdict

def custom_collate_fn(batch):
    # Initializing lists to hold batch data
    question_ids, question_masks = [], []
    all_step_ids, all_step_masks, step_lengths = [], [], []
    all_kc_ids, all_kc_masks, kc_lengths = [], [], []
    all_kc_step_pairs = []
    all_kc_quest_pairs = []
    all_kc_cluster_ids = []
    
    # Define the dicts for keeping track of clusters
    cluster_to_steps = defaultdict(list)
    cluster_to_kcs = defaultdict(list)
    cluster_to_questions = defaultdict(set)  # Using a set to avoid duplicates

    num_prev_steps = 0
    num_prev_kcs = 0

    num_problems = len(batch)
    
    # Process each item in the batch
    for i, item in enumerate(batch):
        question_ids.append(item['question_ids'])
        question_masks.append(item['question_mask'])
        
        # Concatenate all step IDs and masks; track lengths
        all_step_ids.append(item['step_ids'])
        all_step_masks.append(item['step_mask'])
        cur_num_steps = item['step_ids'].size(0)
        step_lengths.append(cur_num_steps)
        
        # Concatenate all kc IDs and masks; track lengths
        all_kc_ids.append(item['kc_ids'])
        all_kc_masks.append(item['kc_mask'])
        cur_num_kcs = item['kc_ids'].size(0)
        kc_lengths.append(cur_num_kcs)
        all_kc_cluster_ids.append(item['kc_cluster_ids'])

        #Shift kc-step pairs according to the position of item in the batch
        kc_step_pairs = item['kc_step_pairs'].detach().clone()
        kc_step_pairs[:,0] += num_prev_kcs
        kc_step_pairs[:,1] += num_prev_steps
        #Concatenate all kc-step pairs
        all_kc_step_pairs.append(kc_step_pairs)

        # Validate kc_step_pairs indices
        assert kc_step_pairs[:, 0].max() < sum(kc_lengths), "kc index out of bounds"
        assert kc_step_pairs[:, 1].max() < sum(step_lengths), "step index out of bounds"


        # Create and append kc-quest pairs
        # to shift the question, use i
        # to shift the kc, use num_prev_kcs
        kc_quest_pairs = torch.ones((cur_num_kcs, 2)) * i
        kc_quest_pairs[:,0] = torch.arange(cur_num_kcs) # all kcs point to question i
        kc_quest_pairs[:,0] += num_prev_kcs #kcs shifted correctly
        kc_quest_pairs = kc_quest_pairs.long()
        all_kc_quest_pairs.append(kc_quest_pairs)

        # Validate kc_quest_pairs indices
        assert kc_quest_pairs[:, 0].max() < sum(kc_lengths), "kc index out of bounds"
        assert kc_quest_pairs[:, 1].max() < num_problems, "question index out of bounds"

        # Collect step indices corresponding to each KC index, adjusted by previously processed steps
        for kc_idx, cluster_id in enumerate(item['kc_cluster_ids']):
            # Get all step indices where the KC index is involved
            step_indices = item['kc_step_pairs'][:, 1][item['kc_step_pairs'][:, 0] == kc_idx]
            # Adjust these step indices by the number of steps previously processed and add to the cluster
            adjusted_step_indices = (num_prev_steps + step_indices).tolist()
            cluster_to_steps[cluster_id.item()].extend(adjusted_step_indices)
            cluster_to_kcs[cluster_id.item()].append(num_prev_kcs + kc_idx)
            cluster_to_questions[cluster_id.item()].add(i)

        #For correct shift, update prev number of kcs and steps
        num_prev_kcs += cur_num_kcs
        num_prev_steps += cur_num_steps

    # Building cluster_kc_step_pairs based on updated cluster_to_steps
    cluster_kc_step_pairs = []
    cluster_kc_quest_pairs = []
    for cluster_id, steps in cluster_to_steps.items():
        kcs = cluster_to_kcs[cluster_id]
        questions = list(cluster_to_questions[cluster_id])  # Convert set to list to index it
        for kc in kcs:
            for step in steps:
                cluster_kc_step_pairs.append([kc, step])
            for question in questions:
                cluster_kc_quest_pairs.append([kc, question])
    
    # Stack question-related tensors
    question_ids = torch.stack(question_ids)
    question_masks = torch.stack(question_masks)
    
    # Convert lists to tensors
    all_step_ids = torch.cat(all_step_ids)
    all_step_masks = torch.cat(all_step_masks)
    all_kc_ids = torch.cat(all_kc_ids)
    all_kc_masks = torch.cat(all_kc_masks)
    all_kc_step_pairs = torch.cat(all_kc_step_pairs)
    all_kc_quest_pairs = torch.cat(all_kc_quest_pairs)
    all_cluster_kc_step_pairs = torch.tensor(cluster_kc_step_pairs, dtype=torch.long)
    all_cluster_kc_quest_pairs = torch.tensor(cluster_kc_quest_pairs, dtype=torch.long)  # New tensor for cluster-question pairs

    # Convert lengths to tensor for use in the model or loss calculations
    step_lengths = torch.tensor(step_lengths)
    kc_lengths = torch.tensor(kc_lengths)
    
    return {
        'question_ids': question_ids,
        'question_mask': question_masks,
        'step_ids': all_step_ids,
        'step_mask': all_step_masks,
        'step_lengths': step_lengths,
        'kc_ids': all_kc_ids,
        'kc_mask': all_kc_masks,
        'kc_lengths': kc_lengths,
        'kc_step_pairs': all_kc_step_pairs,
        'kc_quest_pairs': all_kc_quest_pairs,
        'cluster_kc_step_pairs': all_cluster_kc_step_pairs,
        'cluster_kc_quest_pairs': all_cluster_kc_quest_pairs
    }
