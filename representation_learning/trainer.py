import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import os

class Trainer:
    def __init__(self, model, train_dataloader, optimizer, evaluator, device, args):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.alpha = args.alpha
        self.num_epochs = args.num_epochs
        self.patience = args.patience
        self.model_save_dir = args.model_save_dir
        self.T = args.temperature
        self.device = device
        # A new argument to disable clustering (DEFAULT IS ENABLING CLUSTERS)
        self.disable_clusters = args.disable_clusters

    def train(self):
        best_micro_f1 = 0
        cur_patience = 0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Iterate one epoch
            epoch_loss = self.train_epoch()
            print(f"Total Loss: {epoch_loss:.4f}")

            # Run eval
            avg_acc, total_acc, avg_f1, micro_f1 = self.evaluator.evaluate(self.model)
            print(f"Eval Micro F1: {micro_f1:.4f}")

            # Update Wandb scores after epoch
            wandb.log({'epoch_loss': epoch_loss, 'micro_f1': micro_f1, 'avg_f1': avg_f1})
            
            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                cur_patience = 0
                print("Saving model")
                self.save_model()

            else:
                cur_patience += 1

            if cur_patience >= self.patience:
                print("Stopping training")
                break

        print("Training complete.")

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        # Initialize tqdm progress bar
        progress_bar = tqdm(self.train_dataloader, desc='Training', leave=True)

        for i, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            batch = self._batch_to_device(batch)

            loss = self.process_batch(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar with latest loss
            progress_bar.set_postfix(loss=loss.item())

            # Update Wandb step loss
            wandb.log({'step_loss': loss.item()})

        return total_loss

    def process_batch(self, batch):
        question_embeddings = self.model(batch['question_ids'], batch['question_mask']).last_hidden_state[:, 0, :]
        step_embeddings = self.model(batch['step_ids'], batch['step_mask']).last_hidden_state[:, 0, :]
        kc_embeddings = self.model(batch['kc_ids'], batch['kc_mask']).last_hidden_state[:, 0, :]

        # Calculate similarities
        # kc_question_similarity = F.cosine_similarity(kc_embeddings.unsqueeze(1), question_embeddings.unsqueeze(0), dim=2)
        # kc_step_similarity = F.cosine_similarity(kc_embeddings.unsqueeze(1), step_embeddings.unsqueeze(0), dim=2)
        question_kc_similarity =  F.cosine_similarity(question_embeddings.unsqueeze(1), kc_embeddings.unsqueeze(0), dim=2)
        step_kc_similarity = F.cosine_similarity(step_embeddings.unsqueeze(1), kc_embeddings.unsqueeze(0), dim=2)

        # Continue as normal
        # kc_question_score = torch.exp(kc_question_similarity / self.T)
        # kc_step_score = torch.exp(kc_step_similarity / self.T)
        question_kc_score = torch.exp(question_kc_similarity / self.T)
        step_kc_score = torch.exp(step_kc_similarity / self.T)

        # loss = self.contrastive_loss(kc_question_score, batch['kc_quest_pairs'], batch['cluster_kc_quest_pairs']) + \
        #     self.alpha * self.contrastive_loss(kc_step_score, batch['kc_step_pairs'], batch['cluster_kc_step_pairs'])
        
        loss = self.contrastive_loss(question_kc_score, batch['kc_quest_pairs'], batch['cluster_kc_quest_pairs']) + \
            self.alpha * self.contrastive_loss(step_kc_score, batch['kc_step_pairs'], batch['cluster_kc_step_pairs'])

        #print("Loss:", loss.item())
        return loss


    def contrastive_loss(self, score_matrix, direct_pair_indices, clustered_pair_indices):
        pos_mask = torch.zeros_like(score_matrix)
        neg_mask = torch.ones_like(score_matrix)
        
        # We treat all the direct kc - quest/step pairs as positives
        pos_mask[direct_pair_indices[:, 1], direct_pair_indices[:, 0]] = 1
        neg_mask[direct_pair_indices[:, 1], direct_pair_indices[:, 0]] = 0
        
        # We discard all clustered kc - quest/step pairs as from negatives (in default mode)
        if not self.disable_clusters:
            neg_mask[clustered_pair_indices[:, 1], clustered_pair_indices[:, 0]] = 0
        
        pos_score = score_matrix * pos_mask
        neg_score = score_matrix * neg_mask

        pos_score_sum = pos_score.sum(dim=-1)
        neg_score_sum = neg_score.sum(dim=-1)

        # Check for any zeros or very small numbers in the denominator
        scores_sum = pos_score_sum + neg_score_sum

        # Safe-guarding against division by zero or log of zero
        scores_sum = scores_sum.clamp(min=1e-8)  # Avoid division by zero
        pos_score_sum = pos_score_sum.clamp(min=1e-8)  # Avoid log(0)

        # Calculate the loss for each element in pos_score
        element_loss = -1 * torch.log(pos_score / scores_sum.unsqueeze(-1))

        # Mask out the elements that are zero in pos_score
        element_loss = element_loss * pos_mask

        # Calculate the mean loss for each row, considering only non-zero elements
        row_loss = element_loss.sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1e-8)

        # Filter out rows where pos_mask has no positives
        valid_rows = pos_mask.sum(dim=-1) > 0  # Boolean mask of rows with at least one positive
        row_loss = row_loss[valid_rows]  # Filter to include only valid rows

        # Compute the mean loss over valid rows only
        cl_loss = row_loss.mean()

        return cl_loss


    def _batch_to_device(self, batch):
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
        return batch

    def save_model(self):
        # Define a directory to save the model and tokenizer
        model_dir = self.model_save_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the model
        model_save_path = os.path.join(model_dir, 'bert_finetuned.bin')
        torch.save(self.model.state_dict(), model_save_path)