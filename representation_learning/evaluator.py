import torch
import json

class Evaluator:
    def __init__(self, tokenizer, device, args):
        # Load both dataset and cluster files
        with open(args.json_file_dataset, 'r') as f:
            self.data_questions = json.load(f)
        with open(args.json_file_kc_questions, 'r') as file:
            self.kc_questions_map = json.load(file)
        with open(args.json_file_cluster_kc, 'r') as f:
            self.cluster_to_kcs = json.load(f)

        # Get kc to cluster mapping from cluster to kc
        self.kc_to_cluster = self.__get_kc_to_cluster__(self.cluster_to_kcs)

        # Save other inputs
        self.tokenizer = tokenizer
        self.eval_batch_size = args.eval_batch_size
        self.max_length = args.max_length
        self.max_length_kc = args.max_length_kc
        self.device = device

        self.__init_eval_dataset__()

    def __get_kc_to_cluster__(self, cluster_to_kcs):
        # Create kc_to_cluster mapping from cluster_to_kcs
        kc_to_cluster = {}
        for cluster_id, kcs in cluster_to_kcs.items():
            for kc in kcs:
                kc_to_cluster[kc] = cluster_id
        return kc_to_cluster

    def __init_eval_dataset__(self):
        #Read the kcs, questions and solution steps
        self.list_kcs = [k for k in self.kc_questions_map]
        self.list_questions = [value['question'] for key, value in self.data_questions.items()]
        self.list_sol_steps = [[sol for sol in value['step_by_step_solution_list']] for key,value in self.data_questions.items()]

        # Prepend special tokens
        self.kcs = ['[KC] ' + kc for kc in self.list_kcs]
        self.questions = ['[Q] ' + q for q in self.list_questions]
        self.sol_steps = [['[S] ' + step for step in sol_steps] for sol_steps in self.list_sol_steps]

    # Get the similarity matrix
    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    # Helper function to batch text data and convert to embeddings
    def __text_to_embeddings__(self, texts, model, max_length):
        embeddings = []
        for i in range(0, len(texts), self.eval_batch_size):
            batch_texts = texts[i:i + self.eval_batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :])  # Extract [CLS] token embeddings
        return torch.cat(embeddings, dim=0)

    def evaluate(self, model):
        model.eval()
        
        #Get the embeddings of kcs, questions and steps
        kc_embeddings = self.__text_to_embeddings__(self.kcs, model, max_length=self.max_length_kc)
        question_embeddings = self.__text_to_embeddings__(self.questions, model, max_length=self.max_length)
        # Flatten the solution steps before getting embeddings in a batch
        flat_solution_steps = [step for sublist in self.sol_steps for step in sublist]
        flat_solution_embeddings = self.__text_to_embeddings__(flat_solution_steps, model, max_length=self.max_length)
        # Map flat embeddings back to their respective lists using original lengths
        sol_step_embeddings = []
        start_idx = 0
        for steps in self.sol_steps:
            end_idx = start_idx + len(steps)
            sol_step_embeddings.append(flat_solution_embeddings[start_idx:end_idx])
            start_idx = end_idx

        # Compute similarity matrix between KCs and questions
        similarity_matrix_kc_question = self.sim_matrix(kc_embeddings, question_embeddings)

        # Compute similarity matrix between KCs and max of solution steps
        similarity_matrix_kc_sol_steps = torch.stack([
            torch.max(self.sim_matrix(kc_embeddings, step_embeddings), dim=1).values
            for step_embeddings in sol_step_embeddings
        ]).T

        # Get the average score of both similarity matrices
        final_mean_similarity_matrix = (similarity_matrix_kc_question + similarity_matrix_kc_sol_steps) / 2

        avg_acc, total_acc, avg_f1, micro_f1 = self.calculate_accuracies_clustered(final_mean_similarity_matrix)

        return avg_acc, total_acc, avg_f1, micro_f1 

    def calculate_accuracies_clustered(self, similarity_matrix):
    
        accuracies = {}
        total_correct = 0
        total_problems = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for kc_idx, kc in enumerate(self.list_kcs):
            # Determine the cluster and collect problems indices from all KCs in that cluster
            cluster_kcs = self.cluster_to_kcs[self.kc_to_cluster[kc]]
            true_problem_indices = set()
            for cluster_kc in cluster_kcs:
                true_problem_indices.update(self.kc_questions_map[cluster_kc])

            # The number of problems for calculating F1 should be based only on the current KC
            num_problems = len(self.kc_questions_map[kc])
            total_problems += num_problems

            # Get top num_problems indices from the similarity matrix for this KC
            top_n_indices = torch.topk(similarity_matrix[kc_idx], k=num_problems).indices

            # Calculate initial correct predictions
            predicted_correct = sum([1 for idx in top_n_indices if idx.item() in true_problem_indices])
            total_correct += predicted_correct

            # Calculate accuracy
            accuracy = predicted_correct / num_problems if num_problems > 0 else 0
            
            accuracies[kc] = {
                'total_problems': num_problems,
                'correct_predictions': predicted_correct,
                'accuracy': accuracy
            }

            top_n_indices = torch.topk(similarity_matrix[kc_idx], k=num_problems).indices
            
            tp = sum([1 for idx in top_n_indices if idx.item() in true_problem_indices])
            fp = num_problems - tp
            fn = num_problems - tp

            precision = tp / num_problems if num_problems > 0 else 0
            recall = tp / num_problems if num_problems > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


            accuracies[kc].update({
                'f1': f1,
                'n': num_problems
            })

            total_tp += tp
            total_fp += fp
            total_fn += fn

        total_accuracy = total_correct / total_problems if total_problems > 0 else 0
        average_accuracy = sum([data['accuracy'] for data in accuracies.values()]) / len(accuracies) if accuracies else 0
        average_f1 = sum(data['f1'] for data in accuracies.values()) / len(accuracies) if accuracies else 0
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        #print(f"Average Accuracy: {average_accuracy:.2f}")
        #print(f"Total Accuracy: {total_accuracy:.2f}")
        #print(f"Average F1: {average_f1:.2f}")
        #print(f"Micro F1: {micro_f1:.2f}")

        return average_accuracy, total_accuracy, average_f1, micro_f1
