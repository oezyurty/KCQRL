import argparse
from sentence_transformers import SentenceTransformer
import torch
import hdbscan
import numpy as np
from collections import defaultdict
import json

def normalize_embeddings(embeddings):
    # Normalize each row such that their Euclidean norm will be 1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def main(args):
    path_kc_questions_map = args.path_kc_questions_map
    write_path = args.write_path
    
    
    # Load KC questions map from JSON
    with open(path_kc_questions_map, 'r') as file:
        kc_questions_map = json.load(file)

    list_kcs = [k for k in kc_questions_map]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with GPU support
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    # Generate embeddings
    embeddings = model.encode(list_kcs, show_progress_bar=True, convert_to_tensor=True)

    # Move embeddings from GPU to CPU for clustering
    embeddings = embeddings.cpu().numpy()

    # Normalize embeddings
    embeddings = normalize_embeddings(embeddings)

    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean')
    clusters = clusterer.fit_predict(embeddings)

    # Organizing clusters
    cluster_map = defaultdict(list)
    for idx, cluster_label in enumerate(clusters):
        cluster_map[cluster_label].append(list_kcs[idx])

    # Sort clusters by size (excluding outliers initially)
    sorted_clusters = {}
    non_outlier_clusters = {clust: members for clust, members in cluster_map.items() if clust != -1}
    sorted_cluster_ids = sorted(non_outlier_clusters, key=lambda x: len(non_outlier_clusters[x]), reverse=True)

    # Assign new cluster IDs to sorted clusters
    for new_id, old_id in enumerate(sorted_cluster_ids):
        sorted_clusters[new_id] = non_outlier_clusters[old_id]

    # Handle outliers by giving each a unique cluster ID
    outlier_count = len(sorted_clusters)
    for idx, kc in enumerate(cluster_map[-1]):
        sorted_clusters[outlier_count + idx] = [kc]

    # Convert and write kc_to_problems object to json
    with open(write_path, "w") as outfile:
        json.dump(sorted_clusters, outfile, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster KCs using HDBSCAN.")
    parser.add_argument('path_kc_questions_map', type=str, help="Path to the KC questions map JSON file.")
    parser.add_argument('write_path', type=str, help="Path to write the clustered KC questions map JSON file.")
    args = parser.parse_args()

    main(args)