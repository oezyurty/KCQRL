import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="atkt_que")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
     
    parser.add_argument("--skill_dim", type=int, default=256)
    parser.add_argument("--answer_dim", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=80)
    parser.add_argument("--attention_dim", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=10)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    #New args for functionality
    parser.add_argument("--emb_path", type=str, default="", help="if not empty string, it will overwrite the emb path in data config.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)
    parser.add_argument("--wandb_project_name", type=str, default="", help="if not empty string, it will overwrite the default wandb project name")
    # Important two model configs below
    parser.add_argument('--flag_load_emb', action='store_true', help="Explicitly control if the embeddings will be loaded from path")
    parser.add_argument('--flag_emb_freezed', action='store_true', help="Explicitly control if the embeddings will be freezed or trained")


    args = parser.parse_args()

    params = vars(args)
    main(params)
