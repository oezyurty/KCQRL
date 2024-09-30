import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="iekt_que")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--emb_path", type=str, default="", help="if not empty string, it will overwrite the emb path in data config.")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)

    # model config
    # (self, num_q,num_c,emb_size,max_concepts,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu',seed=0):
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    # Important two model configs below
    parser.add_argument('--flag_load_emb', action='store_true', help="Explicitly control if the embeddings will be loaded from path")
    parser.add_argument('--flag_emb_freezed', action='store_true', help="Explicitly control if the embeddings will be freezed or trained")

    parser.add_argument("--emb_size", type=int, default=64, help='hidden size for nodes')
    parser.add_argument("--lamb", type=int, default=40,help='hyper parameter for loss')
    parser.add_argument("--n_layer", type=int, default=1,help='number of mlp hidden layers')
    parser.add_argument("--cog_levels", type=int, default=10,help='the response action space for cognition estimation')
    parser.add_argument("--acq_levels", type=int, default=10,help='the response action space for  sensitivity estimation')
    parser.add_argument("--gamma", type=float, default=0.93)
    
    # train config
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project_name", type=str, default="", help="if not empty string, it will overwrite the default wandb project name")
    parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)
