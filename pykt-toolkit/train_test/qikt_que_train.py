import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #dataset config
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--fold", type=int, default=0)

    # train config
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=200)

    #log config & save config
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project_name", type=str, default="", help="if not empty string, it will overwrite the default wandb project name")
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved_model")

    # model config
    parser.add_argument("--model_name", type=str, default="qikt_que")
    parser.add_argument("--emb_type", type=str, default="iekt")
    parser.add_argument("--emb_path", type=str, default="", help="if not empty string, it will overwrite the emb path in data config.")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--mlp_layer_num", type=int, default=2)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    parser.add_argument("--loss_q_all_lambda", type=float, default=0)
    parser.add_argument("--loss_c_all_lambda", type=float, default=0)
    parser.add_argument("--loss_q_next_lambda", type=float, default=0)
    parser.add_argument("--loss_c_next_lambda", type=float, default=0)
    
    parser.add_argument("--output_q_all_lambda", type=float, default=1)
    parser.add_argument("--output_c_all_lambda", type=float, default=1)
    parser.add_argument("--output_q_next_lambda", type=float, default=0)
    parser.add_argument("--output_c_next_lambda", type=float, default=1)

    # Important two model configs below
    parser.add_argument('--flag_load_emb', action='store_true', help="Explicitly control if the embeddings will be loaded from path")
    parser.add_argument('--flag_emb_freezed', action='store_true', help="Explicitly control if the embeddings will be freezed or trained")
    
    parser.add_argument("--output_mode", type=str, default="an")
    args = parser.parse_args()

    params = vars(args)
    remove_keys = ['output_mode'] + [x for x in params.keys() if "lambda" in x]
    other_config = {}
    for k in remove_keys:
        other_config[k] = params[k]
        del params[k]
    params['other_config'] = other_config
    main(params)
