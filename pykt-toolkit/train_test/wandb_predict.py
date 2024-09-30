import os
import argparse
import json
import copy
import torch
import pandas as pd

from pykt.models import evaluate,evaluate_question,load_model
from pykt.datasets import init_test_datasets

def get_device():
    if torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for CUDA GPU support
        return torch.device("cuda")
    else:  # Fallback to CPU if neither MPS nor CUDA is available
        return torch.device("cpu")

device = get_device()
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        if "wandb_project_name" in params and params["wandb_project_name"] != "":
            wandb.init(project=params["wandb_project_name"])
        else:
            wandb.init()

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    raw_config = json.load(open(os.path.join(save_dir,"config.json")))

    if params['use_wandb'] == 1:
        # Do the save for wandb
        wandb.config.update(raw_config['params'])
        wandb.config.update({"checkpoint_path": save_dir})

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]    
        # Emb_path should be read from data_config. 
        # data_config is later updated based on the trained_params["emb_path"]. 
        if "emb_path" in model_config:
            del model_config["emb_path"]

        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt", "bakt_time", "sakt_que", "saint_que"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]    

    # Add emb_path from params to data config, if model used emb_path from params
    if "emb_path" in trained_params:
        data_config["emb_path"] = trained_params["emb_path"]
    if model_name not in ["dimkt"]:        
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = ""
    if params["save_all_preds"] == 1:
        save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")
        print(f"Predictions will be saved to {save_test_path}")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))                

    if model.model_name == "rkt":
        testauc, testavg_prc, testacc = evaluate(model, test_loader, model_name, rel, save_path=save_test_path)
    else:
        testauc, testavg_prc, testacc = evaluate(model, test_loader, model_name, save_path=save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = ""
    if params["save_all_preds"] == 1:
        save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
        print(f"Predictions will be saved to {save_test_window_path}")
    
    if model.model_name == "rkt":
        window_testauc, window_testavg_prc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_path=save_test_window_path)
    else:
        window_testauc, window_testavg_prc, window_testacc = evaluate(model, test_window_loader, model_name, save_path=save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    # question_testauc, question_testacc = -1, -1
    # question_window_testauc, question_window_testacc = -1, -1
  
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_path=save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            
    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_path=save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

        
    # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    # print(f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}")
    
    print(dres)
    dres.update(raw_config['params'])

    if params['use_wandb'] ==1:
        wandb.log(dres)

    #Save dres: the data frame of results
    results_save_path = os.path.join(save_dir, "prediction_results.json")
    with open(results_save_path, 'w') as json_file:
        json.dump(dres, json_file, indent=2)  # indent=4 is used for pretty printing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--save_all_preds", type=int, default=0, help="1 for saving all predictions/ground truths as list: takes a lot of time and space.")
    parser.add_argument("--wandb_project_name", type=str, default="The project name of wandb to keep track of predictions.")

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
