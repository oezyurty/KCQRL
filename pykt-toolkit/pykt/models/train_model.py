import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate, evaluate_cskt
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd
import pdb
from sklearn.utils.class_weight import compute_class_weight

import wandb

def get_device():
    if torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for CUDA GPU support
        return torch.device("cuda")
    else:  # Fallback to CPU if neither MPS nor CUDA is available
        return torch.device("cpu")

device = get_device()
#device = torch.device("cpu")

def get_class_weights(y, weighted_loss=0):
    if weighted_loss == 0:
        return torch.ones_like(y)
    else:
        weight = torch.ones_like(y)
        y_np = y.detach().cpu().numpy()
        flat_y_np = y_np.flatten()
        # Check if both labels present, otherwise return default ones as weight
        if not (0 in flat_y_np and 1 in flat_y_np):
            return weight
        else:
            weight_0, weight_1 = compute_class_weight(class_weight="balanced", classes=[0,1], y=flat_y_np)
            weight[y_np == 0] = weight_0
            weight[y_np == 1] = weight_1
            return weight

def cal_loss(model, ys, r, rshft, sm, preloss=[], weighted_loss=0):
    model_name = model.model_name

    if model_name in ["atdkt", "simplekt", "bakt_time", "sparsekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        # Get weights for prediction
        weight_loss = get_class_weights(y, weighted_loss)
        loss1 = binary_cross_entropy(y.float(), t.float(), weight=weight_loss)

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1

    elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # Get weights for prediction
        weight_loss = get_class_weights(y, weighted_loss)
        loss = binary_cross_entropy(y.float(), t.float(), weight=weight_loss)
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        # Get weights for prediction
        weight_loss = get_class_weights(y_next, weighted_loss)
        loss = binary_cross_entropy(y_next.float(), r_next.float(), weight=weight_loss)

        loss_r = binary_cross_entropy(y_curr.float(), r_curr.float()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # Get weights for prediction
        weight_loss = get_class_weights(y, weighted_loss)
        loss = binary_cross_entropy(y.float(), t.float(), weight=weight_loss) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # Get weights for prediction
        weight_loss = get_class_weights(y, weighted_loss)
        #Below only accepts weight per class, not per element
        weight = torch.Tensor([weight_loss.max(), weight_loss.min()])
        criterion = nn.BCELoss(reduction='none', weight=weight)        
        loss = criterion(y, t).sum()
    
    return loss

def model_forward_cskt(model, data, weighted_loss=0):
    dcur = data
    # Get the needed fields from the data
    q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)
    
    y, _ = model(q.long(), c.long(), r.long())
    
    y_masked = torch.masked_select(y, sm)
    r_masked = torch.masked_select(r, sm)

    # Get weights for prediction
    weight_loss = get_class_weights(y_masked, weighted_loss)
    
    loss = binary_cross_entropy(y_masked.float(), r_masked.float(), weight=weight_loss)
    
    return loss


def model_forward(model, data, rel=None, weighted_loss=0):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        for k in dcur.keys():
            dcur[k] = dcur[k].to(device)
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "sparsekt"]:
        for key, value in dcur.items():
            if isinstance(value, torch.Tensor):
                dcur[key] = value.to(device)
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm, weighted_loss=weighted_loss)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm, weighted_loss=weighted_loss)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        # We need to move all components of data to device
        for k in data.keys():
            data[k] = data[k].to(device)
        y,loss = model.train_one_step(data, weighted_loss=weighted_loss)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss, weighted_loss=weighted_loss)
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None, use_wandb=0, weighted_loss=0):
    max_auc, best_epoch = 0, -1
    valid_auc_checkpoint, valid_acc_checkpoint = -1, -1
    train_step = 0

    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            ## YO Edition: Special forward function for CSKT
            if model.model_name == "cskt":
                loss = model_forward_cskt(model, data, weighted_loss=weighted_loss)
            elif model.model_name=='rkt':
                loss = model_forward(model, data, rel, weighted_loss=weighted_loss)
            else:
                loss = model_forward(model, data, weighted_loss=weighted_loss)
            opt.zero_grad()
            loss.backward()#compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name in ["gkt", "skvmn_que"] and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        
        ## YO Edition: Special forward function for CSKT
        if model.model_name == "cskt":
            valid_auc, valid_avg_prc, valid_acc = evaluate_cskt(model, valid_loader, model.model_name, rel)
            train_auc, train_avg_prc, train_acc = evaluate_cskt(model, train_loader, model.model_name, rel)
        elif model.model_name=='rkt':
            valid_auc, valid_avg_prc, valid_acc = evaluate(model, valid_loader, model.model_name, rel)
            train_auc, train_avg_prc, train_acc = evaluate(model, train_loader, model.model_name, rel)
        else:
            valid_auc, valid_avg_prc, valid_acc = evaluate(model, valid_loader, model.model_name)
            train_auc, train_avg_prc, train_acc = evaluate(model, train_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        # It means that we log the metrics
        if use_wandb == 1:
            dict_log = {"Train Loss": loss_mean,
                        "Train AUC": train_auc,
                        "Train AUPRC": train_avg_prc,
                        "Train Accuracy": train_acc,
                        "Validation AUC": valid_auc,
                        "Validation AUPRC": valid_avg_prc,
                        "Validation Accuracy": valid_acc}
            wandb.log(dict_log, step=train_step, commit=True)

        if valid_auc > max_auc+2e-4:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = valid_auc
            valid_auc_checkpoint, valid_avg_prc_checkpoint, valid_acc_checkpoint = valid_auc, valid_avg_prc, valid_acc
            best_epoch = i
            test_auc, test_avg_prc, test_acc = -1, -1, -1
            window_test_auc, window_test_avg_prc, window_test_acc = -1, -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    test_auc, test_avg_prc, test_acc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_test_auc, window_test_avg_prc, window_test_acc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            
        print(f"Epoch: {i}, train_auc: {train_auc:.4}, train_avg_prc: {train_avg_prc:.4}, train_acc: {train_acc:.4}, valid_auc: {valid_auc:.4}, valid_avg_prc: {valid_avg_prc:.4}, valid_acc: {valid_acc:.4}")
        print(f"            best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            test_auc: {round(test_auc,4)}, test_acc: {round(test_acc,4)}, window_test_auc: {round(window_test_auc,4)}, window_test_acc: {round(window_test_acc,4)}")
        
        if model.model_name == "cskt":
            alpha = torch.sigmoid(model.alpha_logit)
            print("Mixing alpha is: ", alpha)

        if i - best_epoch >= 5:
            break
    #return test_auc, test_avg_prc, test_acc, window_test_auc, window_test_avg_prc, window_test_acc, valid_auc_checkpoint, valid_avg_prc_checkpoint, valid_acc_checkpoint, best_epoch

    dict_res = {
        'test_auc': test_auc,
        'test_avg_prc': test_avg_prc,
        'test_acc': test_acc,
        'window_test_auc': window_test_auc,
        'window_test_avg_prc': window_test_avg_prc,
        'window_test_acc': window_test_acc,
        'valid_auc_checkpoint': valid_auc_checkpoint,
        'valid_avg_prc_checkpoint': valid_avg_prc_checkpoint,
        'valid_acc_checkpoint': valid_acc_checkpoint,
        'best_epoch': best_epoch
    }

    return dict_res 