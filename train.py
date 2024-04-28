# from torch.nn.modules.module import _IncompatibleKeys
import torch
from utils.util import EarlyStopping, save_file, set_gpu_devices, pause, set_seed
from utils.logger import logger
import argparse
import numpy as np
import tqdm
import json

parser = argparse.ArgumentParser(description="train parameter")
# General
parser.add_argument("-v", type=str, required=True, help="version")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=16)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=3e-5)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=40)
parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
parser.add_argument("-patience", "-pa", type=int, help="patience of ReduceonPleatu", default=3)
parser.add_argument("-gamma", "-ga", type=float, help="gamma of MultiStepLR", default=0.5)
parser.add_argument("-decay", type=float, help="weight decay", default=0)
parser.add_argument("-load", type=str, default=None, help="load check point")
parser.add_argument("-frame_len", type=int, default=1)

# Model
parser.add_argument("-detector_weights", type=str, required=True)
parser.add_argument("-adapter_mlp_ratio", type=float, default=0.25)
parser.add_argument("-cast_choice", type=str, default="decoder", choices=["encoder", "decoder", "encoder_decoder"])
parser.add_argument("-num_query_qformer", type=int, default=32)
parser.add_argument("-lora_r", type=int, default=8)
parser.add_argument("-lora_alpha", type=int, default=16)
args = parser.parse_args()


set_gpu_devices(args.gpu)
set_seed(3407)

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from networks.model import QAModel
from DataLoader import SQ3DDataset

torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))


def train(model, optimizer, train_loader, device):
    model.train()
    total_step = len(train_loader)
    epoch_loss = 0.0
    prediction_list = []
    answer_list = []
    for inputs in tqdm.tqdm(train_loader):
        pc_inputs, frame_feature, frame_feature_mask, question, question_prompt, qformer_input_ids, qformer_attention_mask, question_id, scene_id, answer_cat, answer_cats, answer_cat_scores = inputs
        
        for key in pc_inputs.keys():
            pc_inputs[key] = pc_inputs[key].to(device)
        frame_feature = frame_feature.to(device)
        frame_feature_mask = frame_feature_mask.to(device)
        qformer_inputs = {
            "qformer_input_ids": qformer_input_ids.to(device),
            "qformer_attention_mask": qformer_attention_mask.to(device)
        }
        
        out_f = model(point_inputs=pc_inputs, frame_feature=frame_feature, frame_mask=frame_feature_mask, qformer_inputs=qformer_inputs, question_prompt=question_prompt)
        
        answer_cat = answer_cat.to(device)
        loss = F.cross_entropy(out_f, answer_cat)
            
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        
        pred_answers_at1 = torch.argmax(out_f, 1)
        epoch_loss += loss.item()
        prediction_list.append(pred_answers_at1)
        answer_list.append(answer_cat)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long().cpu()
    acc_num = (predict_answers == ref_answers).sum()
    
    return epoch_loss / total_step, acc_num * 100.0 / len(ref_answers)
    

def eval(model, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(val_loader):
            pc_inputs, frame_feature, frame_feature_mask, question, question_prompt, qformer_input_ids, qformer_attention_mask, question_id, scene_id, answer_cat, answer_cats, answer_cat_scores = inputs
        
            for key in pc_inputs.keys():
                pc_inputs[key] = pc_inputs[key].to(device)
            frame_feature = frame_feature.to(device)
            frame_feature_mask = frame_feature_mask.to(device)
            qformer_inputs = {
                "qformer_input_ids": qformer_input_ids.to(device),
                "qformer_attention_mask": qformer_attention_mask.to(device)
            }
        
            out = model(point_inputs=pc_inputs, frame_feature=frame_feature, frame_mask=frame_feature_mask, qformer_inputs=qformer_inputs, question_prompt=question_prompt)
        
            pred_answers_at1 = torch.argmax(out, 1)
            prediction_list.append(pred_answers_at1)
            answer_list.append(answer_cat)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long().cpu()
    acc_num = (predict_answers == ref_answers).sum()
    
    return acc_num * 100.0 / len(ref_answers)



if __name__ == "__main__":

    logger, sign =logger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SQ3DDataset('train', frame_len=args.frame_len)
    val_dataset = SQ3DDataset('val', frame_len=args.frame_len)
    test_dataset = SQ3DDataset('test', frame_len=args.frame_len)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=16, pin_memory=True)

    # hyper setting
    ans_dict_path="/storage_fast/cfeng/sq3d/ans_dict/scanqa_format/with_test/answer_cands.json"
    with open(ans_dict_path, "r") as fp:
        ans_words = json.load(fp)
    epoch_num = args.epoch
    model = QAModel(
        cast_choice=args.cast_choice, adapter_mlp_ratio=args.adapter_mlp_ratio, num_query_qformer=args.num_query_qformer, 
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, ans_num=len(ans_words))
    
    if args.detector_weights is not None:
        weight_dict = torch.load(args.detector_weights)
        model.load_state_dict(weight_dict["model"], strict=False)
        logger.debug(f"Successfully load detector weights from {args.detector_weights}")
    
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        logger.debug(f"Successfully load checkpoint from {args.load}")
    
    param_dicts = [
    {"params": [p for n, p in model.named_parameters() if p.requires_grad is True]}]
    optimizer = torch.optim.AdamW(params = param_dicts, lr=args.lr, weight_decay=args.decay)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=args.gamma, patience=args.patience, verbose=True)
    model.to(device)

    # train & val & test
    best_eval_score = 0.0
    best_epoch=1
    for epoch in range(1, epoch_num+1):
        train_loss, train_acc = train(model, optimizer, train_loader, device)
        eval_score = eval(model, val_loader, device)
        scheduler.step(eval_score)
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            best_epoch = epoch 
            best_model_path='./models/best_model-{}.ckpt'.format(sign)
            torch.save(model.state_dict(), best_model_path)
        
        test_score = eval(model, test_loader, device)
        logger.debug("==>Epoch:[{}/{}][LR{}][Train Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f} Test acc: {:.2f}".
        format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss,train_acc, eval_score, test_score))

    logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))
    
    last_model_path = './models/last_epoch_{}_model-{}.ckpt'.format(epoch, sign)
    torch.save(model.state_dict(), last_model_path)