import os
import json
import h5py
from glob import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import networks.detr_3d.utils.pc_util as pc_util
from utils.scanqa_util import Answer

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


class SQ3DDataset(Dataset):
    def __init__(self, split,
                 llm_name="microsoft/deberta-v3-large",
                 
                 frame_len=1,  # 选多少个frame
                 sampled_frame_path="<path_to_sampled_frames>/sq3d_{}_BLIP2_sampled_frames.json",
                 blip2_feat_path="<path_to_blip2_feat>/SQA3D_{}_blip2_multimodal_feature.h5", 
                 
                 qa_data_path="<path_to_qa_data>/SQA_{}.json",
                 ans_dict_path="<path_to_answer_dict>/answer_{}.json",
                 
                 num_points=40000,
                 pc_path="<path_to_point_clouds>",
    ):
        
        self.split = split
        self.llm_name = llm_name
        
        self.frame_len = frame_len
        self.sampled_frame_path = sampled_frame_path
        self.blip2_feat_path = blip2_feat_path
        
        self.num_points = num_points
        self.pc_path = pc_path
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_name, add_bos_token=False)
        self.qtokenizer = AutoTokenizer.from_pretrained("CH3COOK/bert-base-embedding")
        self.qtokenizer.pad_token = tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        self.max_length = 70
        self.tokenizer_config = dict(
            max_length=self.max_length, 
            padding='max_length', 
            truncation=None, 
            return_tensors='pt'
        )
            
        # Question
        with open(qa_data_path.format(split), "r") as fp:
            self.qa_data = json.load(fp)  # type: list
        
        # Answer
        with open(ans_dict_path.format("with_test", "cands"), "r") as fp:
            self.answer_cands = json.load(fp)
        with open(ans_dict_path.format("with_test", "counter"), "r") as fp:
            self.answer_counter = json.load(fp)
        self.answer_vocab = Answer(self.answer_cands)
        self.num_answers = len(self.answer_cands) 
        
        # Sampled Frame
        with open(sampled_frame_path.format(split), "r") as fp:
            self.sampled_frame = json.load(fp)
        
        print(f"Finish loading {split} dataset.")
    
    
    def __len__(self):
        return len(self.qa_data)
    
    
    def get_answer_score(self, freq):
        if freq == 0:
            return .0
        elif freq == 1:
            return .3
        elif freq == 2:
            return .6
        elif freq == 3:
            return .9
        else:
            return 1.
    
    
    def __getitem__(self, index):
        scene_id = self.qa_data[index]["scene_id"]
        
        # Get question & answer
        question_id = self.qa_data[index]['question_id']
        question = self.qa_data[index]['question']
        situation = self.qa_data[index]['situation']
        if self.llm_name == "FacebookAI/roberta-base":
            question_prompt = "Situation: {} Question: {} Answer: <mask>".format(
                self.qa_data[index]['situation'], 
                self.qa_data[index]['question']
                )
        else:
            question_prompt = "Situation: {} Question: {} Answer: [MASK]".format(
                self.qa_data[index]['situation'], 
                self.qa_data[index]['question']
                )
        qformer_inputs = self.qtokenizer.batch_encode_plus([situation + " " + question], **self.tokenizer_config)
        qformer_input_ids =  qformer_inputs['input_ids'][0].int()
        qformer_attention_mask = qformer_inputs['attention_mask'][0].float()
        assert qformer_attention_mask.size(0) == self.max_length, qformer_attention_mask.size(0)
        
        answers = self.qa_data[index].get('answers', [])
        answer_cats = torch.zeros(self.num_answers) 
        answers = self.qa_data[index]["answers"]
        answer_inds = [self.answer_vocab.stoi(answer) for answer in answers]
     
        answer_cat_scores = np.zeros(self.num_answers)
        for answer, answer_ind in zip(answers, answer_inds):
            if answer_ind < 0:
                continue                    
            answer_cats[answer_ind] = 1
            answer_cat_score = self.get_answer_score(self.answer_counter.get(answer, 0))
            answer_cat_scores[answer_ind] = answer_cat_score
        
        assert answer_cats.sum() > 0
        assert answer_cat_scores.sum() > 0
        
        answer_cat = answer_cats.argmax()
        
        # Get Point Cloud
        mesh_vertices = np.load(os.path.join(self.pc_path, scene_id) + "_aligned_vert.npy")
        # use color
        point_cloud = mesh_vertices[:, 0:6]
        point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
        pcl_color = point_cloud[:, 3:]
        # use nomal
        normals = mesh_vertices[:,6:9]
        point_cloud = np.concatenate([point_cloud, normals], 1)
        # use height
        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        pcl_color = pcl_color[choices]
        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)
        pc_inputs = {
            "point_clouds": torch.from_numpy(point_cloud.astype(np.float32)),
            "point_cloud_dims_min": torch.from_numpy(point_cloud_dims_min.astype(np.float32)),
            "point_cloud_dims_max": torch.from_numpy(point_cloud_dims_max.astype(np.float32))
            }
        
        # Get Multimoal Feature #
        with h5py.File(self.blip2_feat_path.format(question_id)) as fp:
            multmodal_feature = fp["multimodal_feature"][:]
        multmodal_feature = torch.from_numpy(multmodal_feature).float()[:self.frame_len, ...]
        
        num_frames, num_patch, feat_dim = multmodal_feature.size()
        multmodal_feature_mask = torch.ones(num_frames * num_patch).bool()
        padding_len = self.frame_len - num_frames
        
        if padding_len > 0:  # 需要padding
            multmodal_feature = torch.concat([multmodal_feature, torch.zeros(padding_len, num_patch, feat_dim)], dim=0)
            multmodal_feature = multmodal_feature.view(self.frame_len * num_patch, feat_dim)
            multmodal_feature_mask = torch.concat([multmodal_feature_mask, torch.zeros(padding_len * num_patch).bool()], dim=0)
        else:
            multmodal_feature = multmodal_feature.view(self.frame_len * num_patch, feat_dim)
        
        assert self.frame_len * num_patch == multmodal_feature.size(0), f"multmodal_feature size: {multmodal_feature.size()}, expected: {self.frame_len}"
        assert self.frame_len * num_patch == multmodal_feature.size(0), f"multmodal_feature_mask size: {multmodal_feature_mask.size()}, expected: {self.frame_len}"
        
        return pc_inputs, multmodal_feature, multmodal_feature_mask, situation + " " + question, question_prompt, qformer_input_ids, qformer_attention_mask, question_id, scene_id, answer_cat, answer_cats, answer_cat_scores
        


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils.util import set_seed
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    seed = 999
    set_seed(seed)
    
    train_dataset = SQ3DDataset(split="test", frame_len=10)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=16)
    print(len(train_loader))

    for idx, sample in enumerate(train_loader):
        pc_inputs, multmodal_feature, multmodal_feature_mask, question, question_prompt, qformer_input_ids, qformer_attention_mask, question_id, scene_id, answer_cat, answer_cats, answer_cat_scores = sample

        print(pc_inputs["point_clouds"].size(), pc_inputs["point_clouds"].dtype)
        print(pc_inputs["point_cloud_dims_min"].size(), pc_inputs["point_cloud_dims_min"].dtype)
        print(pc_inputs["point_cloud_dims_max"].size(), pc_inputs["point_cloud_dims_max"].dtype)
        print(multmodal_feature.size(), multmodal_feature.dtype)
        print(multmodal_feature_mask.size(), multmodal_feature_mask.dtype)
        print(question, type(question))
        print(question_prompt, type(question_prompt))
        print(qformer_input_ids.size(), type(qformer_input_ids))
        print(qformer_attention_mask.size(), type(qformer_attention_mask))
        print(question_id)
        print(scene_id, scene_id[0])
        print(answer_cat, answer_cat.size())
        print(answer_cats, answer_cats.size())
        print(answer_cat_scores, answer_cat_scores.size())
        
        print("\n===================================================\n")
        print(idx, "/", len(train_loader))
        # if idx == 0:
        #     break