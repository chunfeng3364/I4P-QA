import torch
import torch.nn as nn

from transformers import AutoTokenizer, DebertaV2Tokenizer


class LanguageEncoder(nn.Module):
    def __init__(self, llm_name="microsoft/deberta-v3-large", lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        
        if llm_name == "microsoft/deberta-v3-large":
            from transformers import DebertaV2Model
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(llm_name)
            model = DebertaV2Model.from_pretrained(llm_name)
            target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
        elif llm_name == "microsoft/deberta-v2-xlarge":
            from transformers import DebertaV2Model
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
            model = DebertaV2Model.from_pretrained("microsoft/deberta-v2-xlarge")
            target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
        elif llm_name == "microsoft/deberta-base":
            from transformers import DebertaModel
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
            model = DebertaModel.from_pretrained("microsoft/deberta-base")
            target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
        elif llm_name == "FacebookAI/roberta-base":
            from transformers import RobertaModel
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
            model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
            target_modules = ["query", "key", "value", "dense"]  # roberta的mask token为 <mask>
        elif llm_name == "google-bert/bert-base-uncased":
            from transformers import BertModel
            self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            model = BertModel.from_pretrained("google-bert/bert-base-uncased")
            target_modules = ["query", "key", "value", "dense"]
        else:
            raise ValueError("Invalid type of language model!")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none")
        self.model = get_peft_model(model, config)
        self.model.print_trainable_parameters()
    
    
    def forward(self, visual_feature: torch.Tensor, visual_mask: torch.Tensor, qas_word: list) -> torch.Tensor:
        device = visual_feature.device
        BS = visual_feature.size(0)
        
        language_input = self.tokenizer(qas_word, return_tensors='pt', padding="longest")
        language_embeddings = self.model.get_input_embeddings()(language_input.input_ids.to(device))

        if visual_mask is not None:
            feat_mask = torch.concat([visual_mask, language_input.attention_mask.to(device)], dim=1)
            ans_mask = torch.concat([torch.zeros_like(visual_mask), (language_input.input_ids == self.tokenizer.mask_token_id).bool().to(device)], dim=1)
        else:
            feat_mask = torch.concat([torch.ones(visual_feature.size(0), visual_feature.size(1)), language_input.attention_mask], dim=1).to(device)
            ans_mask = torch.concat([torch.zeros(visual_feature.size(0), visual_feature.size(1)), (language_input.input_ids == self.tokenizer.mask_token_id)], dim=1).to(device)
        feat = torch.concat([visual_feature, language_embeddings], dim=1)
        
        output = self.model(inputs_embeds=feat, attention_mask=feat_mask)
        output = (output.last_hidden_state * ans_mask.float().unsqueeze(-1)).sum(dim=1)
        
        return output