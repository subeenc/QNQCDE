import torch
import torch.nn as nn

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
import config
import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import PreTrainedTokenizerBase
from transformers import BertConfig
from modeling_bert_diff import BertModel, BertLMPredictionHead

from itertools import groupby
import random
from typing import Optional, Union, List, Dict, Tuple

from sampler import IdentitySampler, BaseSampler, GreedyCoresetSampler, ApproximateGreedyCoresetSampler
from config import huggingface_mapper


class BertAVG(nn.Module):
    """
    对BERT输出的embedding求masked average
    """
    def __init__(self, eps=1e-12):
        super(BertAVG, self).__init__()
        self.eps = eps

    def forward(self, hidden_states, attention_mask):
        mul_mask = lambda x, m: x * torch.unsqueeze(m, dim=-1)
        reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (torch.sum(m, dim=1, keepdims=True) + self.eps)

        avg_output = reduce_mean(hidden_states, attention_mask)
        return avg_output

    def equal_forward(self, hidden_states, attention_mask):
        mul_mask = hidden_states * attention_mask.unsqueeze(-1)
        avg_output = torch.sum(mul_mask, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + self.eps)
        return avg_output


class Dial2vec(nn.Module):
    """
    Dial2vec模型
    """
    def __init__(self, args):
        super(Dial2vec, self).__init__()
        self.args = args
        self.result = {}
        num_labels, total_steps, self.sep_token_id = args.num_labels, args.total_steps, args.sep_token_id

        if args.backbone.lower() == 'plato':
            self.config = PlatoConfig.from_json_file(self.args.config_file)
            self.bert = PlatoModel(self.config)
        elif args.backbone.lower() in ['bert', 'roberta', 'todbert', 't5', 'blender', 'unsup_simcse', 'sup_simcse']:
            self.config = AutoConfig.from_pretrained(huggingface_mapper[args.backbone.lower()])
            self.bert = AutoModel.from_pretrained(huggingface_mapper[args.backbone.lower()])
            # special cases
            if args.backbone.lower() in ['t5']:
                self.config.hidden_dropout_prob = self.config.dropout_rate
                self.bert = self.bert.encoder
            elif args.backbone.lower() in ['blender']:
                self.config.hidden_dropout_prob = self.config.dropout
                self.bert = self.bert.encoder
        else:
            raise NameError('Unknown backbone model: [%s]' % args.backbone)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.labels_data = None
        self.sample_nums = 11
        # ============================= Our Model: 확인 필요 ============================
        self.tokenizer = AutoTokenizer.from_pretrained(config.huggingface_mapper[self.args.backbone])
        self.generator = transformers.DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')# if cls.model_args.generator_name is None else transformers.AutoModelForMaskedLM.from_pretrained(cls.model_args.generator_name)
        self.mlm_probability = 0.15 # 조절 필요
        # self.bert = BertModel(config, add_pooling_layer=False)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.discriminator = BertModel(bert_config, add_pooling_layer=False)
        self.lm_head = BertLMPredictionHead(bert_config)
        self.electra_head = torch.nn.Linear(768, 2)
        self.electra_acc = 0.0
        self.electra_rep_acc = 0.0
        self.electra_fix_acc = 0.0
        # ===============================================================================
        self.avg = BertAVG(eps=1e-6)
        self.logger = args.logger
        
        self.cos = nn.CosineSimilarity(dim=1)
        self.criterion = nn.CrossEntropyLoss() # nn.NLLLoss(), nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
        
        sampler_name = args.sampler 
        percentage = args.percentage
        device = args.device
        
        if sampler_name == 'identity':
            self.embeddingsampler = IdentitySampler()
        elif sampler_name == 'base':
            self.embeddingsampler = GreedyCoresetSampler(percentage)
        elif sampler_name == 'greedy_coreset':
            self.embeddingsampler = GreedyCoresetSampler(percentage, device)
        elif sampler_name == 'approx_greedy_coreset':
            self.embeddingsampler = ApproximateGreedyCoresetSampler(percentage, device)
        else:
            raise ValueError(f"Unsupported sampler: {sampler_name}")

    def set_finetune(self):
        """
        设置微调层数: "set_finetune" 메서드는 모델의 미세 조정을 위해 필요한 파라미터를 설정. 미세 조정할 레이어를 선택한다고 보면 됨 (레이어 6개는 고정)
        """
        self.logger.debug("******************")
        name_list = ["11", "10", "9", "8", "7", "6"]
        # name_list = ["11",'10','9']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for s in name_list:
                if s in name:
                    self.logger.debug(name)
                    param.requires_grad = True
     
    # ============================= Our Model: Masking =============================
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK]): 마스크된 인덱스의 80%는 [MASK] 토큰으로 대체
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word: 마스크된 인덱스의 10%는 랜덤한 단어로 대체
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged: 나머지 10%는 변경되지 않고 유지
        return inputs, labels
    # ==============================================================================
     
    # ========================== Our Model: Calculate Loss =========================
    def train_loss(self, p, n): # Positive와 Negative 샘플에 대한 임베딩 간 유사도를 계산하고, Contrastive Loss를 계산
        
        loss = []
        
        for i in range(len(p)): # 하나의 대화단위로 접근, len(pr)는 전체 대화의 개수
            # print("p[i]:", p[i].shape) # torch.Size([2, 768])
            # print("n[i]:", n[i].shape) # torch.Size([9, 768])
            
            # anchor와 postive pair 간의 코사인 유사도의 평균 계산
            p_cos = self.cos(p[i][0].unsqueeze(0), p[i][1].unsqueeze(0)) / self.args.temperature
            # p_cos = p_cos.fill_diagonal_(0)
            # p_cos_avg = torch.mean(p_cos)
            
            # anchor와 negative pairs 간의 코사인 유사도의 평균 계산
            n_cos = self.cos(p[i][0].unsqueeze(0), n[i]) / self.args.temperature
            # n_cos = n_cos.fill_diagonal_(0)
            
            # loss 계산
            # print("p_cos:", p_cos)
            # print("n_cos:", n_cos)
            p_n_cos = torch.cat((p_cos, n_cos), dim=0)
            p_n_label = torch.tensor([1., 0., 0., 0., 0., 0., 0. ,0. ,0., 0.]).to(p_n_cos.device)
            dial_loss = self.criterion(p_n_cos, p_n_label) # criterion = nn.CrossEntropyLoss(), 가중치 적용 필요, BCE로 변경 필요
            loss.append(dial_loss)
        # print("======= loss: =======")
        # print(torch.stack(loss).sum() / len(loss))
        return torch.stack(loss).sum() / len(loss)
    
    def forward(self, data):
        """
        前向传递过程: "forward" 메서드는 모델의 순전파(forward propagation) 과정을 수행
        입력 데이터를 받아 BERT 또는 Plato 모델을 통해 인코딩
        마스킹된 평균 임베딩을 계산
        손실을 계산하고 반환
        """
        if len(data) == 7:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels = data # input_ids: torch.Size([10, 10, 512])
        else:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels, guids = data
        
        # ========== positive에 dropout을 적용하기 위해 동일한 postiive 대화 2개 생성 ==========
        input_ids = torch.cat([input_ids[:, :1, :], input_ids], dim=1) # input_ids: torch.Size([10, 11, 512])
        attention_mask = torch.cat([attention_mask[:, :1, :], attention_mask], dim=1)
        token_type_ids = torch.cat([token_type_ids[:, :1, :], token_type_ids], dim=1)
        role_ids = torch.cat([role_ids[:, :1, :], role_ids], dim=1)
        turn_ids = torch.cat([turn_ids[:, :1, :], turn_ids], dim=1)
        position_ids = torch.cat([position_ids[:, :1, :], position_ids], dim=1)
        # =====================================================================================
        
        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1]) # torch.Size([10*11, 512])
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])
        '''
        이상하다고 느껴지면, diffcse/train.py에서 아래 코드 검색해서 다시 생각해보기
        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}
        '''

        # Get raw embeddings & Pooling
        self_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)
        
        self_output = self_output * attention_mask.unsqueeze(-1)
        self_output = self.avg(self_output, attention_mask)
        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size) # (batch, sample_nums=11, hidden)
        output = self_output[:, 0, :]
        
        # ==================================== DiffCSE 적용 ===================================
        # Produce MLM augmentations and perform conditional ELECTRA using the discriminator
        # Masking
        mlm_input_ids, mlm_labels = self.mask_tokens(input_ids) # torch.Size([110, 512]), # torch.Size([110, 512])
        with torch.no_grad():
            g_pred = self.generator(mlm_input_ids, attention_mask)[0].argmax(-1) # torch.Size([110, 512])
        g_pred[:, 0] = 101 # cls_token(모든 문장의 첫 번째 토큰을 [CLS] 토큰으로 대체하는 것) -> diffcse에서 bert: 101, roberta: 0 으로 정의
        '''
        이 부분이 이상하다면, DiffCSE에선, g_pred[:, 0]가 모두 1012로 할당되는데, 우리 코드에서는 다양하게 할당됨...
        그리고 CLS TOKEN을 사용하지 않는 방안으로 다시 생각해야할 듯
        '''
        replaced = (g_pred != input_ids) * attention_mask # torch.Size([110, 512])
        e_inputs = g_pred * attention_mask # torch.Size([110, 512])
        
        mlm_outputs = self.discriminator(
            e_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True
            # cls_input=pooler_output.view((-1, pooler_output.size(-1))),
        )
        # print("mlm_outputs:", mlm_outputs.last_hidden_state.shape) # torch.Size([110, 512, 768])
        
        positive_embeddings = self_output[:, :2, :] # torch.Size([10, 2, 768])
        negative_embeddings = self_output[:, 2:, :] # torch.Size([10, 9, 768])
        
        loss = self.train_loss(positive_embeddings, negative_embeddings).to(output.device)
        
        # Calculate loss for condiif mlm_outputs is not None and mlm_labels is not None:
        # mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        e_labels = replaced.view(-1, replaced.size(-1))
        # prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
        prediction_scores = self.electra_head(mlm_outputs.last_hidden_state)
        rep = (e_labels == 1) * attention_mask
        fix = (e_labels == 0) * attention_mask
        prediction = prediction_scores.argmax(-1)
        self.electra_rep_acc = float((prediction*rep).sum()/rep.sum())
        self.electra_fix_acc = float(1.0 - (prediction*fix).sum()/fix.sum())
        self.electra_acc = float(((prediction == e_labels) * attention_mask).sum()/attention_mask.sum())
        # masked_lm_loss = self.criterion(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        masked_lm_loss = self.criterion(prediction_scores.view(-1, 2), e_labels.view(-1))
        loss = loss + self.args.lambda_weight * masked_lm_loss
        # print("loss:", loss)
        output_dict = {'loss': loss,
                       'final_feature': output 
                       }
        return output_dict
    # =====================================================================================

    def encoder(self, *x):
        """
        BERT编码过程: "encoder" 함수는 BERT 모델에 입력 데이터를 전달하여 인코딩하는 과정을 담당
        입력 데이터를 BERT 또는 Plato와 같은 백본 모델에 전달하여 인코딩하는 메서드
        """
        input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids = x     # 每个都是[batch_size * num_turn, hidden_size]
        if self.args.backbone in ['bert', 'roberta', 'todbert', 'unsup_simcse', 'sup_simcse']:
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               output_hidden_states=True,
                               return_dict=True)
        elif self.args.backbone in ['t5', 'blender']:
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True,
                               return_dict=True)
            output['pooler_output'] = output['last_hidden_state']   # Notice: 为了实现便利，此处赋值一个tensor占位，但实际上不影响，因为没用到pooler output进行计算。
        elif self.args.backbone in ['plato']:
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               turn_ids=turn_ids,
                               role_ids=role_ids,
                               return_dict=True)
        else:
            raise ValueError('Unknown backbone name: [%s]' % self.args.backbone)

        all_output = output['hidden_states']
        pooler_output = output['pooler_output']
        return all_output[-1], pooler_output

    def calc_cos(self, x, y):
        """
        计算cosine相似度
        두 벡터 간의 코사인 유사도를 계산하는 메서드
        """
        cos = torch.cosine_similarity(x, y, dim=1)
        cos = cos / self.args.temperature   # cos = cos / 2.0
        return cos

    def calc_loss(self, pred, labels):
        """
        计算损失函数
        모델의 출력과 실제 레이블 간의 손실을 계산하는 메서드
        손실 함수로는 로그 소프트맥스 교차 엔트로피를 사용
        """
        # pred = pred.float()
        loss = -torch.mean(self.log_softmax(pred) * labels)
        return loss

    def get_result(self):
        """# 모델의 결과를 반환하는 메서드"""
        return self.result

    def get_labels_data(self):
        """레이블 데이터를 반환하는 메서드"""
        return self.labels_data#