import torch
import torch.nn as nn
import torch.nn.functional as F

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
from transformers import AutoModel, AutoConfig

from itertools import groupby
import random

import config
from config import huggingface_mapper
from sampler import IdentitySampler, BaseSampler, GreedyCoresetSampler, ApproximateGreedyCoresetSampler

from transformers import AutoTokenizer, AutoConfig
from transformers import BertTokenizer, BertForMaskedLM, BertModel


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
            
            # PLATO tokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(config.huggingface_mapper[self.args.backbone])
            # self.tokenizer_config = PlatoConfig.from_json_file(self.args.config_file)
            
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
            
            
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
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = BertAVG(eps=1e-6)
        self.logger = args.logger
        
        self.cos = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss() # nn.NLLLoss()
        
        sampler_name = args.sampler 
        percentage = args.percentage
        device = args.device
        
        if sampler_name == 'identity':
            self.embeddingsampler = IdentitySampler()
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

    # ========================== our model: for sampling ==========================
    def embedding_matching_turn(self, role_ids, embeddings):
        '''
        role_ids를 이용해 embedding 결과를 turn 단위로 묶어서 결과를 리스트로 반환하는 함수
        우선 대화자를 speaker라고 변수 이름 붙였는데, 상의 필요
        '''
        turn_emb_ls = []

        # pr을 1차원 리스트로 변환
        r_ls = role_ids.squeeze().tolist()

        # 현재 발화자와 그 발화자의 연속 횟수를 파악하기 위한 변수
        current_speaker = r_ls[0]
        count = 1

        for i in range(1, len(r_ls)):
            if r_ls[i] == current_speaker:
                count += 1
            else:
                # 이전 발화자의 임베딩 결과를 새로운 리스트에 저장
                turn_emb_ls.append(embeddings[:, i-count:i, :].squeeze(0))
                # 새로운 발화자와 카운트 업데이트
                current_speaker = r_ls[i]
                count = 1

        # 마지막 발화자 처리
        turn_emb_ls.append(embeddings[:, len(r_ls)-count:len(r_ls), :].squeeze(0))
        
        return turn_emb_ls
    
    def random_sample_turns(self, pooled_emb, ratio=0.5):
        turns, _ = pooled_emb.shape
        num_sample = int(turns * ratio)
        
        # 랜덤 샘플링하여 선택된 턴의 인덱스
        sampled_idx = torch.randint(high=turns, size=(num_sample,))
        # 선택된 턴의 인덱스로부터 특정 턴만 선택하여 새로운 변수 생성
        sampled_turns = pooled_emb[sampled_idx]
        
        return sampled_idx, sampled_turns
    
    def cal_turn_loss(self, ar, pr, nr, a, p, n, labels):
        
        logits = []
        for i in range(len(ar)): # 하나의 대화단위로 접근, len(pr)는 전체 대화의 개수
            # print("pr[i]:", pr[i].shape) # torch.Size([1, 512])
            # print("nr[i]:", nr[i].shape) # torch.Size([9, 512])
            # print("p[i]:", p[i].shape) # torch.Size([1, 512, 768])
            # print("n[i]:", n[i].shape) # torch.Size([9, 512, 768])
                        
            # anchor 생성
            anc_emb_ls = self.embedding_matching_turn(ar[i].unsqueeze(0), a[i]) # turn 단위로 임베딩 결과 묶은 리스트 생성
            pooled_anc = [tensor.mean(dim=0, keepdim=True) for tensor in anc_emb_ls]
            pooled_anc = torch.stack(pooled_anc, dim=0).squeeze()
            sampled_idx, sampled_anc = self.random_sample_turns(pooled_anc, ratio=0.5)
            # print("pooled_anc:", pooled_anc.shape) # (turn개수, 768)
            # print("sampled_anc:", sampled_anc.shape) # (sample된 turn개수, 768)
            
            # positive pair 생성
            pos_emb_ls = self.embedding_matching_turn(pr[i].unsqueeze(0), p[i])
            pooled_pos = [tensor.mean(dim=0, keepdim=True) for tensor in pos_emb_ls]
            pooled_pos = torch.stack(pooled_pos, dim=0).squeeze()
            sampled_pos = pooled_pos[sampled_idx]
            # anchor & postive 코사인유사도의 평균 계산
            p_cos = self.calc_cos(sampled_anc, sampled_pos) # 턴 단위
            p_cos_avg = torch.mean(p_cos) # 대화 단위
            
            # negative pair 생성
            n_cos_avgs = []
            for nri, ni in zip(nr[i], n[i]):
                neg_emb_ls = self.embedding_matching_turn(nri.unsqueeze(0), ni.unsqueeze(0))
                pooled_neg = [tensor.mean(dim=0, keepdim=True) for tensor in neg_emb_ls]
                pooled_neg = torch.stack(pooled_neg, dim=0).squeeze()
                sampled_neg = pooled_neg[sampled_idx]
                # anchor & postive 코사인유사도의 평균 계산
                n_cos = self.calc_cos(sampled_anc, sampled_neg)
                n_cos_avg = torch.mean(n_cos)
                n_cos_avgs.append(n_cos_avg)
            n_cos_avgs = torch.stack(n_cos_avgs)
            
            # loss 계산
            logit = torch.cat((p_cos_avg.unsqueeze(0), n_cos_avgs), dim=0)
            logits.append(logit)
        turn_loss = self.calc_loss(torch.stack(logits), labels)
        return turn_loss
    # ==============================================================================
    
    def forward(self, data):
        """
        前向传递过程: "forward" 메서드는 모델의 순전파(forward propagation) 과정을 수행
        입력 데이터를 받아 BERT 또는 Plato 모델을 통해 인코딩
        마스킹된 평균 임베딩을 계산
        손실을 계산하고 반환
        """
        if len(data) == 7:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels, guids = data
        
        # input_ids shape: torch.Size([10, 11, 512])
            
        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])

        self_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)
        # 우리 모델의 loss 적용
        # print("+++++++++output before our loss++++++++")
        # print("self_output:", self_output.shape) # torch.Size([110, 512, 768]) / torch.Size([50, 768])
        # print("pooled_output:", pooled_output.shape) # torch.Size([110, 768]) / torch.Size([50, 768])
        # print("role_ids:", role_ids.shape) # torch.Size([100, 768]) / torch.Size([50, 768])
        
        role_id = role_ids.view(-1, self.sample_nums, self.args.max_seq_length)
        anchor_roleid = role_id[:, 0, :].unsqueeze(1) # torch.Size([10, 1, 512])
        positive_roleid = role_id[:, 1, :].unsqueeze(1) # torch.Size([10, 1, 512])
        negative_roleid = role_id[:, 2:, :] #  torch.Size([10, 9, 512])
        
        output_embeddings = self_output.view(-1, self.sample_nums, self.args.max_seq_length, self.config.hidden_size)  # torch.Size([10, 11, 512, 768])
        pooled_output_embeddings = torch.mean(output_embeddings, dim=2)  # torch.Size([10, 11, 768])
        
        anchor_embedding = output_embeddings[:, 0, :, :].unsqueeze(1)
        positive_embedding = output_embeddings[:, 1, :, :].unsqueeze(1)
        negative_embeddings = output_embeddings[:, 2:, :, :]
        # print('output_embeddings:', output_embeddings.shape)  # torch.Size([10, 11, 512, 768])
        # print('anchor_embedding:', anchor_embedding.shape)  # torch.Size([10, 1, 512, 768])
        # print('positive_embedding:', positive_embedding.shape)  # torch.Size([10, 1, 512, 768])
        # print('negative_embeddings:', negative_embeddings.shape)  # torch.Size([10, 9, 512, 768])  
        
        # turn loss 계산 
        turn_loss = self.cal_turn_loss(anchor_roleid, positive_roleid, negative_roleid,
                                       anchor_embedding, positive_embedding, negative_embeddings, labels)  

        # dialogue representation
        self_output = self_output * attention_mask.unsqueeze(-1)
        self_output = self.avg(self_output, attention_mask)
        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)  # torch.Size([10, 11, 768])
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size)
        
        output = self_output[:, 0, :]
        # print("====================output shape==================")
        # print("self_output: ", self_output.shape) 
        # print("output: ", output.shape)
        
        logits = []
        for i in range(1, self.sample_nums):
            # print(output_embeddings[:, 0, :].shape)

            cos_output = self.calc_cos(pooled_output_embeddings[:, 0, :], pooled_output_embeddings[:, i, :])
            # print(cos_output)
            logits.append(cos_output)

        logits = torch.stack(logits, dim=1)

        dial_loss = self.calc_loss(logits, labels)
        
        # print("=====================Our Loss===================")
        # print(dial_loss, turn_loss, 0.8*dial_loss + 0.2*turn_loss) # turn_loss가 dial_loss보다 0.01~0.03 정도 큼
        output_dict = {'loss': 0.5*dial_loss + 0.5*turn_loss,
                       'final_feature': output 
                       }
        return output_dict

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
        return self.labels_data