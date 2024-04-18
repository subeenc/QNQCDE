import torch
import torch.nn as nn

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
from transformers import AutoModel, AutoConfig

from itertools import groupby
import random

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
        self.sample_nums = 10
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = BertAVG(eps=1e-6)
        self.logger = args.logger
        
        self.cos = nn.CosineSimilarity(dim=2)
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
    
    def find_most_frequent_speaker(self, role_ids):
        '''
        가장 많이 등장한 대화자 찾고, turn 단위로 idx 새로 생성하는 함수
        '''
        r_ls = role_ids.squeeze().tolist()
        speaker_counts = {}
        turn_idx_ls = []

        # 각 대화자의 등장 횟수 카운트
        for speaker, _ in groupby(r_ls):
            turn_idx_ls.append(speaker)
            if speaker in speaker_counts:
                speaker_counts[speaker] += 1
            else:
                speaker_counts[speaker] = 1
                
        most_frequent_speaker = max(speaker_counts, key=speaker_counts.get)
        most_frequent_count = speaker_counts[most_frequent_speaker]

        # print(f"가장 많이 등장한 대화자: 대화자 {most_frequent_speaker}, 등장 횟수: {most_frequent_count}")
        return turn_idx_ls, most_frequent_speaker
    
    def set_anchor(self, turn_idx_ls, turn_emb_ls, most_frequent_speaker): # window
        '''
        # anchor를 정하는 함수
        '''
        # 기준이 되는 대화자(=most_frequent_speaker)의 인덱스 중 1개 랜덤 샘플링
        speaker_turns = [i for i, speaker in enumerate(turn_idx_ls) if speaker == most_frequent_speaker]
        
        valid_indices = []
        anchor_turn_idx = None
        
        # window 정의
        # window = len(turn_idx_ls) // 4
        window = 2
        
        # turn_idx_ls가 window * 2 + 1 보다 작을 경우 샘플링 없이 절반을 anchor를 사용 <- 상의 필요
        if len(turn_idx_ls) < window * 2 + 1:
            anchor = turn_emb_ls[0:len(turn_emb_ls)//2]
        
        else:
            # window 사이즈만큼 앞뒤 인덱스를 제외한 범위에서 랜덤하게 하나의 인덱스를 선택
            valid_indices = [idx for idx in speaker_turns if window <= idx < len(turn_idx_ls) - window]
            anchor_turn_idx = random.choice(valid_indices)
            
            # window 사이즈를 고려하여 anchor에 포함될 turn의 인덱스 범위를 계산
            start_idx = anchor_turn_idx - window
            end_idx = anchor_turn_idx + window
            
            # 계산된 범위에 해당하는 turn_emb_ls의 값을 가져와서 앵커로 설정
            anchor = turn_emb_ls[start_idx:end_idx+1]
            
        # anchor 리스트의 각 요소를 [1, 768]이 되도록 mean pooling
        pooled_anchor = [tensor.mean(dim=0, keepdim=True) for tensor in anchor]
        # mean pooling 결과를 stack: [len(anchor), 768]
        pooled_anchor = torch.stack(pooled_anchor, dim=0).squeeze()
        # 텐서의 크기가 [768]일 경우만 [1, 768]으로 변경
        if pooled_anchor.dim() == 1:
            pooled_anchor = pooled_anchor.unsqueeze(0)
        
        return valid_indices, anchor_turn_idx, anchor, pooled_anchor, window
    
    def generate_pairs(self, valid_indices, anchor_turn_idx, anchor, turn_emb_ls, window):
        '''
        pair, hard pair를 생성하는 함수
        pos를 기준으로 변수를 정의한 것으로, pair의 경우 anchor에 stride를 적용하고, hard pair는 anchor_turn_idx에서 가장 거리가 먼 turn(동일 대화자)을 이용
        '''
        
        # stride 정의: anchor 길이 // 2
        stride = len(anchor) // 2
        
        # turn_emb_ls가 window * 2 + 1 보다 작을 경우
        if anchor_turn_idx is None: 
            pair_start_idx , hard_start_idx = len(turn_emb_ls)//2, len(turn_emb_ls)//2
            pair_end_idx, hard_end_idx = len(turn_emb_ls) - 1, len(turn_emb_ls) - 1
        
        else:
            # Pair
            pair_start_idx = (anchor_turn_idx + window) - stride + 1 # anchor_turn_idx + window: anchor의 end_idx
            pair_end_idx = pair_start_idx + len(anchor) - 1
            
            # Hard Pair: anchor_turn_idx에서 가장 거리가 먼 turn (동일 대화자)
            dist_to_first = abs(anchor_turn_idx - valid_indices[0])
            dist_to_last = abs(anchor_turn_idx - valid_indices[-1])
            
            # 더 큰 거리를 가진 인덱스 선택
            if dist_to_first > dist_to_last:
                farther_index = valid_indices[0]
            else:
                farther_index = valid_indices[-1]
            
            hard_start_idx = farther_index - window
            hard_end_idx = farther_index + window
        
        pair = turn_emb_ls[pair_start_idx:pair_end_idx+1]
        hard_pair = turn_emb_ls[hard_start_idx:hard_end_idx+1]
        
        # pair, hard_pair 리스트의 각 요소를 [1, 768]이 되도록 mean pooling
        # print("============ pair:", len(pair), pair[0].device)
        # print("hard_start_idx, hard_end_idx:", hard_start_idx, hard_end_idx)
        # print("============ hard_pair:", len(hard_pair), hard_pair[0].device)
        pooled_pair = [tensor.mean(dim=0, keepdim=True) for tensor in pair]
        # print("============ pooled_pair:", len(pooled_pair), pooled_pair[0].device)
        pooled_hard_pair = [tensor.mean(dim=0, keepdim=True) for tensor in hard_pair]
        
        # mean pooling 결과를 stack: [len(pooled_pair), 768], [len(pooled_hard_pair), 768]
        pooled_pair = torch.stack(pooled_pair, dim=0).squeeze()
        pooled_hard_pair = torch.stack(pooled_hard_pair, dim=0).squeeze()
        
        # 텐서의 크기가 [768]일 경우만 [1, 768]으로 변경
        if pooled_pair.dim() == 1:
            pooled_pair = pooled_pair.unsqueeze(0)
        if pooled_hard_pair.dim() == 1:
            pooled_hard_pair = pooled_hard_pair.unsqueeze(0)
        return pair, hard_pair, pooled_pair, pooled_hard_pair
     # ==============================================================================
     
     # ================ our model: generate pairs and calculate loss ================
    def train_loss_with_sampling(self, pr, nr, p, n): # Positive와 Negative 샘플에 대한 임베딩 간 유사도를 계산하고, Contrastive Loss를 계산
        
        loss = []
        
        for i in range(len(pr)): # 하나의 대화단위로 접근, len(pr)는 전체 대화의 개수
            # print("pr[i]:", pr[i].shape) # torch.Size([1, 512])
            # print("nr[i]:", nr[i].shape) # torch.Size([9, 512])
            # print("p[i]:", p[i].shape) # torch.Size([1, 512, 768])
            # print("n[i]:", n[i].shape) # torch.Size([9, 512, 768])
                            
            # turn 단위로 임베딩 결과 묶은 리스트 생성
            pos_emb_ls = self.embedding_matching_turn(pr[i], p[i])
            # 가장 많이 등장한 대화자 찾고, turn 단위로 idx 재정의
            turn_idx_ls, most_frequent_speaker = self.find_most_frequent_speaker(pr[i])
            # anchor 생성
            valid_indices, anchor_turn_idx, anchor, pooled_anchor, window = self.set_anchor(turn_idx_ls, pos_emb_ls, most_frequent_speaker)
            # postive pair, hard postive pair 생성
            _, _, pooled_pos, pooled_hard_pos = self.generate_pairs(valid_indices, anchor_turn_idx, anchor, pos_emb_ls, window)
            
            # anchor & postive, anchor & hard postive pair 코사인유사도의 평균 계산
            p_cos = self.cos(pooled_anchor.unsqueeze(1), pooled_pos.unsqueeze(0)) / self.args.temperature
            # p_cos = p_cos.fill_diagonal_(0)
            p_cos_avg = torch.mean(p_cos)
            hp_cos = self.cos(pooled_anchor.unsqueeze(1), pooled_hard_pos.unsqueeze(0)) / self.args.temperature
            # hp_cos = hp_cos.fill_diagonal_(0)
            hp_cos_avg = torch.mean(hp_cos)
            
            n_cos_avgs = []

            for nri, ni in zip(nr[i], n[i]):
                # print("nr[i].unsqueeze(0):", nri.unsqueeze(0).shape) # torch.Size([1, 512])
                # print("ni.unsqueeze(0):", ni.unsqueeze(0).shape) # torch.Size([1, 512, 768])
                neg_emb_ls = self.embedding_matching_turn(nri.unsqueeze(0), ni.unsqueeze(0))
                # negative pair 생성
                _, _, _, pooled_neg = self.generate_pairs(valid_indices, anchor_turn_idx, anchor, neg_emb_ls, window)
                # anchor & negative 코사인유사도의 평균 계산
                n_cos = self.cos(pooled_anchor.unsqueeze(1), pooled_neg.unsqueeze(0)) / self.args.temperature
                # n_cos = n_cos.fill_diagonal_(0)
                n_cos_avg = torch.mean(n_cos)
                n_cos_avgs.append(n_cos_avg)
            n_cos_avgs = torch.stack(n_cos_avgs)
            
            # loss 계산
            p_n_cos = torch.cat((p_cos_avg.unsqueeze(0), hp_cos_avg.unsqueeze(0), n_cos_avgs), dim=0)
            p_n_label = torch.tensor([1., 1., 0., 0., 0., 0., 0., 0. ,0. ,0., 0.]).to(p_n_cos.device)
            # print("######### POS, HARD-POS, NEG X 9", p_n_cos, '##########')
            # p_n_label = torch.tensor([1., 1.])
            # zeros_tensor = torch.zeros(n[i].shape[0])
            # p_n_label = torch.cat((p_n_label, zeros_tensor), dim=0).to(p_n_cos.device)
            dial_loss = self.criterion(p_n_cos, p_n_label) # criterion = nn.CrossEntropyLoss(), 가중치 적용 필요, BCE로 변경 필요
            loss.append(dial_loss)
            
                
        # print("======= loss: =======")
        # print(torch.stack(loss).sum() / len(loss))

        return torch.stack(loss).sum() / len(loss)
    
    def train_loss_with_coreset_sampling(self, pr, nr, p, n):
        
        loss = []
        
        for i in range(len(pr)): # 하나의 대화단위로 접근, len(pr)는 전체 대화의 개수
            # print("pr[i]:", pr[i].shape) # torch.Size([1, 512])
            # print("nr[i]:", nr[i].shape) # torch.Size([9, 512])
            # print("p[i]:", p[i].shape) # torch.Size([1, 512, 768])
            # print("n[i]:", n[i].shape) # torch.Size([9, 512, 768])
                            
            # turn 단위로 임베딩 결과 묶은 리스트 생성
            pos_emb_ls = self.embedding_matching_turn(pr[i], p[i])
            # 가장 많이 등장한 대화자 찾고, turn 단위로 idx 재정의
            turn_idx_ls, most_frequent_speaker = self.find_most_frequent_speaker(pr[i])
            # anchor 생성
            valid_indices, anchor_turn_idx, anchor, pooled_anchor, window = self.set_anchor(turn_idx_ls, pos_emb_ls, most_frequent_speaker)
            # postive pair 생성
            pooled_pos = [tensor.mean(dim=0, keepdim=True) for tensor in pos_emb_ls]
            pooled_pos = torch.stack(pooled_pos, dim=0).squeeze()
            pos_samples = self.embeddingsampler.run(pooled_pos).to(pooled_anchor.device)
            # anchor & postive 코사인유사도의 평균 계산
            p_cos = self.cos(pooled_anchor.unsqueeze(1), pos_samples.unsqueeze(0)) / self.args.temperature
            p_cos_avg = torch.mean(p_cos)
            
            n_cos_avgs = []

            for nri, ni in zip(nr[i], n[i]):
                # print("nr[i].unsqueeze(0):", nri.unsqueeze(0).shape) # torch.Size([1, 512])
                # print("ni.unsqueeze(0):", ni.unsqueeze(0).shape) # torch.Size([1, 512, 768])
                neg_emb_ls = self.embedding_matching_turn(nri.unsqueeze(0), ni.unsqueeze(0))
                # negative pair 생성
                pooled_neg = [tensor.mean(dim=0, keepdim=True) for tensor in neg_emb_ls]
                pooled_neg = torch.stack(pooled_neg, dim=0).squeeze()
                neg_samples = self.embeddingsampler.run(pooled_neg).to(pooled_anchor.device)
                # anchor & negative 코사인유사도의 평균 계산
                n_cos = self.cos(pooled_anchor.unsqueeze(1), neg_samples.unsqueeze(0)) / self.args.temperature
                # n_cos = n_cos.fill_diagonal_(0)
                n_cos_avg = torch.mean(n_cos)
                n_cos_avgs.append(n_cos_avg)
            n_cos_avgs = torch.stack(n_cos_avgs)
            
            # loss 계산
            p_n_cos = torch.cat((p_cos_avg.unsqueeze(0), n_cos_avgs), dim=0)
            p_n_label = torch.tensor([1., 0., 0., 0., 0., 0., 0. ,0. ,0., 0.]).to(p_n_cos.device)
            # print("######### POS, HARD-POS, NEG X 9", p_n_cos, '##########')
            dial_loss = self.criterion(p_n_cos, p_n_label) # criterion = nn.CrossEntropyLoss(), 가중치 적용 필요, BCE로 변경 필요
            loss.append(dial_loss)
                
        # print("======= loss: =======")
        # print(torch.stack(loss).sum() / len(loss))

        return torch.stack(loss).sum() / len(loss)
    # ===========================================================================
    
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

        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])


        self_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)
        
        # 우리 모델의 loss 적용
        # print("+++++++++output before ourmodel++++++++")
        # print("self_output:", self_output.shape) # torch.Size([100, 512, 768])
        # print("pooled_output:", pooled_output.shape) # torch.Size([100, 768])
        # print("role_ids:", role_ids.shape) # torch.Size([100, 768])
        
        role_id = role_ids.view(-1, self.sample_nums, self.args.max_seq_length)
        positive_roleid = role_id[:, :1, :]
        negative_roleid = role_id[:, 1:, :] 
        # print("role_ids:", role_ids.shape)
        # print("positive_roleid:", positive_roleid.shape)
        # print("negative_roleid:", negative_roleid.shape)
        
        output_embeddings = self_output.view(-1, self.sample_nums, self.args.max_seq_length, self.config.hidden_size)
        positive_embeddings = output_embeddings[:, :1, :, :]
        negative_embeddings = output_embeddings[:, 1:, :, :]
        # print('output_embeddings:', output_embeddings.shape)
        # print('positive_embeddings:', positive_embeddings.shape)
        # print('negative_embeddings:', negative_embeddings.shape)
        
        self_output = self_output * attention_mask.unsqueeze(-1)
        self_output = self.avg(self_output, attention_mask)
        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size)
        output = self_output[:, 0, :]
        
        # our_loss = self.train_loss_with_sampling(positive_roleid, negative_roleid,
        #                                          positive_embeddings, negative_embeddings).to(output.device) # self.args.window

        our_loss = self.train_loss_with_coreset_sampling(positive_roleid, negative_roleid,
                                                 positive_embeddings, negative_embeddings).to(output.device)
        output_dict = {'loss': our_loss,
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
        return self.labels_data#