import torch
import torch.nn as nn

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
from transformers import AutoModel, AutoConfig

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
        self.criterion = nn.NLLLoss()
        
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
                    
    # our model: for sampling
    def embedding_matching_role(self, role_ids, embeddings):
        embeddings_turn = []
        start_idx = 0
        role_ids_cpu = role_ids.cpu().numpy()[0]

        for i in range(1, len(role_ids_cpu)):
            if role_ids_cpu[i] != role_ids_cpu[i-1]:
                turn = embeddings[:, start_idx:i, :].mean(dim=1)
                embeddings_turn.append(turn)
                start_idx = i

        turn = embeddings[:, start_idx:, :].mean(dim=1)
        embeddings_turn.append(turn)

        embeddings_turn = torch.cat(embeddings_turn, dim=0)
        return embeddings_turn
    
    def train_loss_with_sampling(self, pr, nr, p, n): # Positive와 Negative 샘플에 대한 임베딩 간 유사도를 계산하고, Contrastive Loss를 계산
        
        loss = []
        
        for i in range(10): # 하나의 대화단위로 접근
            
            # print("pr:", pr[i].shape) # torch.Size([1, 512])
            # print("nr:", nr[i].shape) # torch.Size([9, 512])
            # print("p:", p[i].shape) # torch.Size([1, 512, 768])
            # print("n:", n[i].shape) # torch.Size([9, 512, 768])
                
            if n[i].shape[0] == self.sample_nums -1: # negtive sample이 9개인 경우
                
                pos_turn = self.embedding_matching_role(pr[i], p[i])
                
                neg_turn = []
                for ni in n[i]:
                    n_turn = self.embedding_matching_role(nr[i], ni.unsqueeze(0))
                    neg_turn.append(n_turn)
                neg_turns = torch.cat(neg_turn, dim=0)
                
                # 샘플링 진행
                pos_sample = self.embeddingsampler.run(pos_turn)
                neg_samples = self.embeddingsampler.run(neg_turns)
                # print("pos sample:", pos_sample.shape)
                # print("neg samples:", neg_samples.shape)
                
                # loss 계산
                pos_cos_sim = self.cos(pos_sample.unsqueeze(1), pos_sample.unsqueeze(0)) / self.args.temperature
                pos_cos_sim = pos_cos_sim.fill_diagonal_(0)
                # print("pos_cos_sim: ", pos_cos_sim.mean())
                
                neg_cos_sim = self.cos(neg_samples.unsqueeze(1), neg_samples.unsqueeze(0)) / self.args.temperature
                neg_cos_sim = neg_cos_sim.fill_diagonal_(0)
                # print("neg_cos_sim: ", neg_cos_sim.mean())
                
                # criterion = nn.NLLLoss() 인 경우
                log_probs_pos = nn.LogSoftmax(dim=1)(pos_cos_sim)
                labels_pos = torch.arange(log_probs_pos.size(0)).long().to(pos_cos_sim.device) # device 설정 부분 주의
                pos_loss = self.criterion(log_probs_pos, labels_pos)

                
                log_probs_neg = nn.LogSoftmax(dim=1)(neg_cos_sim)
                labels_neg = torch.arange(log_probs_neg.size(0)).long().to(neg_cos_sim.device) # device 설정 부분 주의
                neg_loss = self.criterion(log_probs_neg, labels_neg)
                
                # # criterion = nn.CrossEntropyLoss() 인 경우
                # labels_pos = torch.arange(pos_cos_sim.size(0)).long().to(self.args.device) 
                # pos_loss = config['criterion'](pos_cos_sim, labels_pos)
                # labels_neg = torch.arange(neg_cos_sim.size(0)).long().to(self.args.device) 
                # pos_loss = config['criterion'](neg_cos_sim, labels_neg            
                
                # dial_loss = pos_loss / neg_loss
                dial_loss = pos_loss + neg_loss
                loss.append(dial_loss)
                
            else:
                pos_turn = self.embedding_matching_role(pr[i], p[i])
                # loss 계산
                pos_cos_sim = self.cos(pos_turn.unsqueeze(1), pos_turn.unsqueeze(0)) / self.args.temperature
                pos_cos_sim = pos_cos_sim.fill_diagonal_(0)
                # criterion = nn.NLLLoss() 인 경우
                log_probs_pos = nn.LogSoftmax(dim=1)(pos_cos_sim)
                labels_pos = torch.arange(log_probs_pos.size(0)).long().to(pos_cos_sim.device) # device 설정 부분 주의
                pos_loss = self.criterion(log_probs_pos, labels_pos)
                loss.append(pos_loss)
                
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
        # print("======self_output=======")
        # print(self_output.shape) # torch.Size([100, 512, 768])
        # print("======pooled_output=======") # torch.Size([100, 768])
        # print(pooled_output.shape)
        # print("======role_ids=======") # torch.Size([100, 768])
        # print(role_ids.shape)
        
        role_id = role_ids.view(10, -1, 512) # self.args.batch_size = 10, self.args.seq_len = 512
        positive_roleid = role_id[:, :1, :]
        negative_roleid = role_id[:, 1:, :] 
        
        _, _, hidden_size = self_output.size()
        output_embeddings = self_output.view(10, -1, 512, hidden_size)
        positive_embeddings = output_embeddings[:, :1, :, :]
        negative_embeddings = output_embeddings[:, 1:, :, :]
        # print('self_output:', self_output.shape)
        # print('output_embeddings:', output_embeddings.shape)
        # print('negative_embeddings:', negative_embeddings.shape)
        self_output = self_output * attention_mask.unsqueeze(-1)
        self_output = self.avg(self_output, attention_mask)
        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size)
        output = self_output[:, 0, :]
        
        our_loss = self.train_loss_with_sampling(positive_roleid, negative_roleid, positive_embeddings, negative_embeddings).to(output.device)

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
        return self.labels_data