import torch
import torch.nn as nn

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
from transformers import AutoModel, AutoConfig

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
        self.sample_nums = 9 # 수정: 데이터에 따라 수정 필요
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = BertAVG(eps=1e-6)
        self.logger = args.logger

    def set_finetune(self):
        """
        设置微调层数
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

    # ========================= our model: for stage loss ==========================
    def cal_stage_loss(self, qa_ids, turn_ids, attention_mask, self_output, labels):
        one_mask = torch.ones_like(qa_ids) # 모든 요소 1로 초기화
        zero_mask = torch.zeros_like(qa_ids) # 모든 요소 0으로 초기화
        q_mask = torch.where((qa_ids == 1) | (qa_ids == 2), one_mask, zero_mask) # qa_ids가 1, 2인 경우에 1, 그렇지 않으면 0 
        a_mask = torch.where((qa_ids == 0) | (qa_ids == 2), one_mask, zero_mask) # qa_ids가 0, 2인 경우에 1, 그렇지 않으면 0 
        n_mask = torch.where(qa_ids == 3, one_mask, zero_mask) # qa_ids가 3인 경우에 1, 그렇지 않으면 0
        # q_mask = torch.where(qa_ids == 1, one_mask, zero_mask) # qa_ids가 1인 경우에 1, 그렇지 않으면 0 
        # a_mask = torch.where(qa_ids == 0, one_mask, zero_mask) # qa_ids가 0인 경우에 1, 그렇지 않으면 0 
        # n_mask = torch.where(qa_ids == 2, one_mask, zero_mask) # qa_ids가 2인 경우에 1, 그렇지 않으면 0
        
        q_attention_mask = (attention_mask * q_mask)
        a_attention_mask = (attention_mask * a_mask)
        n_attention_mask = (attention_mask * n_mask)
        
        q_self_output = self_output * q_attention_mask.unsqueeze(-1) # torch.Size([100, 512, 768])
        a_self_output = self_output * a_attention_mask.unsqueeze(-1)
        n_self_output = self_output * n_attention_mask.unsqueeze(-1)
        
        q_w = torch.matmul(q_self_output, (a_self_output + n_self_output).transpose(-1, -2))
        a_w = torch.matmul(a_self_output, (q_self_output + n_self_output).transpose(-1, -2))
        n_w = torch.matmul(n_self_output, (q_self_output + a_self_output).transpose(-1, -2))
        
        if turn_ids is not None:
            view_turn_mask = turn_ids.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)
            view_turn_mask_transpose = view_turn_mask.transpose(2, 1)
            view_range_mask = torch.where(abs(view_turn_mask_transpose - view_turn_mask) <= self.args.max_turn_view_range,
                                          torch.ones_like(view_turn_mask),
                                          torch.zeros_like(view_turn_mask))
            filtered_q_w = q_w * view_range_mask
            filtered_a_w = a_w * view_range_mask
            filtered_n_w = n_w * view_range_mask
        
        filtered_w = filtered_q_w + filtered_a_w + filtered_n_w
        q_cross_output = torch.matmul(filtered_w, q_self_output)
        a_cross_output = torch.matmul(filtered_w, a_self_output)
        n_cross_output = torch.matmul(filtered_w, n_self_output)
        
        q_self_output = self.avg(q_self_output, q_attention_mask) # torch.Size([100, 768])
        q_cross_output = self.avg(q_cross_output, a_attention_mask + n_attention_mask)
        a_self_output = self.avg(a_self_output, a_attention_mask)
        a_cross_output = self.avg(a_cross_output, q_attention_mask + n_attention_mask)
        n_self_output = self.avg(n_self_output, n_attention_mask)
        n_cross_output = self.avg(n_cross_output, q_attention_mask + a_attention_mask)
        
        q_self_output = q_self_output.view(-1, self.sample_nums, self.config.hidden_size) # torch.Size([10, 10, 768])
        q_cross_output = q_cross_output.view(-1, self.sample_nums, self.config.hidden_size)
        a_self_output = a_self_output.view(-1, self.sample_nums, self.config.hidden_size)
        a_cross_output = a_cross_output.view(-1, self.sample_nums, self.config.hidden_size)
        n_self_output = n_self_output.view(-1, self.sample_nums, self.config.hidden_size)
        n_cross_output = n_cross_output.view(-1, self.sample_nums, self.config.hidden_size)
        
        q_output = q_self_output[:, 0, :]
        a_output = a_self_output[:, 0, :]
        n_output = n_self_output[:, 0, :]
        # q_contrastive_output = q_cross_output[:, 0, :]
        # a_contrastive_output = a_cross_output[:, 0, :]
        # n_contrastive_output = n_cross_output[:, 0, :]        
        
        logit_q = []
        logit_a = []
        logit_n = []
        for i in range(self.sample_nums):
            cos_q = self.calc_cos(q_self_output[:, i, :], q_cross_output[:, i, :])
            cos_a = self.calc_cos(a_self_output[:, i, :], a_cross_output[:, i, :])
            cos_n = self.calc_cos(n_self_output[:, i, :], n_cross_output[:, i, :])
            logit_a.append(cos_a)
            logit_q.append(cos_q)
            logit_n.append(cos_n)
        
        logit_a = torch.stack(logit_a, dim=1)
        logit_q = torch.stack(logit_q, dim=1)
        logit_n = torch.stack(logit_n, dim=1)

        loss_a = self.calc_loss(logit_a, labels)
        loss_q = self.calc_loss(logit_q, labels)
        loss_n = self.calc_loss(logit_n, labels)

        stage_loss = (loss_a + loss_q + loss_n)/3
        # stage_loss = loss_a + loss_q + loss_n
        # stage_loss = 0.4*loss_a + 0.4*loss_q + 0.2*loss_n
        return stage_loss, q_output, a_output, n_output
    # ==============================================================================
    
    
    def forward(self, data, strategy='mean_by_role', output_attention=False): 
        """
        前向传递过程
        """
        if len(data) == 8: # 수정: qa 추가
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, qa_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, qa_ids, labels, guids = data

        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])
        qa_ids = qa_ids.view(qa_ids.size()[0] * qa_ids.size()[1], qa_ids.size()[-1])
        # print("role_ids:", role_ids.shape) # torch.Size([100, 512])
        # print("qa_ids:", qa_ids.shape) # torch.Size([100, 512])

        self_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)
        stage_loss, q_output, a_output, n_output = self.cal_stage_loss(qa_ids, turn_ids, attention_mask, self_output, labels)
                
        self_output = self.avg(self_output, attention_mask)

        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size)

        output = self_output[:, 0, :]
        
        # print("q_output:", q_output.shape) # torch.Size([10, 768])
        # print("a_output:", a_output.shape) # torch.Size([10, 768])
        # print("n_output:", n_output.shape) # torch.Size([10, 768])
        # print("output:", output.shape) # torch.Size([10, 768])

        output_dict = {'loss': stage_loss,
                       'final_feature': output} # output 또는 q_output+a_output+n_output

        return output_dict

    def encoder(self, *x):
        """
        BERT编码过程
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
        """
        cos = torch.cosine_similarity(x, y, dim=1)
        # 수정: 코사인 유사도가 1인 경우는 0으로
        # mask = (cos == 1)
        # cos[mask] = 0
        # 수정: 코사인 유사도가 1인 경우는 nan으로 
        cos[(cos == 1)] = float('nan')
        cos = cos / self.args.temperature   # cos = cos / 2.0
        return cos

    def calc_loss(self, pred, labels):
        """
        计算损失函数
        """
        # pred = pred.float()
        loss = -torch.nanmean(self.log_softmax(pred) * labels) # 수정: nan값 제외 시, torch.nanmean
        return loss

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data