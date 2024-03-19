import torch
from torch import nn


class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert # bert는 사전에 학습된 BERT 모델

    def forward(self, inputs, mode):

        if mode=='train':

            """positive_attention_mask = self.gen_attention_mask(inputs['positive']['input_ids'],
                                                              inputs['positive']['input_mask'],
                                                              inputs['positve']['segment_ids'],
                                                              inputs['positve']['role_ids'],
                                                              inputs['positve']['turn_ids'],
                                                              inputs['positve']['position_ids'])

            negative_attention_mask = self.gen_attention_mask(inputs['negative']['input_ids'],
                                                              inputs['negative']['input_mask'],
                                                              inputs['negative']['segment_ids'],
                                                              inputs['negative']['role_ids'],
                                                              inputs['negative']['turn_ids'],
                                                              inputs['negative']['position_ids'])"""

            # print('===========bert_forward_input_check===========')
            # print(inputs['positive']['input_ids'])
            # print(inputs['positive']['role_ids'])
            # print(inputs['positive']['turn_ids'])
            # print((inputs['positive']['input_ids'].view(4, -1)).shape)
            # print(inputs['positive']['role_ids'].shape)
            # print(inputs['positive']['turn_ids'].shape)
            
            positive_output = self.bert(input_ids=inputs['positive']['input_ids'].view(4, -1),
                                           token_type_ids=inputs['positive']['role_ids'].view(4, -1),
                                           attention_mask=inputs['positive']['turn_ids'].view(4, -1))
            
            positive_last_hidden_state = positive_output['last_hidden_state']
            positive_lhs_output = positive_last_hidden_state.view(4, -1, 16, 768)
            positive_pooler = positive_output['pooler_output'] # cls token
            
            negative_output = self.bert(input_ids=inputs['negative']['input_ids'].view(4, -1),
                                           token_type_ids=inputs['negative']['role_ids'].view(4, -1),
                                           attention_mask=inputs['negative']['turn_ids'].view(4, -1))
            
            negative_last_hidden_state = negative_output['last_hidden_state']
            negative_lhs_output = negative_last_hidden_state.view(4, -1, 16, 768)
            negative_pooler = negative_output['pooler_output']
            
            # print("========== positive_output", positive_output)
            # print("========== positive_last_hidden_state", positive_last_hidden_state, positive_last_hidden_state.shape) # torch.Size([4, 256, 768])
            # print("========== positive_pooler_tmp", positive_pooler_tmp, positive_pooler_tmp.shape) # torch.Size([4, 768])
            
            return positive_lhs_output, negative_lhs_output

        else:
            dialogue_output = self.bert(input_ids=inputs['dialogue']['input_ids'].view(4, -1),
                                           token_type_ids=inputs['dialogue']['role_ids'].view(4, -1),
                                           attention_mask=inputs['dialogue']['turn_ids'].view(4, -1))
            
            dialogue_last_hidden_state = dialogue_output['last_hidden_state']
            dialogue_lhs_output = dialogue_last_hidden_state.view(4, -1, 16, 768)
            dialogue_pooler = dialogue_output['pooler_output']
            return dialogue_lhs_output

    def encode(self, inputs, device): # 입력된 문장에 대한 임베딩을 계산

        # attention_mask = self.gen_attention_mask(inputs['input_ids'], inputs['valid_length'])

        embeddings_outputs = self.bert(input_ids=inputs['input_ids'].view(4, -1).to(device),
                                  token_type_ids=inputs['role_ids'].view(4, -1).to(device),
                                  attention_mask=inputs['turn_ids'].view(4, -1).to(device))
        embeddings = embeddings_outputs['last_hidden_state'].view(4, -1, 16, 768)
        #embeddings = embeddings_outputs['pooler_output']

        return embeddings

"""    def gen_attention_mask(self, token_ids, valid_length): # 어텐션 마스크를 생성, 유효한 길이 이후의 토큰은 어텐션 마스크가 0이 되도록 생성하여 패딩

        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1

        return attention_mask.float()"""
