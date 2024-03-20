import torch
from torch import nn


class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert

    def forward(self, inputs, mode, batch_size, seq_len):

        if mode=='train':
            
            # print("========== inputs['positive']['input_ids'] ==========")
            # print(inputs['positive']['input_ids'].shape) # torch.Size([4, 8, seq_len])
            # print((inputs['positive']['input_ids'].view(4, -1)).shape) # torch.Size([4, 128])
            
            positive_output = self.bert(input_ids=inputs['positive']['input_ids'].view(batch_size, -1),
                                           token_type_ids=inputs['positive']['role_ids'].view(batch_size, -1),
                                           attention_mask=inputs['positive']['turn_ids'].view(batch_size, -1))
            
            positive_last_hidden_state = positive_output['last_hidden_state']
            positive_lhs_output = positive_last_hidden_state.view(batch_size, -1, seq_len, 768)
            positive_lhs_output = torch.mean(positive_lhs_output, dim=2)
            positive_pooler = positive_output['pooler_output'] # cls token
            
            negative_output = self.bert(input_ids=inputs['negative']['input_ids'].view(batch_size, -1),
                                           token_type_ids=inputs['negative']['role_ids'].view(batch_size, -1),
                                           attention_mask=inputs['negative']['turn_ids'].view(batch_size, -1))
            
            negative_last_hidden_state = negative_output['last_hidden_state']
            negative_lhs_output = negative_last_hidden_state.view(batch_size, -1, seq_len, 768)
            negative_lhs_output = torch.mean(negative_lhs_output, dim=2)
            # negative_pooler = negative_output['pooler_output'] # cls token
            
            # print("========== positive_output", positive_output)
            # print("========== positive_last_hidden_state", positive_last_hidden_state, positive_last_hidden_state.shape) # torch.Size([4, 256, 768])
            # print("========== positive_lhs_output", positive_lhs_output.shape, "negative_lhs_output:", negative_lhs_output.shape)
            # print("========== positive_pooler_tmp", positive_pooler_tmp, positive_pooler_tmp.shape) # torch.Size([4, 768])
            
            return positive_lhs_output, negative_lhs_output

        else:
            dialogue_output = self.bert(input_ids=inputs['dialogue']['input_ids'].view(batch_size, -1),
                                           token_type_ids=inputs['dialogue']['role_ids'].view(batch_size, -1),
                                           attention_mask=inputs['dialogue']['turn_ids'].view(batch_size, -1))
            
            dialogue_last_hidden_state = dialogue_output['last_hidden_state']
            dialogue_lhs_output = dialogue_last_hidden_state.view(batch_size, -1, seq_len, 768)
            dialogue_lhs_output = torch.mean(dialogue_lhs_output, dim=2)
            # dialogue_pooler = dialogue_output['pooler_output'] # cls token
            
            return dialogue_lhs_output

    def encode(self, inputs, batch_size, seq_len, device):

        embeddings_outputs = self.bert(input_ids=inputs['input_ids'].view(batch_size, -1).to(device),
                                  token_type_ids=inputs['role_ids'].view(batch_size, -1).to(device),
                                  attention_mask=inputs['turn_ids'].view(batch_size, -1).to(device))
        embeddings = embeddings_outputs['last_hidden_state'].view(batch_size, -1, seq_len, 768)
        embeddings = torch.mean(embeddings, dim=2)
        # embeddings = embeddings_outputs['pooler_output']

        return embeddings

