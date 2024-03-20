
import numpy
import pandas as pd
import glob2
import itertools
import torch
import csv
from tqdm import tqdm
import logging
import gluonnlp as nlp
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoTokenizer, AutoConfig
#from KoBERT.kobert.utils import get_tokenizer
#from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model # 사전학습된 kobert 사용
#import config
#from model.plato.configuration_plato import PlatoConfig

import warnings
warnings.filterwarnings('ignore') 
# warnings.filterwarnings('default')


logger = logging.getLogger(__name__)

class DialogueFeatures():
    def __init__(self, input_ids, input_mask, segment_ids, role_ids, turn_ids=None, position_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.role_ids = role_ids
        self.turn_ids = turn_ids
        self.position_ids = position_ids

        self.batch_size = len(self.input_ids)

class DialogueDataset(Dataset): # 기존 ModelDataLoader
    def __init__(self, file_path, args, metric, type):
        
        """
        data: 각 대화 샘플을 포함하는 리스트.
        예시)
        각 대화 샘플은 positive 대화 2개와 negative 대화 2개를 포함하는 리스트이다.
        """

        #  -------------------------------------
        self.args = args
        self.type = type
        self.metric = metric
        self.file_path = file_path
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #self.tokenizer_config = PlatoConfig.from_json_file("plato/config.json")
        

    def load_data(self, type):
        df = pd.read_csv(self.file_path)  # 확인용으로 10개만 추출
        self.len_data = len(df)
        # train
        self.positive_pairs = []
        self.negative_pairs = []
        # valid, test
        self.dialogues = []
        self.labels = []
        self.roles = []
        
        if type=='train':
            for _, row in df.iterrows():
                pair = row['pairs'].split('||')
                positive_pair = pair[0].split('|')  # Positive pair 분리
                negative_pair = pair[1].split('|')[:len(positive_pair)]  # Negative pair 분리 (positive pair와 개수 동일하게)
                label = [1] * len(positive_pair) + [0] * len(negative_pair)
                role = row['turn']
                
                self.positive_pairs.append(positive_pair)
                self.negative_pairs.append(negative_pair)
                self.labels.append(label)
                self.roles.append(role)
                
            self.positive_features = self.get_dialfeature(self.positive_pairs, self.type)
            self.negative_features = self.get_dialfeature(self.negative_pairs, self.type) 
        
        else:
            for _, row in df.iterrows():
                pair = row['pairs'].split('||')
                dialgoue = pair[0].split('|')
                label = row['label'] # 여기서의 label은 domain label
                role = row['turn']
                
                self.dialogues.append(dialgoue) # dialogue가 하나의 대화
                self.labels.append(label) # label
                self.roles.append(role)
            
            self.dialogues_features = self.get_dialfeature(self.dialogues, self.type)
            
        if type=='train':
            assert len(self.positive_pairs) == len(self.negative_pairs)
        else:
            pass
            

    def get_dialfeature(self, all_dial_pairs, type):  # 기존: data2tensor(self, line, type), pos, neg 각각 실행
        """
        all_dial_pairs = [
            
            [pair1, pair2,...,pairN]  # 하나의 대화 내 pair들
            ['new contact add#name', "5''1'#eye color"],
            ["No, that is it, thanks so much!#You're welcome.", "5''1'#eye color"]
            ...
            전체 대화 개수
        ]
        """
        
        # 차후 config에 넣고 활용
        line_sep_token = "\t"
        sample_sep_token = "|"
        turn_sep_token = "#" 
        
        use_response = False
        features = []
        if type in ['train', 'valid', 'test']:
            for dial_pairs, dial_role in zip(all_dial_pairs, self.roles):
                
                # all_dial_pairs: 전체 대화와 각 대화의 pair들이 존재하는 list of list 형태
                # dial_pairs: 한 대화 내 pair들로 구성된 리스트
                # pair: 한 대화 내 각 pair가 문자열로 구성 -> 한 pair는 두 개의 turn을 구성, #구분자로 분리 가능
            
                # 한 대화 내 pair를 모두 담기 위한 list
                sample_input_ids = []
                sample_segment_ids = []
                sample_role_ids = []
                sample_input_mask = []
                sample_turn_ids = []
                sample_position_ids = []
                        
                for t, pair in enumerate(dial_pairs):
                    text_tokens = []
                    text_turn_ids = []
                    text_role_ids = []
                    text_segment_ids = []
                        
                    # config.turn_sep_token='#'
                    text_list = pair.split(turn_sep_token)
                    
                    # 하나의 pair에 대한 role
                    # 하나의 pair에 존재하는 turn에 따라 role 구분
                    # 0으로 padding되므로 role은 2,1 내림차순으로 생성 -> 다중대화에선 추출한 대화 인덱스 부분의 role 정보 필요
                    # role_list = [2 if i % 2 == 0 else 1 for i in range(len(text_list))]
                    role_list = [0, 1]

                    # token: token [eou] token [eou] [bos] token [eos]
                    # role:   0     0     1     1     0     0      0
                    # turn:   2     2     1     1     0     0      0
                    # pos:    0     1     0     1     0     1      2
                    
                    # bou: begin of utterance
                    # eou: end of utterance
                    # bos: begin of sentence
                    # eos: end of sentence
                    bou, eou, bos, eos = "[unused0]", "[unused1]", "[unused0]", "[unused1]"

                    # use [CLS] as the latent variable of PLATO
                    # text_list[0] = self.args.start_token + ' ' + text_list[0]

                    if use_response == True:   # specify the context and response
                        context, response = text_list[:-1], text_list[-1]
                        word_list = self.tokenizer.tokenize(response)
                        uttr_len = len(word_list)

                        start_token, end_token = bou, eou

                        role_id, turn_id = role_list[-1], 0

                        response_tokens = [start_token] + word_list + [end_token]
                        response_role_ids = [role_id] * (1 + uttr_len + 1)
                        response_turn_ids = [turn_id] * (1 + uttr_len + 1)
                        response_segment_ids = [0] * (1 + uttr_len + 1)                   # not use

                    else:
                        context = text_list
                        response_tokens, response_role_ids, response_turn_ids, response_segment_ids = [], [], [], []

                        # limit the context length
                        # context = context[-self.args.max_context_length:]
                        # context = context[-self.args.seq_len:]  # 가장 최근 512개 턴만 유지(특정 길이 데이터 제한, hyperparameter 변경 필요)

                    '''
                    use_response == False일 경우, 한 대화(샘플)에서 분리된 턴이 하나씩 들어감
                    
                    '''
                    # 한 turn씩 반복
                    for i, text in enumerate(context):
                        # print(text)
                        word_list = self.tokenizer.tokenize(text)
                        uttr_len = len(word_list)

                        end_token = eou

                        role_id, turn_id = role_list[i], len(context) - i

                        text_tokens.extend(word_list + [end_token])
                        text_role_ids.extend([role_id] * (uttr_len + 1))
                        text_turn_ids.extend([turn_id] * (uttr_len + 1))
                        text_segment_ids.extend([0] * (uttr_len + 1))

                    
                    text_tokens.extend(response_tokens)
                    text_role_ids.extend(response_role_ids)
                    text_turn_ids.extend(response_turn_ids)
                    text_segment_ids.extend(response_segment_ids)

                    if len(text_tokens) > self.args.seq_len:
                        text_tokens = text_tokens[:self.args.seq_len]
                        text_turn_ids = text_turn_ids[:self.args.seq_len]
                        text_role_ids = text_role_ids[:self.args.seq_len]
                        text_segment_ids = text_segment_ids[:self.args.seq_len]

                    #  max_context_length=15
                    assert (max(text_turn_ids) <= 15)

                    # 制作text_position_id序列  -> Make text_position_id sequence
                    text_position_ids = []
                    text_position_id = 0
                    for i, turn_id in enumerate(text_turn_ids):
                        if i != 0 and turn_id < text_turn_ids[i - 1]:   # PLATO
                            text_position_id = 0
                        text_position_ids.append(text_position_id)
                        text_position_id += 1
                    

                    # max_turn_id = max(text_turn_ids)
                    # text_turn_ids = [max_turn_id - t for t in text_turn_ids]

                    text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
                    text_input_mask = [1] * len(text_input_ids)
                    

                    # Zero-pad up to the sequence length.
                    while len(text_input_ids) < self.args.seq_len:
                        text_input_ids.append(0)
                        text_turn_ids.append(0)
                        text_role_ids.append(0)
                        text_segment_ids.append(0)
                        text_position_ids.append(0)
                        text_input_mask.append(0)

                    # max_context_lengt=512
                    assert len(text_input_ids) == self.args.seq_len
                    assert len(text_turn_ids) == self.args.seq_len
                    assert len(text_role_ids) == self.args.seq_len
                    assert len(text_segment_ids) ==self.args.seq_len
                    assert len(text_position_ids) == self.args.seq_len
                    assert len(text_input_mask) == self.args.seq_len
                    
                    sample_input_ids.append(text_input_ids)
                    sample_turn_ids.append(text_turn_ids)
                    sample_role_ids.append(text_role_ids)
                    sample_segment_ids.append(text_segment_ids)
                    sample_position_ids.append(text_position_ids)
                    sample_input_mask.append(text_input_mask)

                bert_feature = DialogueFeatures(input_ids=sample_input_ids,
                                            input_mask=sample_input_mask,
                                            segment_ids=sample_segment_ids,
                                            role_ids=sample_role_ids,
                                            turn_ids=sample_turn_ids,
                                            position_ids=sample_position_ids)

                features.append(bert_feature)        
        return features

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 대화 샘플을 반환한다.
        각 대화 샘플은 positive 대화 2개와 negative 대화 2개를 포함한다.
        """

        if self.type=='train':
            
            # 선택된 대화 샘플을 반환
            positive_features = self.positive_features[idx]        
            negative_features = self.negative_features[idx]
            
            inputs = {'positive':{
                        'input_ids':torch.LongTensor(positive_features.input_ids),
                        'role_ids':torch.LongTensor(positive_features.role_ids),
                        'turn_ids':torch.LongTensor(positive_features.turn_ids),
                    },
                    'negative':{
                        'input_ids':torch.LongTensor(negative_features.input_ids),
                        'role_ids':torch.LongTensor(negative_features.role_ids),
                        'turn_ids':torch.LongTensor(negative_features.turn_ids),
                    },
            }
        
        else:
            dialogues_features = self.dialogues_features[idx]
            labels = self.labels[idx]
            # print("===== labels = self.labels[idx]")
            # print(labels)
                    
            inputs = {'dialogue':{
                        'input_ids':torch.LongTensor(dialogues_features.input_ids),
                        'role_ids':torch.LongTensor(dialogues_features.role_ids),
                        'turn_ids':torch.LongTensor(dialogues_features.turn_ids),
                        },
                      'label': torch.tensor(labels, dtype=torch.float)
                    }
        
        # print('========== getitem_inputs_check ===========')
        # print(inputs)
        inputs = self.metric.move2device(inputs, self.args.device)

        return inputs

    def __len__(self):
        # 데이터셋의 길이는 전체 대화 샘플의 수
        if self.type=='train':
            return self.len_data
        else:
            return len(self.labels)

def custom_collate_fn_train(batch):
    # batch는 리스트 형태이며, 각 원소는 __getitem__에서 반환된 결과입니다.
    # 각 특성별로 패딩을 적용합니다.
    pos_input_ids = pad_sequence([torch.tensor(sample['positive']['input_ids']) for sample in batch], batch_first=True, padding_value=0)
    pos_role_ids = pad_sequence([torch.tensor(sample['positive']['role_ids']) for sample in batch], batch_first=True, padding_value=0)
    pos_turn_ids = pad_sequence([torch.tensor(sample['positive']['turn_ids']) for sample in batch], batch_first=True, padding_value=0)
    
    neg_input_ids = pad_sequence([torch.tensor(sample['negative']['input_ids']) for sample in batch], batch_first=True, padding_value=0)
    neg_role_ids = pad_sequence([torch.tensor(sample['negative']['role_ids']) for sample in batch], batch_first=True, padding_value=0)
    neg_turn_ids = pad_sequence([torch.tensor(sample['negative']['turn_ids']) for sample in batch], batch_first=True, padding_value=0)
    
    # 모든 특성에 대한 패딩 적용 후 최종 배치 데이터 구성
    batched_data = {
        'positive': {
            'input_ids': pos_input_ids,
            'role_ids': pos_role_ids,
            'turn_ids': pos_turn_ids
        },
        'negative': {
            'input_ids': neg_input_ids,
            'role_ids': neg_role_ids,
            'turn_ids': neg_turn_ids
        }
    }

    return batched_data

def custom_collate_fn_test(batch):

    dial_input_ids = pad_sequence([torch.tensor(sample['dialogue']['input_ids']) for sample in batch], batch_first=True, padding_value=0)
    dial_role_ids = pad_sequence([torch.tensor(sample['dialogue']['role_ids']) for sample in batch], batch_first=True, padding_value=0)
    dial_turn_ids = pad_sequence([torch.tensor(sample['dialogue']['turn_ids']) for sample in batch], batch_first=True, padding_value=0)
    dial_label_ids = torch.tensor([sample['label'] for sample in batch], dtype=torch.float)
    
    batched_data = {
        'dialogue': {
            'input_ids': dial_input_ids,
            'role_ids': dial_role_ids,
            'turn_ids': dial_turn_ids
            # 'label_ids': dial_label_ids
        },
        'label': dial_label_ids
    }

    return batched_data

# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=50, pad=True, pair=False)

    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data  + '/' + args.test_data    

    if args.train == 'True' and args.test == 'False':
        train_iter = DialogueDataset(path_to_train_data, args, metric, type='train') # type='train'
        valid_iter = DialogueDataset(path_to_valid_data, args, metric, type='valid') # type='valid'
        
        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=custom_collate_fn_train,
                                      drop_last=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=custom_collate_fn_test,
                                      drop_last=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = DialogueDataset(path_to_test_data, args, metric, type='test') # type='test'
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     collate_fn=custom_collate_fn_test,
                                     drop_last=True)}

    else:
        loader = None

    return bert_model, loader, tokenizer


"""def convert_to_tensor(corpus, transform):
    tensor_corpus = []
    tensor_valid_length = []
    tensor_segment_ids = []
    for step, sentence in enumerate(corpus):
        cur_sentence, valid_length, segment_ids = transform([sentence])

        tensor_corpus.append(cur_sentence)
        tensor_valid_length.append(numpy.array([valid_length]))
        tensor_segment_ids.append(segment_ids)

    inputs = {'source': torch.LongTensor(tensor_corpus),
              'segment_ids': torch.LongTensor(tensor_segment_ids),
              'valid_length': torch.tensor(tensor_valid_length)}

    return inputs"""


def example_model_setting(model_ckpt):

    from model.ourcse.bert import BERT

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=50, pad=True, pair=False)

    model = BERT(bert_model)

    model.load_state_dict(torch.load(model_ckpt)['model'])
    model.to(device)
    model.eval()

    return model, transform, device


if __name__ == '__main__':
    get_loader('test')
