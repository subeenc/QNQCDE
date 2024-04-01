
import os
import numpy
import pandas as pd
import glob2
import itertools
import torch
import csv
from tqdm import tqdm
import logging
import gluonnlp as nlp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoTokenizer, AutoConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

#from KoBERT.kobert.utils import get_tokenizer
#from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model # 사전학습된 kobert 사용
#import config

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel

from .generate_pairs import PairGenerator

import warnings
warnings.filterwarnings('ignore') 
# warnings.filterwarnings('default')


logger = logging.getLogger(__name__)

class DialogueFeatures():
    def __init__(self, input_ids, input_mask, segment_ids, role_ids, label_id, turn_ids=None, position_ids=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.role_ids = role_ids
        self.turn_ids = turn_ids
        self.position_ids = position_ids
        self.label_id = label_id
        self.guid = guid

        self.batch_size = len(self.input_ids)

class DialogueDataset():
    def __init__(self, file_path, args, metric, type):
        self.args = args
        self.type = type
        self.metric = metric
        self.file_path = file_path       
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #self.tokenizer_config = PlatoConfig.from_json_file("plato/config.json")
 
    def load_data(self):
        column_names = ['turn', 'conversation', 'label']

        df = pd.read_csv(self.file_path, sep='\t', header=None, names=column_names)
        df = df.dropna(subset=['label'])
        df = df[:12]
        # tqdm_pandas = tqdm(total=len(df), desc="Processing")

        features = []
        # features = df.apply(lambda row: self.get_dialfeature(row, self.type), axis=1).dropna().tolist()

        # with ThreadPoolExecutor(max_workers=4) as executor:  # 동시에 실행할 작업 수를 정의
        #     # 각 행에 대한 처리 작업을 제출하고 Future 객체를 리스트에 저장
        #     futures = [executor.submit(self.get_dialfeature, row, self.type) for _, row in df.iterrows()]
            
        #     # as_completed()를 사용하여 완료된 순서대로 결과를 받음
        #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        #         result = future.result()
        #         if result is not None:
        #             features.append(result)
        
        features = df.apply(lambda row: self.get_dialfeature(row, self.type), axis=1).tolist()
        # features = [self.get_dialfeature(row, self.type) for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing")]
        features = [feature for feature in features if feature is not None]
        
        return features

    def get_dialfeature(self, example, type):  # 기존: data2tensor(self, line, type), pos, neg 각각 실행
        """
        examples : turn, conversation, label 컬럼으로 구성된 데이터프레임 행
        """
        
        # 차후 config에 넣고 활용
        line_sep_token = "\t"
        sample_sep_token = "|"
        turn_sep_token = "#"
        
        use_response = False
        if type in ['train', 'valid', 'test']:
            sample_list = example['conversation'].split(sample_sep_token)  # positive sample 1개, negative sample 9개
            role_list = [int(r) for r in example['turn'].split(sample_sep_token)] \
                        if example['turn'].find(turn_sep_token) != -1 \
                        else [int(r) for r in example['turn']]

            sample_input_ids = []
            sample_segment_ids = []
            sample_role_ids = []
            sample_input_mask = []
            sample_turn_ids = []
            sample_position_ids = []
            
            for t, s in enumerate(sample_list):
                if len(sample_list)>10 or len(role_list)>15:
                    # print(t, len(sample_list), len(role_list))
                    return None
                
                text_tokens = []
                text_turn_ids = []
                text_role_ids = []
                text_segment_ids = []
                
                # config.turn_sep_token = '#'
                text_list = s.split(turn_sep_token)

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
                    context, response = text_list[:-1], text_list[-1]  # 마지막 발화를 response로, 나머지 발화들을 context로 분리
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
                context = context[-self.args.seq_len:]

                for i, text in enumerate(context):
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

                # self.args.seq_len=512
                if len(text_tokens) > self.args.seq_len:
                    text_tokens = text_tokens[:self.args.seq_len]
                    text_turn_ids = text_turn_ids[:self.args.seq_len]
                    text_role_ids = text_role_ids[:self.args.seq_len]
                    text_segment_ids = text_segment_ids[:self.args.seq_len]

                assert (max(text_turn_ids) <= self.args.seq_len)

                # 制作text_position_id序列
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

                assert len(text_input_ids) == self.args.seq_len
                assert len(text_turn_ids) == self.args.seq_len
                assert len(text_role_ids) == self.args.seq_len
                assert len(text_segment_ids) == self.args.seq_len
                assert len(text_position_ids) == self.args.seq_len
                assert len(text_input_mask) == self.args.seq_len

                sample_input_ids.append(text_input_ids)
                sample_turn_ids.append(text_turn_ids)
                sample_role_ids.append(text_role_ids)
                sample_segment_ids.append(text_segment_ids)
                sample_position_ids.append(text_position_ids)
                sample_input_mask.append(text_input_mask)
                
                label_id = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                feature = DialogueFeatures(input_ids=sample_input_ids,
                                                input_mask=sample_input_mask,
                                                segment_ids=sample_segment_ids,
                                                role_ids=sample_role_ids,
                                                turn_ids=sample_turn_ids,
                                                position_ids=sample_position_ids,
                                            label_id=label_id)
            return feature

    def _get_loader(self, features):
        if self.type=="train":
            self.num_train_steps = int(len(features) / self.args.batch_size * self.args.epochs)

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(features))
            logger.info("  Batch size = %d", self.args.batch_size)
            logger.info("  Num steps = %d", self.num_train_steps)
            
            # train_sampler = RandomSampler(data) if self.args.local_rank == -1 else DistributedSampler(data, num_replicas=torch.cuda.device_count(), rank=self.args.local_rank)

        # torch.Size([12, 10, 64]) -> #data, #samples seq_len
        all_input_ids = torch.tensor([(f.input_ids) for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_role_ids = torch.tensor([f.role_ids for f in features], dtype=torch.long)
        all_turn_ids = torch.tensor([f.turn_ids for f in features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        # print("====================Dataloader loader data dim==========================")
        # print(all_input_ids.shape)  # torch.Size([2197, 10, 64])
        
        data = TensorDataset(all_input_ids,
                            all_input_mask,
                            all_segment_ids,
                            all_role_ids,
                            all_turn_ids,
                            all_position_ids,
                            all_label_ids)
        
        self.loader = DataLoader(data,
                                batch_size=self.args.batch_size)
                                # shuffle=True)
        
        # inputs = self.metric.move2device(inputs, self.args.device)
        return self.loader

def load_model(args):    
    config = PlatoConfig.from_json_file(args.config_file)
    plato = PlatoModel(config, args)
    
    if args.init_checkpoint is not None:
        state_dict = torch.load(args.init_checkpoint) #, map_location="cpu"
        if args.init_checkpoint.endswith('pt'):
            plato.load_state_dict(state_dict, strict=False)
        else:
            plato.load_state_dict(state_dict, strict=False)

        # logger.debug(plato.module) if hasattr(plato, 'module') else logger.debug(plato)
    return plato
                
# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    # bert_model = BertModel.from_pretrained('bert-base-uncased')
    plato_model = load_model(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=50, pad=True, pair=False)
    
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data  + '/' + args.test_data    

    if args.train == 'True' and args.test == 'False':
        train_iter = DialogueDataset(path_to_train_data, args, metric, type='train') # type='train'
        valid_iter = DialogueDataset(path_to_valid_data, args, metric, type='valid') # type='valid'
        
        train_features = train_iter.load_data()
        valid_features = valid_iter.load_data()
        
        # train_loader = train_iter._get_loader(train_features)
        # print("train_iter._get_loader(train_features)")
        # print(train_loader)
        
        loader = {'train': train_iter._get_loader(train_features),
                  'valid': train_iter._get_loader(valid_features)}
    elif args.train == 'False' and args.test == 'True':
        test_iter = DialogueDataset(path_to_test_data, args, metric, type='test') # type='test'
        test_features = test_iter.load_data()

        loader = {'test': test_iter._get_loader(test_features)}
    else:
        loader = None

    return plato_model, loader, tokenizer


if __name__ == '__main__':
    get_loader('test')