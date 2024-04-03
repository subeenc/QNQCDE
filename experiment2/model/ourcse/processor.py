import os
import logging
from typing import List
#from apex import amp
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from model.loss import *
from model.utils import Metric
from data.dataloader import get_loader
from model.ourcse.bert import BERT

from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.loss = Loss(args)
        self.total_steps = 0
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0}
        self.dev_progress = {'score_RI': 0,
                    'score_NMI': 0,
                    'score_acc': 0,
                    'score_purity': 0,
                    'score_SR': 0,
                    'score_MRR': 0,
                    'score_MAP': 0,
                    'score_alignment': 0,
                    'score_adjusted_alignment': 0,
                    'score_uniformity': 0,
                    'iter':0}
        self.model_progress = {'loss': 0, 'iter': 0}
        self.best_dev_evaluation_result = EvaluationResult()
        self.best_test_evaluation_result = EvaluationResult()
        
        # logging configuration
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)# if args.local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger(__name__)
        self.logger = logger
    
    
    def plato_outuput(self, inputs):
        # torch.Size([4, 10, 64])
        input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels, domain_labels = inputs
        
        self.input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])
        self.attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        self.token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        self.role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        self.turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        self.position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])
        self.domain_labels = domain_labels
        
        outputs = self.config['model'](input_ids=self.input_ids.to(self.args.device), # .to(self.args.device) 붙이니까 에러남
                                    attention_mask=self.attention_mask.to(self.args.device), 
                                    token_type_ids=self.token_type_ids.to(self.args.device), 
                                    position_ids=self.position_ids.to(self.args.device), 
                                    turn_ids=self.turn_ids.to(self.args.device), 
                                    role_ids=self.role_ids.to(self.args.device),
                                    return_dict=True)
        # print("===== dialogue embeddings =====")
        # print("labels:", labels)
        # print(outputs['last_hidden_state'].shape)  # torch.Size([40, 64, 768])
        # print(outputs['hidden_states'])  # all_hidden_states(12 layers) -> 각 레이어 당 torch.Size([40, 64, 768])
        # print(outputs['pooler_output'].shape)  # torch.Size([40, 768])
        # print(outputs['response_selection_scores'].shape)  # torch.Size([40, 64, 1])  
        # print("input_ids:", input_ids.shape, input_ids)
        # print("role_ids:", role_ids.shape, role_ids)
        return outputs
    
    
    
    def run(self, all_dialgoue_embeddings, all_dialogue_domain_labels=None, type=None, tasks: List[str] = None):
        if type == 'train':           
            # positive, negative sample 분리       
            ## 실험1. 대화 단위      
            # lh_avg_output = torch.mean(outputs['last_hidden_state'], dim=1)
            # lh_avg_output = lh_avg_output.view(outputs['last_hidden_state'].size(0), -1)
            # print("lh_avg_output:", lh_avg_output.shape) # torch.Size([40, 768]) 
            # lh_avg_output = lh_avg_output.view(self.args.batch_size, -1, lh_avg_output.size(-1))
            # print("lh_avg_output(reshape):", lh_avg_output.shape) # torch.Size([4, 10, 768]) 
            
            # positive_embeddings = lh_avg_output[:, :1, :] # 첫 번째 샘플을 positive로 설정
            # negative_embeddings = lh_avg_output[:, 1:, :] # # 나머지 샘플을 positive로 설정
            # print("positive_embeddings:", positive_embeddings.shape) # torch.Size([4, 1, 768])
            # print("negative_embeddings:", negative_embeddings.shape) # torch.Size([4, 9, 768])
            
            ## 실험2. role_ids를 이용한 턴 단위
            role_id = self.role_ids.view(self.args.batch_size, -1, self.args.seq_len)
            positive_roleid = role_id[:, :1, :] # torch.Size([1, 64])
            negative_roleid = role_id[:, 1:, :] 
            
            _, _, hidden_size = all_dialgoue_embeddings['last_hidden_state'].size()
            output_embeddings = all_dialgoue_embeddings['last_hidden_state'].view(self.args.batch_size, -1, self.args.seq_len, hidden_size)
            positive_embeddings = output_embeddings[:, :1, :, :]
            negative_embeddings = output_embeddings[:, 1:, :, :]
            # print("positive_embeddings:", positive_embeddings.shape) # torch.Size([4, 1, 64, 768])
            # print("negative_embeddings:", negative_embeddings.shape) # torch.Size([4, 9, 64, 768])
            
            
            # 샘플링과 해당 결과를 바탕으로 loss 구하는 과정
            # train_loss_with_sampling 안에서 sampling과 loss 구하는 과정 함께 진행
            loss = self.loss.train_loss_with_sampling(self.config, positive_roleid, negative_roleid, positive_embeddings, negative_embeddings)
            return loss
        
        else:
            # dialgoue_embeddings = self.config['model'](inputs, type, self.args.batch_size, self.args.seq_len)
            # pooled_dialgoue_embeddings = torch.sum(dialgoue_embeddings, dim=1) # torch.Size([4, 768])
            
            # print("===== dialgoue_embeddings, inputs['label'] =====")
            # print(inputs['label'])
            # print(dialgoue_embeddings.shape)
            # print(pooled_dialgoue_embeddings.shape)
            # print(pooled_dialgoue_embeddings.shape)
            # print(len(dialgoue_embeddings), len(inputs['label']))
            
            best_evaluation_result = self.best_test_evaluation_result if type == 'test' else self.best_dev_evaluation_result
            evaluation_result = EvaluationResult()
            # print("================inputs, labels================")   
            # print(inputs.shape)
            # print(labels.shape)
            
            if 'clustering' in tasks:
                # print("len",len(all_dialgoue_embeddings))
                n_average = max(3, 10 - self.total_data_count // 500)
                print("==============cluster data input: features.shape[0], n_average==============")
                print(self.total_data_count, all_dialgoue_embeddings.shape[0], n_average) 
                er = self.loss.evaluation_during_training(features=all_dialgoue_embeddings,
                                                          labels=all_dialogue_domain_labels,
                                                          gpu_features=None,
                                                          n_average=n_average,
                                                          tasks=['clustering'],
                                                          dtype='float32',
                                                          tsne_visualization_output=None,
                                                          logger=None,
                                                          note=type)
                
                evaluation_result.RI = er.RI
                evaluation_result.NMI = er.NMI
                evaluation_result.acc = er.acc
                evaluation_result.purity = er.purity

            # based on acc, plz ref metrics.py->EvaluationResult.__lr__()
            # print("evaluation_result",evaluation_result)
            # print()
            # print('is_best test -> evaluation_result:', evaluation_result)
            # print('is_best test -> best_evaluation_result:', best_evaluation_result)
            is_best = True if evaluation_result.acc > best_evaluation_result.acc else False
            # print("is_best:", is_best)   
                
            if 'semantic_relatedness' in tasks or 'session_retrieval' in tasks:
                #if type == 'test':
                er = self.loss.evaluation_during_training(features=all_dialgoue_embeddings,
                                                        labels=all_dialogue_domain_labels,
                                                        gpu_features=None,
                                                        n_average=0,
                                                        tasks=['semantic_relatedness', 'session_retrieval'],
                                                        dtype='float32',
                                                        tsne_visualization_output=None,
                                                        logger=None,
                                                        note=type)
                
                evaluation_result.SR = er.SR
                evaluation_result.MRR = er.MRR
                evaluation_result.MAP = er.MAP

            if 'align_uniform' in tasks:
                #if type == 'test':
                er = self.loss.evaluation_during_training(features=all_dialgoue_embeddings,
                                                        labels=all_dialogue_domain_labels,
                                                        gpu_features=None,
                                                        n_average=0,
                                                        tasks=['align_uniform'],
                                                        dtype='float32',
                                                        tsne_visualization_output=None,
                                                        logger=None,
                                                        note=type)

                evaluation_result.alignment = er.alignment.item()
                evaluation_result.adjusted_alignment = er.adjusted_alignment.item()
                evaluation_result.uniformity = er.uniformity.item()

            # print("=======best_evaluation_result==========")
            # print(best_evaluation_result)
            
            # if is_best:
            #     # best_evaluation_result 객체를 업데이트하는 대신
            #     # 직접 self.best_dev_evaluation_result 또는 self.best_test_evaluation_result 객체를 업데이트합니다.
            #     if type == 'test':
            #         self.best_test_evaluation_result.update(er)
            #         self.best_test_evaluation_result.show(logger=self.logger, note=type)
            #     else:
            #         self.best_dev_evaluation_result.update(er)
            #         self.best_dev_evaluation_result.show(logger=self.logger, note=type)
                
            # # self.best_dev_evaluation_result.update(best_evaluation_result)
            # # best_evaluation_result.show(logger=self.logger,note=type)
            
            # return is_best, er
            # evaluation_result.show(logger=self.logger, note=type)

            return is_best, evaluation_result
            
                            
    def progress(self, loss):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1

    # def progress_validation(self, score):
    #     self.dev_progress['score'] += score
    #     self.dev_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'].data.cpu().numpy() / self.model_progress['iter']

        return loss, acc

    def get_object(self, tokenizer, model):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=self.args.warmup_ratio * train_total,
                                                    num_training_steps=train_total)

        return scheduler, train_total

    def model_setting(self):
        model, loader, tokenizer = get_loader(self.args, self.metric) # 데이터로더 가져오는 부분 /data/dataloader.py
        print()
        
        # model = BERT(model) # 사전학습된 bert 모델이나 가중치가 사용되면 여기서는 skt가 학습했던 kobert가 들어가는 것으로 판단
        model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)

        if self.args.train == 'True':
            scheduler, total_steps = self.get_scheduler(optimizer, loader['train'])
            self.total_steps = total_steps
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': self.args,
                  'model': model}

        if config['args'].fp16 == 'True':
            #config['model'], config['optimizer'] = amp.initialize(
            #    config['model'], config['optimizer'], opt_level=config['args'].opt_level)
            config['scaler'] = GradScaler() # apex 에러 수정

        self.config = config

        return self.config

    def train(self, epoch):
        print("==== Train ====")
        self.config['model'].train()

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            # batch = tuple(t.to(self.args.device) for t in batch)
            inputs = batch           
            all_dialgoue_embeddings = self.plato_outuput(inputs)

            with autocast(enabled=self.args.fp16 == 'True'): # apex 에러 해결
                train_loss = self.run(all_dialgoue_embeddings, type='train')

            if self.args.fp16 == 'True':
                #with amp.scale_loss(train_loss, self.config['optimizer']) as scaled_loss: # apex 에러 해결
                #    scaled_loss.backward()
                self.config['scaler'].scale(train_loss).backward()
                self.config['scaler'].step(self.config['optimizer'])
                self.config['scaler'].update()
                
            else:
                train_loss.backward() # apex 에러 해결
                self.config['optimizer'].step()

            # self.config['optimizer'].step() # apex 에러 해결
            self.config['scheduler'].step()

            self.progress(train_loss.data)
            
            # print("train_loss: ", train_loss)
            # print("==============self.args.eval_steps, self.model_progress['iter']==============")
            # print(self.args.eval_steps )
            # print(self.model_progress['iter']) # = globel_step
            
            # if self.model_progress['iter'] % self.args.eval_steps == 0 or self.model_progress['iter'] == self.total_steps:
        self.valid()
        
        self.config['model'].train()
  
    def valid(self):
        print("==== Valid ====")
        self.config['model'].eval()
        self.dev_progress = self.dev_progress.fromkeys(self.dev_progress, 0)

        # score_indicator = {'eval_RI': 0,
        #             'eval_NMI': 0,
        #             'eval_acc': 0,
        #             'eval_purity': 0,
        #             'eval_SR': 0,
        #             'eval_MRR': 0,
        #             'eval_MAP': 0,
        #             'eval_alignment': 0,
        #             'eval_adjusted_alignment': 0,
        #             'eval_uniformity': 0}
    
        with torch.no_grad():
            all_dialgoue_embeddings = []
            all_dialogue_domain_labels = []
            self.total_data_count = 0
            for step, batch in enumerate(tqdm(self.config['loader']['valid'])):
                # print("step:", step)
                # batch = tuple(t.to(self.args.device) for t in batch)
                inputs = batch
                dialgoue_embeddings = self.plato_outuput(inputs)
                pooled_dialgoue_embeddings = torch.sum(dialgoue_embeddings.last_hidden_state, dim=1) # torch.Size([4, 768])
                # print(pooled_dialgoue_embeddings.shape)
                batch_domain_label = self.domain_labels
                
                batch_size = dialgoue_embeddings.last_hidden_state.size(0)
                self.total_data_count += batch_size
                
                all_dialgoue_embeddings.append(pooled_dialgoue_embeddings)
                all_dialogue_domain_labels.append(batch_domain_label)
                # print("batch_domain_label",batch_domain_label)
                            
            all_dialgoue_embeddings = torch.cat(all_dialgoue_embeddings, dim=0) 
            all_dialogue_domain_labels = torch.cat(all_dialogue_domain_labels, dim=0)
            # print(all_dialogue_domain_labels)
            
            is_best, dev_evaluation_result = self.run(all_dialgoue_embeddings, all_dialogue_domain_labels, type='valid',
                                 tasks=['clustering', 'semantic_relatedness', 'session_retrieval'])#, 'semantic_relatedness', 'session_retrieval', 'align_uniform'])
            print("dev_evaluation_result:", dev_evaluation_result)    
                
            if is_best:
                self.best_dev_evaluation_result.update(dev_evaluation_result)
                # self.best_dev_evaluation_result.show(logger=self.logger, note=type)

            # self.best_dev_evaluation_result.update(best_evaluation_result)
            # best_evaluation_result.show(logger=self.logger,note=type)             
                    
        self.best_dev_evaluation_result.show(logger=logger, note='valid')

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.args.path_to_saved_model)['model'], strict=False)
        self.config['model'].eval()
        self.dev_progress = self.dev_progress.fromkeys(self.dev_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['test']):
                inputs = batch
                _, test_evaluation_result = self.run(inputs, type='test',
                                 tasks=['clustering', 'semantic_relatedness', 'session_retrieval'])#, 'semantic_relatedness', 'session_retrieval', 'align_uniform'])

                self.best_test_evaluation_result.update(test_evaluation_result)
                # self.best_test_evaluation_result.show(logger=self.logger, note=type)
                # self.best_test_evaluation_result = test_evaluation_result
                #self.progress_validation(score)

        logger.info('### TEST SCORE ###')
        self.best_test_evaluation_result.show(logger=logger, note='test')
        #score = self.metric.cal_dev_score(self.dev_progress, score_indicator)