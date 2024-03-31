import pandas as pd
import random
from tqdm import tqdm
tqdm.pandas()
from collections import Counter

class PairGenerator:
    def __init__(self, random_seed=None):
        """
        Initialize the PairGenerator class.
        :param datasetname: Name of the dataset.
        :param random_seed: Seed for random number generation.
        """
        self.random_seed = random_seed
        if self.random_seed is not None:
             random.seed(self.random_seed)
    
    def extract_turns(self, conversation, sample_idx, turn_sequence):
        """
        Extract turns from the conversation based on the given indices.
        :param conversation: String representing the conversation.
        :param idx: List of indices of turns to extract.
        :return: String containing extracted turns separated by '#'.
        """
        turns = conversation.split("#")
        all_turns = []
        all_interlocutors = []
        
        for idx in sample_idx:
            current_speaker = turn_sequence[idx]
            for i in range(idx, len(turn_sequence)):
                if i >= len(turns):
                    break
                # 다음 턴이 현재 대화자와 동일하거나, 리스트의 끝에 도달하면 중지
                if i > idx and turn_sequence[i] == current_speaker:
                    break
                # 현재 턴이 이전 턴과 다르거나, 첫 턴인 경우 추가
                if not all_turns or turns[i] != all_turns[-1]:
                    all_turns.append(turns[i])
                    all_interlocutors.append(turn_sequence[i])
                    
        return "#".join(all_turns), "".join(all_interlocutors)

    def generate_pairs(self, turn, positive_conversation, negative_conversations):
        """
        Generate positive and negative pairs based on the given conversation and turn information.
        :param turn: String representing the turn information.
        :param positive_conversation: String representing the positive conversation.
        :param negative_conversations: List of strings representing negative conversations.
        :return: String containing positive and negative pairs separated by '||'.
        """
        
        # 등장 횟수가 가장 많은 대화자를 기준으로 idx 추출
        basis_interlocutors = Counter(turn).most_common(len(turn))        
                
        idx = []
        pos_samples = []
        neg_samples = []
        neg_interlocutors = []
        interlocutors_turns = []
        for interlocutor, _ in basis_interlocutors:
            interlocutor_idx = [i for i, char in enumerate(turn) if char == interlocutor]
            # 추출 샘플 수는 기준 대화자의 절반
            sample_num = len(interlocutor_idx) // 2
            idx = random.sample(interlocutor_idx, sample_num)
            idx.sort()
            
            # print("==================================대화자 정보===========================================")
            # print("대화자",interlocutor)
            # print("idx",idx)
            # print("sample_num",sample_num)
            
            pos_sample, interlocutors_turn = self.extract_turns(positive_conversation, idx, turn)
            neg_sample = [self.extract_turns(neg_conv, idx, turn)[0] for neg_conv in negative_conversations]
            
            pos_samples.append(pos_sample)
            neg_samples.extend(neg_sample)
            interlocutors_turns.append(interlocutors_turn)
    
        # 기준 대화자의 발화 시점을 사용하여 positive 및 negative 샘플 생성
        pos_samples_joined = '|'.join(pos_samples)
        neg_samples_joined = '|'.join(neg_samples)
        
        # print("==========================pos_samples==========================")
        # print(pos_samples_joined)
        # print("==========================neg_samples==========================")
        # print(neg_samples_joined)
        
        
        # 최종 output
        pos_neg_pairs = f"{pos_samples_joined}||{neg_samples_joined}"

        return pos_neg_pairs, interlocutors_turn
    
    def generate_pairs_for_dataset(self, df):
        """
        Generate pairs for the dataset and save them to a new CSV file.
        """
        df['turn'] = df['turn'].astype('str')
        df['pairs'] = df.progress_apply(lambda row: self.generate_pairs(row['turn'], row['conversation'].split('|')[0], row['conversation'].split('|')[1:]), axis=1)
        
        # print("===================conversation check=================")
        # print("positive\n",df['conversation'].loc[0].split('|')[0])
        # print("negative\n",df['conversation'].loc[0].split('|')[1:])
        # print("pairs\n", df['pairs'].loc[0])
        # df.to_csv(f'mwoz_train_with_pairs.csv', index=False)
        
        return df['pairs']