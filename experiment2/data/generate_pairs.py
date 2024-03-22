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
        basis_interlocutor = Counter(turn).most_common(1)[0][0]
        basis_interlocutor_turn_idx = [i for i, char in enumerate(turn) if char == basis_interlocutor]

        # 추출 샘플 수는 기준 대화자 발화의 절반(반올림)
        sample_num = round(len(basis_interlocutor_turn_idx) / 2)
        idx = random.sample(basis_interlocutor_turn_idx, sample_num)
        idx.sort()
        
        # 기준 대화자의 발화 시점을 사용하여 positive 및 negative 샘플 생성
        pos_sample, interlocutors_turn = self.extract_turns(positive_conversation, idx, turn)
        neg_samples = [self.extract_turns(neg_conv, idx, turn)[0] for neg_conv in negative_conversations]
        neg_samples_joined = '|'.join(neg_samples)
        
        pos_neg_pairs = f"{pos_sample}||{neg_samples_joined}"

        return pos_neg_pairs, interlocutors_turn
    
    def generate_pairs_for_dataset(self, df):
        """
        Generate pairs for the dataset and save them to a new CSV file.
        """
        df['turn'] = df['turn'].astype('str')
        df['pairs'] = df.progress_apply(lambda row: self.generate_pairs(row['turn'], row['conversation'].split('|')[0], row['conversation'].split('|')[1:]), axis=1)
        # df.to_csv(f'{self.datasetname}_train_with_pairs.csv', index=False)
        return df['pairs']