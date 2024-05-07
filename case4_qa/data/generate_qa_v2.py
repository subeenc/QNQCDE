import pandas as pd
import argparse

class Generate_QA:
    def __init__(self, datasetname, mode, path):
        self.datasetname = datasetname
        self.mode = mode
        self.path = path
        self.column_names = ['turn', 'dialogue', 'label']

    def load_data(self):
        if self.mode == "train":
            file_name = "train.tsv"
        elif self.mode == "dev":
            file_name = "clustering_dev.tsv"
        elif self.mode == "test":
            file_name = "clustering_test.tsv"
            
        data = pd.read_csv(f'{self.path}{self.datasetname}/{file_name}', sep='\t', header=None, names=self.column_names)
        data['turn'] = data['turn'].astype('str')
        return data

    def add_qa_turn(self, dialogues):
        all_qa_turn = []
        for dial in dialogues:
            turns = dial.split("#")
            qa_turn = []
            for i, turn in enumerate(turns):
                if i == 0:  # 첫 번째 턴인 경우
                    if '?' in turn:
                        qa_turn.append('1')
                    else:
                        qa_turn.append('2')
                elif i == len(turns) - 1:  # 마지막 턴인 경우
                    if '?' in turns[i - 1]:
                        qa_turn.append('0')
                    else:
                        qa_turn.append('2')
                else:  # 중간에 있는 턴인 경우
                    if '?' in turns[i - 1] and '?' not in turn:  # 대답 턴 (0)
                        qa_turn.append('0')
                    elif '?' not in turns[i - 1] and '?' in turn:  # 질문 턴 (1)
                        qa_turn.append('1')
                    elif '?' in turns[i - 1] and '?' in turn:  # 질문이자 대답인 경우 아래와 같이 할당
                        qa_turn.append('0' if qa_turn[-1] == '1' else '1') # 수정
                    else:  # 질문도 대답도 아닌 턴 (2)
                        qa_turn.append('2')
            qa_turn = ''.join(qa_turn)
            all_qa_turn.append(qa_turn)
        return "|".join(all_qa_turn)

    def preprocess_data(self):
        train = self.load_data()
        train['dial_split'] = train['dialogue'].str.split('|').tolist()
        train['qa_turn'] = train['dial_split'].apply(self.add_qa_turn)
        train.drop(columns=['dial_split'], inplace=True)
        return train

    def save_data(self):
        data_qa = self.preprocess_data()
        
        if self.mode == "train":
            file_name = "train_qa_v2.tsv"
        elif self.mode == "dev":
            file_name = "clustering_dev_qa_v2.tsv"
        elif self.mode == "test":
            file_name = "clustering_test_qa_v2.tsv"
            
        data_qa.to_csv(f"{self.path}{self.datasetname}/{file_name}", sep="\t", index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset name and path.")
    parser.add_argument("--datasetname", type=str, help="Name of the dataset (mwoz, selfdialog, sgd).")
    parser.add_argument("--mode", type=str, help="Mode of the dataset (train, dev, test).")
    parser.add_argument("--path", type=str, help="Path to the dataset directory.")
    args = parser.parse_args()

    generate_qa = Generate_QA(args.datasetname, args.mode, args.path)
    generate_qa.save_data()

# python generate_qa_v2.py --datasetname sgd --mode train --path /home/jihyeon41/research_dial_embedding/dial2vec_git/dial2vec/datasets/






