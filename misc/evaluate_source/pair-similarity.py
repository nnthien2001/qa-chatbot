from transformers import TextGenerationPipeline

from datasets import Dataset, load_dataset

from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer

import torch
from torch.nn import functional as F
from huggingface_hub import login

import os
from tqdm import tqdm


class PairSimilarityCalc():
    def __init__(self):
        self.m3 = BGEM3FlagModel('BAAI/bge-m3')
        self.vi = SentenceTransformer('dangvantuan/vietnamese-embedding-LongContext', trust_remote_code=True)

    def calc_similarity(self, sentence_1: str, sentence_2:str):
        pair = [sentence_1, sentence_2]
        score_m3 = self.m3.compute_score(pair, max_passage_length=256, weights_for_different_modes=[1.0, 0.2, 1.0])
        
        embed_vi = self.vi.encode(pair, show_progress_bar=False)
        score_vi = F.cosine_similarity(torch.tensor(embed_vi[0]), torch.tensor(embed_vi[1]), dim=0).item()
        
        return (score_m3['dense'], score_m3['colbert+sparse+dense'], score_vi)
        

class PairSimilarityEvaluator():
    """
    A simple Evaluate class using PairSimilarity metric.

    Args:
        test_data_file (str): The path to test data CSV file.

    Methods:
        evaluate().
    """
    def __init__(self, test_data_file: str):
        self.test_data_file = test_data_file
        self.dataset = self.get_dataset(test_data_file)
        self.sim_calculator = PairSimilarityCalc()
        
    
    def evaluate(self, answers) -> dict:
        ds = Dataset.from_dict({
                # "question": self.dataset['test']['Question'],
                "answer": answers,
                # "contexts": self.dataset['test']['Context'],
                "ground_truth": self.dataset['test']['Answer'],
        })
        
        # log = []
        acc_score = 0
        for sample in tqdm(ds, desc='Calculating similarity'):
            sample_score = self.sim_calculator.calc_similarity(sample['answer'], sample['ground_truth'])
            avg = (min(sample_score[0], sample_score[1]) + sample_score[2]) / 2
            # log.append(dict(...))
            acc_score += avg
        
        return acc_score / len(ds)
            
    
    def get_inference_answers(self, generator_pipeline: TextGenerationPipeline):
        answers = []
        for chat in tqdm(self.dataset['test']['text'], desc='Inferencing'):
            answers.append(generator_pipeline(chat)[-1]['generated_text'])
        return answers
        # return generator_pipeline(self.dataset['test']['text'])  # [[{'generated_text': ...}],]
    
    def get_dataset(self, data_file: str):
        dataset = load_dataset('csv', data_files=dict(test=data_file))
        chat_template = """Bạn được cung cấp cho một ngữ cảnh. 
Chỉ dựa vào ngữ cảnh ấy, hãy trả lời cho câu hỏi bên dưới. Tuyệt đối không sử dụng thông tin bên ngoài, không có trong ngữ cảnh.

### Ngữ cảnh:
{CONTEXT}

### Câu hỏi:
{QUESTION}

### Trả lời:
"""
        dataset = dataset.map(
            lambda x: {'text': chat_template.format(
                    CONTEXT = x['Context'], 
                    QUESTION = x['Question']),}
        )
        dataset = dataset.map(lambda x: {'Context': [x['Context']]})
#         dataset = dataset.map(lambda x: {'Answer': [x['Answer']]})
        
        return dataset