# requirement.txt
# ragas
# sentence-transformers

from transformers import pipeline

# from langchain_huggingface.llms import HuggingFacePipeline as LcHfPipeline
from langchain import HuggingFacePipeline as LcHfPipeline
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerSimilarity, AnswerRelevancy, Faithfulness
from ragas.embeddings import HuggingfaceEmbeddings as RagasHfEmbeddings

from datasets import Dataset, load_dataset

import json
from tqdm import tqdm


class RagasEvaluator():
    """
    ***NOT WORKING***, can return errors.

    A simple Evaluate class using Ragas packages.

    Args:
        test_data_file (str): The path to test data CSV file.

    Methods:
        evaluate().
    """
    def __init__(self, test_data_file: str):
        self.test_data_file = test_data_file
        self.dataset = self.get_dataset(dict(test=test_data_file))
        
    
    def evaluate(self, judge_model, judge_tokenizer, answers) -> dict:
        ds = Dataset.from_dict({
                "question": self.dataset['test']['Question'],
                "answer": answers,
                "contexts": self.dataset['test']['Context'],
                "ground_truth": self.dataset['test']['Answer'],
        })
        
        pipe = LcHfPipeline(pipeline=pipeline(
            model=judge_model,
            tokenizer=judge_tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            max_new_tokens=512,
            temperature=0.1, 
            repetition_penalty=1.1,
            eos_token_id=judge_tokenizer.eos_token_id,
            pad_token_id=judge_tokenizer.eos_token_id,
        ))
        vllm = LangchainLLMWrapper(pipe)
        embed = RagasHfEmbeddings(model_name='BAAI/bge-m3')
        metrics = [
            AnswerSimilarity(embeddings=embed, llm=vllm),
#             AnswerRelevancy(embeddings=embed, llm=vllm),
#             Faithfulness(llm=vllm),
        ]
        score = evaluate(ds, metrics=metrics, raise_exceptions=True)
        
        return score
            
    
    def get_inference_answers(self, generator_pipeline):
        answers = []
        for chat in tqdm(self.dataset['test']['text']):
            answers.append(generator_pipeline(chat)[-1]['generated_text'])
        return answers
        # return generator_pipeline(self.dataset['test']['text'])  # [[{'generated_text': ...}],]
    
    def get_dataset(self, data_file: dict):
        dataset = load_dataset('csv', data_files=data_file)
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