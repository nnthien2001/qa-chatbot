from transformers import pipeline

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
# from deepeval import evaluate
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from datasets import Dataset, load_dataset

import json
from tqdm import tqdm


class DeepEvalJudge(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> str:
        model = self.load_model()

        judge_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=2,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            judge_pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = judge_pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3.2-1B-Intruct"

    
    
class DeepEvalEvaluator():
    """
    ***NOT WORKING***, can return errors.

    A simple Evaluate class using DeepEval packages.

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
        judge = DeepEvalJudge(model=judge_model, tokenizer=judge_tokenizer)
        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=judge,
            include_reason=False,
            async_mode=True,
        )
        
        scores = []
        for sample in ds:
            test_case = LLMTestCase(
                input = sample['question'],
                actual_output = sample['answer']
            )
            scores.append(metric.measure(test_case).score)
        return scores
            
    
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
    
