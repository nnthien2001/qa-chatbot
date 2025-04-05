from transformers import LlamaForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset, load_dataset

from huggingface_hub import login

import os
from datetime import datetime
from tqdm import tqdm


class FinetuneEngine():
    """
    A simple trainer class to finetune LoRA adapter of 
    a base LLM using trl.SFTTrainer.

    Args:
        base_model_dir (str): The path (or Huggingface link) to base LLM file.
        adapter_dir (str): The path to LoRA adapter weight (if any). Ignored if init_adapter=True.
        train_data_file (str): The path to train data CSV file.
        test_data_file (str): The path to test data CSV file.
        save_model_dir (str): The path to save LoRA weight after train/finetune.
        device (str): Device to used for training (default: 'cuda').
        init_adapter (bool): Flag to init new LoRA. Overwrite 'adapter_dir'.

    Methods:
        train(): Start training loop using SFTTrainer.
    """
    def __init__(
        self,
        base_model_dir: str,
        adapter_dir: str,
        train_data_file: str,
        test_data_file: str,
        save_model_dir: str,
        device: str = 'cuda',
        init_adapter: bool = False,
    ) -> None:
        
        self.base_dir = base_model_dir
        self.adapter_dir = (adapter_dir if not init_adapter else None)
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.save_model_dir = save_model_dir
        self.device = device
        
        self.base_model, self.tokenizer = self.get_base_model(self.base_dir)
        self.train_dataset = self.get_train_dataset(self.train_data_file)
        self.output_dir = None
        
    
    def train(self, adapter_dir = None, init_adapter = False):
        if init_adapter:
            self.adapter_dir = None
        elif adapter_dir is not None:
            self.adapter_dir = adapter_dir
        peft_model = self.get_peft_model(self.adapter_dir)
        
        trainer =  self.get_default_sfttrainer(peft_model)
        trainer.train()
        
    
    ## ----------------------------------------------------------
    # load func
    def get_base_model(self, model_dir: str):
        """
        Load base casual LLM via Huggingface (and set pad_token_id = eos_token_id)

        Args:
            model_dir (str): The path (or Huggingface link) to base LLM file.

        Returns:
            Tuple(LlamaForCausalLM, AutoTokenizer).
        """
        model = LlamaForCausalLM.from_pretrained(model_dir, device=self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, device=self.device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.pad_token =  "<|end_of_text|>"

        return model, tokenizer

    
    def get_peft_model(self, adapter_dir: str = None):
        """
        Load LoRA weights of base LLM. If not provided, initiate a default LoRA with randomized weights.

        Args:
            adapter_dir (str): The path (or Huggingface link) to LoRA file (default: None).

        Returns:
            PeftModel: Peft with LoRA combined with base LLM.
        """
        if adapter_dir is None:       
            lora_config = LoraConfig(
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0.05, # [0, 0.05]
                bias = "none",
                task_type=TaskType.CAUSAL_LM,
            )
            peft_model = get_peft_model(self.base_model, lora_config)
            # self.adapter_dir = os.path.join(self.save_model_dir, 'init')
            # peft_model.save_pretrained(self.old_adapter_dir)
        else:
            # lora_config = PeftConfig.from_pretrained(adapter_dir)
            peft_model = PeftModel.from_pretrained(self.base_model, adapter_dir, is_trainable=True)
        
        return peft_model
    
    
    # map train dataset
    def get_train_dataset(self, data_file: str):
        dataset = load_dataset('csv', data_files=dict(train=data_file))
        chat_template = """Bạn được cung cấp cho một ngữ cảnh. 
Chỉ dựa vào ngữ cảnh ấy, hãy trả lời cho câu hỏi bên dưới. Tuyệt đối không sử dụng thông tin bên ngoài, không có trong ngữ cảnh.

### Ngữ cảnh:
{CONTEXT}

### Câu hỏi:
{QUESTION}

### Trả lời:
{ANSWER}"""
        dataset = dataset.map(
            lambda x: {'text': chat_template.format(
                CONTEXT = x['Context'], 
                QUESTION = x['Question'], 
                ANSWER = x['Answer']
        )})
        return dataset
    
    
    # trainer
    def get_default_sfttrainer(self, peft_model):
        """
        Init a preset SFTTrainer.

        Args:
            peft_model (PeftModel).

        Return:
            SFTTrainer.
        """
        # def formatting_prompts_func(example):
        #     output_texts = []
        #     for i in range(len(example['instruction'])):
        #         text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        #         output_texts.append(text)
        #     return output_texts

        response_template = "\n### Trả lời:\n"
        response_template_ids = self.tokenizer.encode(response_template, add_special_tokens=False)[1:]
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=self.tokenizer)
        # max_seq_len = self.tokenizer.model_max_length  # =131072
        max_seq_len = 2048
        self.output_dir = os.path.join(self.save_model_dir, 'run-' + datetime.now().strftime('%m%d-%H%M%S'))
        
        trainer = SFTTrainer(
            model = peft_model,
            tokenizer = self.tokenizer,
            # peft_config = lora_config,
            train_dataset = self.train_dataset['train'],
            dataset_text_field = "text",  # to create ConstantLengthDataset here
            max_seq_length = max_seq_len,
            # formatting_func=formatting_prompts_func,
            data_collator=collator,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps=4,
                num_train_epochs = 1,
                learning_rate = 2e-4,
                logging_steps = 5,
                optim = "adamw_torch",
                weight_decay = 0.01,
                warmup_steps = 10,
                output_dir = self.output_dir,
                # save_strategy='steps',
                # save_steps=0.1,
                # save_total_limit=2,
                # metric_for_best_model='loss',
                # load_best_model_at_end=True,
                report_to='none',
            ),
        )
        
        return trainer