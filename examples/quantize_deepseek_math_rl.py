from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--num_calibration_tokens', type=int, default=1_000)
    parser.add_argument('--backend_layout', type=str, default='vllm_gptq')
    return parser.parse_args()


def main():
    # Parsing command-line arguments
    args = parse_args()

    # Loading model and tokenizer
    model_id = 'deepseek-ai/deepseek-math-7b-rl'
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side='left', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map='auto', dtype='auto')

    # Loading dataset for calibration
    def tokenize(example, with_completion=True):
        if with_completion:
            messages = [
                {'role': 'user', 'content': example['problem']},
                {'role': 'assistant', 'content': example['expected_answer']},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        else:
            messages = [{'role': 'user', 'content': example['problem']}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors='pt').input_ids[0]
    dataset = load_dataset('nvidia/OpenMathReasoning', split='additional_problems')
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Quantizing the model
    layers_to_quantize = ['*experts*'] # Quantizing experts only
    forge.gptq_quantize(
        model,
        dataset,
        bits=args.bits,
        group_size=args.group_size,
        num_calibration_tokens=args.num_calibration_tokens,
        backend_layout=args.backend_layout,
        layers_to_quantize=layers_to_quantize,
        output_checkpoint_dir='quantized-deepseek-math-7b-rl',
    )

if __name__ == "__main__":
    main()
