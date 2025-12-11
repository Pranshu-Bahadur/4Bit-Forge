from argparse import ArgumentParser
from datasets import load_dataset
from transformers import DeepseekV3ForCausalLM, DeepseekV3Config, AutoTokenizer

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

    # Loading tokenizer and macking a small mock model
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-Math-V2', use_fast=True, padding_side='left', trust_remote_code=True)
    model = DeepseekV3ForCausalLM(
        DeepseekV3Config(
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=4,
            first_k_dense_replace=2
        )
    )

    # Loading dataset for calibration
    def tokenize(example, with_completion=True):
        messages = [
            {'role': 'user', 'content': example['text']},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        return tokenizer(text, return_tensors='pt')
    dataset = load_dataset('Salesforce/wikitext', name='wikitext-2-v1', split='test')
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Quantizing the model
    layers_to_quantize = ['*mlp.experts*'] # Quantizing routed experts only
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
