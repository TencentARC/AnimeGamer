from transformers import BertTokenizer


def bert_tokenizer(pretrained_model_name_or_path):
    # Create an instance of BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
    truncation_side='right')
    # Add a custom beginning-of-sequence token
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    # Return the configured tokenizer
    return tokenizer