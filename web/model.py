import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel #AdamW, get_linear_schedule_with_warmup
torch.set_grad_enabled(False)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', do_lower_case=True)
model = GPT2LMHeadModel.from_pretrained('gpt2-large').eval()


def extend(text, size=100):
    tokens = tokenizer.encode(text)
    predictions, past = torch.tensor([tokens]), None
    for i in range(size):
        predictions, past = model(predictions, past=past)
        predictions = torch.multinomial(F.softmax(predictions[:,-1]),1)
        tokens.append(predictions.item())
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    test_text = 'Today the weather is '
    extended_text = extend(test_text,size=100)
    print(extended_text)
