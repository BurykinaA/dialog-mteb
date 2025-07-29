from transformers import AutoTokenizer
from dataloader import get_dataloader

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['[USR]', '[SYS]']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# Get a small sample from dataloader
dataloader = get_dataloader('processed_dialogues.txt', tokenizer, batch_size=1, max_len=128)

# Get one batch
batch = next(iter(dataloader))

print("Batch keys:", batch.keys())
print("\nContext input ids shape:", batch['context_input_ids'].shape)
print("Full input ids shape:", batch['full_input_ids'].shape)

# Decode the sequences to see what they look like
context_tokens = tokenizer.decode(batch['context_input_ids'][0], skip_special_tokens=False)
full_tokens = tokenizer.decode(batch['full_input_ids'][0], skip_special_tokens=False)

print("\nContext sequence:")
print(context_tokens)
print("\nFull sequence:")
print(full_tokens)

print("\nContext attention mask:", batch['context_attention_mask'][0][:20])  # First 20 tokens
print("Full attention mask:", batch['full_attention_mask'][0][:20])  # First 20 tokens 