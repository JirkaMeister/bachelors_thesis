from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling as DCLM
from transformers import Trainer, TrainingArguments
import torch

# Since there are some empty sstrings in the dataset, this function will return the first non-empty example
def get_non_empty_example(dataset):
    for example in dataset['train']:
        if len(example['text'].split()) > 5:
            return example

def print_masked_example(dataset, tokenizer, data_collator):
    example = get_non_empty_example(dataset)
    print(f"This is an example in the string format:\n {example['text']}")

    # This will transform the example string into list of tokens (words)
    tokenized_example = tokenizer(example['text'], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # This will mask some of the tokens in the example (some words will be replaced with [MASK]), the tokens will be converted to ids
    masked_example = data_collator([tokenized_example])

    # This will convert the token ids back to tokens (words)
    input_ids_list = masked_example['input_ids'].squeeze().tolist()
    print(f"This is the example after tokenization:\n {tokenizer.convert_ids_to_tokens(input_ids_list)}")

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
#dataset = load_dataset("glue", "sst2")
#dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the texts
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DCLM(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Show an example of training data after before and after tokenization
print_masked_example(dataset, tokenizer, data_collator)

# Load the BERT model
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Define the training arguments
training_args = TrainingArguments(
    output_dir = "./wikitext-2-mlm",
    evaluation_strategy = "epoch",
    overwrite_output_dir = True,
    num_train_epochs = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    save_steps = 10_000,
    save_total_limit = 2,
)

# Define the trainer
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Test the model on a masked sentence
sentences = [
    "The capital of France is [MASK].",
    "The capital of Italy is [MASK].",
    "The capital of Spain is [MASK].",
    "The quick brown [MASK] jumps over the lazy [MASK].",
]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors = "pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_logits = model(**inputs).logits[0, mask_token_index.item()]
    mask_token_probs = torch.softmax(mask_token_logits, dim=0)
    predicted_token_index = torch.argmax(mask_token_logits).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_index])[0]
    print(f"Original sentence: {sentence}")
    print(f"Predicted token: {predicted_token}")
    print()
