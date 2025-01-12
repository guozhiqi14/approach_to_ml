# config.py
import transformers

# this is the maximum number of tokens in the sentence
# MAX_LEN = 512
MAX_LEN = 256

# batch sizes is small because model is huge!
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

# let's train for a maximum of 10 epochs
EPOCHS = 1

# define path to BERT model files
# BERT_PATH = "../input/bert_base_uncased/"
BERT_PATH = "bert-base-uncased"

# this is where you want to save the model
MODEL_PATH = "../models/model.bin"

# training file
TRAINING_FILE = "../input/IMDB_Dataset.csv"

# define the tokenizer
# we use tokenizer and model
# from huggingface's transformers
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True)


