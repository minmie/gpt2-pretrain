from transformers import GPT2Tokenizer,GPT2LMHeadModel, set_seed, AutoTokenizer
from tokenizers import decoders
set_seed(42)

model_path = '/home/chenjq/pythonWork/nlp/train_new_gpt2/sp-tok-v5'
tokenizer1 = AutoTokenizer.from_pretrained(model_path)
tokenizer1.tokenize("北京天气真好。I am happy")
model_path = "/home/chenjq/model/gpt2"


tokenizer2 = AutoTokenizer.from_pretrained(model_path)

model_path = "/home/chenjq/model/m3e-base/"


tokenizer3 = AutoTokenizer.from_pretrained(model_path)
print(tokenizer1.decode(tokenizer1.encode("北京天气真好。I am happy")))
print(tokenizer2.decode(tokenizer2.encode("北京天气真好。I am happy")))
print(tokenizer3.decode(tokenizer3.encode("北京天气真好。I am happy")))
print(111)
