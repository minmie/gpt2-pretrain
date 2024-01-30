from transformers import AutoTokenizer
from datasets import load_dataset


path = r'/tmp/pycharm_project_806/LCSTS_new/train.json'  # a chinese text dataset
raw_data = load_dataset("json", data_files=path, split='train')

training_corpus = (
    raw_data[i : i + 1000]["content"]
    for i in range(0, len(raw_data), 1000)
)

old_tokenizer = AutoTokenizer.from_pretrained("/home/chenjq/model/gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

example = '就是去美国大使馆的官方网站，它有中文版，去把每一条仔细研究透了，把每一个表格和材料都准备好了'  # chinese text
old_tokens = old_tokenizer.tokenize(example)
print('old_tokens:',old_tokens)

new_tokens = tokenizer.tokenize(example)
print('new_tokens',new_tokens)
tokenizer.save_pretrained("./my-tok")
