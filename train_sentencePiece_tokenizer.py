from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset
from tokenizers import Regex
# path = r'wiki.json'  # a chinese text dataset
path = r'all_train.json'  # a chinese text dataset
# path = r'cluener.jsonl'  # a chinese text dataset
# path = r'/tmp/pycharm_project_806/cluener.json'  # a chinese text dataset
raw_data = load_dataset("json", data_files=path, split='train')
# raw_data = raw_data.select(range(10000))
training_corpus = (
    raw_data[i : i + 1000]["content"]
    for i in range(0, len(raw_data), 1000)
)


tokenizer = Tokenizer(models.Unigram())

# NLG不应当加入 normalizers.Lowercase()，因为在decode的时候，就无法生成大写的了
# 在bert等NLU模型中，可以加入 normalizers.Lowercase()，因为NLU一般不用于文本生成，而是用于文本理解（如文本分类，实体抽取），
# 这种情况下其实大写小写无所谓
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(add_prefix_space=False)

print(tokenizer.pre_tokenizer.pre_tokenize_str("北京是中国的首都，今天天气真好。Let's test this tokenizer."))
print(1)

special_tokens = ["<bos>","<eos>", '<sep>'] + [f'<unused{i}>' for i in range(50)]
trainer = trainers.UnigramTrainer(
    vocab_size=52000, special_tokens=special_tokens, unk_token="<unk>",max_piece_length=4,
)
tokenizer.train_from_iterator(training_corpus, trainer=trainer)

encoding = tokenizer.encode("北京是中国的首都，今天天气真好。Let's test this tokenizer.")
print(encoding.tokens)

bos_token_id = tokenizer.token_to_id("<bos>")
eos_token_id = tokenizer.token_to_id("<eos>")
sep_token_id = tokenizer.token_to_id("<sep>")


tokenizer.post_processor = processors.TemplateProcessing(
    single=f"<bos>:0 $A:0 <eos>:0",
    pair=f"<bos>:0 $A:0 <sep>:0 $B:1 <eos>:1",
    special_tokens=[("<bos>", bos_token_id), ("<eos>", eos_token_id), ("<sep>", sep_token_id)],
)

encoding = tokenizer.encode("北京是中国的首都，今天天气真好。Let's test this tokenizer.")
print(encoding.tokens)

encoding = tokenizer.encode("北京是中国的首都，今天天气真好。Let's test this tokenizer." ,'i am happy.')
print(encoding.tokens)

print(tokenizer.decode(encoding.ids))

tokenizer.decoder = decoders.Metaspace(add_prefix_space=False)

print(tokenizer.decode(encoding.ids))


from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<bos>",
    eos_token="<eos>",
    sep_token="<sep>",
)
wrapped_tokenizer.save_pretrained('./sp-tok-v5')

print(wrapped_tokenizer.tokenize("北京是中国的首都，今天天气真好。"))