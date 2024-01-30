from transformers import GPT2Tokenizer,GPT2LMHeadModel, set_seed, AutoTokenizer
set_seed(42)

model_path = '/home/chenjq/pythonWork/nlp/train_new_gpt2/tmp/test-clm-sp-v5/checkpoint-4000'
# model_path = "/home/chenjq/model/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(model_path,pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
# input_ids = tokenizer.encode('张齐娥（Claire Chiang，10月4日），祖籍海南琼海，出生于新加坡。她是', return_tensors='pt', add_special_tokens=False)
input_ids = tokenizer.encode('<bos>美国', return_tensors='pt', add_special_tokens=False)

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


# activate beam search and early_stopping
beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))



# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))



# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))




# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))




# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=0,
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))





# set top_k to 50
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50
)

print("top_k Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))





# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids,
    num_beams=5,
    do_sample=True,
    max_length=50,
    top_p=0.92,
    top_k=6,
    temperature=0.7
)

print("top_p Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=False))

