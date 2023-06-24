import tiktoken 
from byte_pair_encoder.bpe import BytePairEncoder

# Your BPE
bpe = BytePairEncoder()
vocab = bpe.train('Here we go! Ale, ale, ale! Go, go, go! Ale, ale, ale!')
print(vocab)
text, encoding = bpe.encode('a large whale logo')

# OpenAI's BPE
print('='*100)
print('OpenAI BPE encoding')
print('='*100)
encoder = tiktoken.get_encoding("gpt2")
openai_encoding = encoder.encode_ordinary('a large whale logo')
openai_text = [encoder.decode([id]) for id in openai_encoding]
print(openai_text)
print(openai_encoding)