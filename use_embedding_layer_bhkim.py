from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

texts = ['You are the Best', 'You are the Nice']

tokenizer = Tokenizer(num_words=10, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

binary_results = tokenizer.sequences_to_matrix(sequences, mode= 'binary')

print(tokenizer.word_index)
print('+++++++++++++++++')

print(f'sequences : {sequences} \n')
print(f'binary_vectors : \n {binary_results} \n')

print(to_categorical(sequences))

test_text = ['You are the One']
test_seq = tokenizer.texts_to_sequences(test_text)

print(f'test sequences : {test_seq}')
