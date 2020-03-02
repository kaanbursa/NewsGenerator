
# split by new lines and spaces
news_split = all_text.split('\n\n')
all_text = ' '.join(news_split)

# create a list of words
words = all_text.split()

# feel free to use this import
from collections import Counter

## Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
int_to_vocab = {word: ii for ii, word in vocab_to_int.items()}



woc_int = []
for w in words:
    woc_int.append([vocab_to_int[word] for word in w.split()])



def pad_features(woc_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(woc_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(woc_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features
