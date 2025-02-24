from collections import deque


# compute n-gram probabilities
# k = int > 0, corput = file path to corpus
# return: [[n=1 probabilities list], [n=2], ..., [n=k]]
def compute_probabilities(k, corpus, smoothing=False):
    corpus = preprocess(corpus)
    n_gram_counts = count_n_grams(k, corpus)
    if smoothing:
        n_gram_counts = smooth(n_gram_counts)
    probabilities = convert_to_probability(k, n_gram_counts)
    return probabilities


def preprocess(corpus):
    return corpus


def count_n_grams(k, corpus):
    # setup
    history = deque([])
    n_gram_counts = dict()
    get_eof = open(corpus, "a")
    eof = get_eof.tell()
    get_eof.close()
    corpus = open(corpus, 'r')
    file_index = 0
    # read (512 bytes) characters from the corpus
    current_token = ''
    while file_index < eof:
        buffer = corpus.read(512)
        for buffer_index in range(len(buffer)):
            # for each word in the corpus add to its count
            if ((ord(buffer[buffer_index]) >= ord('\t') and
                ord(buffer[buffer_index]) <= ord('\r')) or
                    buffer[buffer_index] == ' '):
                if len(current_token) > 0:  # shouldn't be nessesary, but..
                    add_to_count(k, history, current_token, n_gram_counts)
                    history.append(current_token)
                    if len(history) >= k:
                        history.popleft()
                    current_token = ''
            else:
                current_token += buffer[buffer_index]
        file_index += 512
    corpus.close()
    return n_gram_counts


def add_to_count(k, history, current_token, n_gram_counts):
    # n-grams, the length of the history cannot be
    #   larger than highest requrested n-gram
    for m in range(len(history)):
        # reset starting point
        current_dict = n_gram_counts
        for n in range(m, len(history)):
            # get the nth element of the history from this dictionary
            _, current_dict = current_dict.get(history[n])
        # add the current_token to this dictionary
        conditional_dict = current_dict
        value_tupel = conditional_dict.get(current_token)
        if value_tupel:
            current_count, current_dict = value_tupel
        else:
            current_count = 0
            current_dict = dict()
        conditional_dict.update({current_token:
                              (current_count+1, current_dict)})

    # unigram
    value_tupel = n_gram_counts.get(current_token)
    if value_tupel:
        current_count, current_dict = value_tupel
    else:
        current_count = 0
        current_dict = dict()
    n_gram_counts.update({current_token:
                          (current_count+1, current_dict)})


def smooth(n_gram_counts):
    return n_gram_counts


def convert_to_probability(k, n_gram_counts):
    probabilities = n_gram_counts
    rec_conver_to_prob(k, 0, probabilities)
    return probabilities


def rec_conver_to_prob(k, depth, current_dict):
    if depth >= k:
        return
    # get total number for level
    total = 0
    for (count, _) in current_dict.values():
        total += count
    # divide the counts by the total for that dictionary
    for token, (count, token_dict) in current_dict.items():
        current_dict.update({token: ((count/total), token_dict)})
        rec_conver_to_prob(k, depth+1, token_dict)
    return


probabilities = compute_probabilities(3, 'sentence.txt')
print(probabilities)
