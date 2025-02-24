from collections import deque


# compute n-gram probabilities
# k = int > 0, corput = file path to corpus
# return: [[n=1 probabilities list], [n=2], ..., [n=k]]
def compute_probabilities(max_k_gram, corpus, smoothing='Laplace', smoothing_k=1):
    corpus = preprocess(corpus)
    n_gram_counts = count_n_grams(max_k_gram, corpus)
    if smoothing.lower() == 'laplace':
        n_gram_counts = laplace_smooth(max_k_gram, n_gram_counts)
    if smoothing.lower() == 'add-k':
        n_gram_counts = add_k_smoothing(max_k_gram, n_gram_counts, smoothing_k)
    probabilities = convert_to_probability(max_k_gram, n_gram_counts)
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
    add_unencountered(k, n_gram_counts)
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


def add_unencountered(k, n_gram_counts):
    tokens = []
    # get all toplevel tokens (unigrams)
    for token in n_gram_counts:
        tokens.append(token)
    tokens.append('< UNK >')
    n_gram_counts.update({'< UNK >': (0, dict())})
    for (_, token_dict) in n_gram_counts.values():
        rec_add_unencountered(k, 1, token_dict, tokens)
    return


def rec_add_unencountered(k, depth, current_dict, tokens):
    if depth >= k:
        return
    for token in tokens:
        value_tupel = current_dict.get(token)
        if value_tupel:
            # recurse on its dictionary
            _, token_dict = value_tupel
        else:
            # create it and recurse
            current_dict.update({token: (0, dict())})
            _, token_dict = current_dict.get(token)
        rec_add_unencountered(k, depth+1, token_dict, tokens)
    return


def laplace_smooth(max_k_gram, n_gram_counts):
    rec_add_k(max_k_gram, 0, n_gram_counts, 1)
    return n_gram_counts


def add_k_smoothing(max_k_gram, n_gram_counts, k_smoothing):
    rec_add_k(max_k_gram, 0, n_gram_counts, k_smoothing)
    return n_gram_counts


def rec_add_k(max_k_gram, depth, current_dict, k_smoothing):
    if depth >= max_k_gram:
        return
    for token, (count, token_dict) in current_dict.items():
        current_dict.update({token: ((count+k_smoothing), token_dict)})
        rec_add_k(max_k_gram, depth+1, token_dict, k_smoothing)


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
    if total > 0:
        for token, (count, token_dict) in current_dict.items():
            current_dict.update({token: ((count/total), token_dict)})
            rec_conver_to_prob(k, depth+1, token_dict)


probabilities = compute_probabilities(2, 'sentence.txt')
print(probabilities)
