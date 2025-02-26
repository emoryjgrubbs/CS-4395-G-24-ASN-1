import numpy as np
from time import perf_counter
from math import inf
from collections import deque
from os import path, remove


# compute n-gram probabilities
# k = int > 0, corput = file path to corpus
# return: [[n=1 probabilities list], [n=2], ..., [n=k]]
def compute_probabilities(max_k_gram, corpus, smoothing='add-k', smoothing_k=1, lambda_array=None):
    n_gram_counts = count_n_grams(max_k_gram, corpus)
    if smoothing.lower() == 'add-k' or smoothing.lower() == 'both':
        n_gram_counts = add_k_smoothing(max_k_gram, n_gram_counts, smoothing_k)
    probabilities = convert_to_probability(max_k_gram, n_gram_counts)
    if smoothing.lower() == 'linear-interpolation' or smoothing.lower() == 'both':
        probabilities = linear_interpolation_smoothing(max_k_gram, probabilities, lambda_array)
    return probabilities


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
    token = ''
    while file_index < eof:
        buffer = corpus.read(512)
        for buffer_index in range(len(buffer)):
            # for each word in the corpus add to its count
            if ((ord(buffer[buffer_index]) >= ord('\t') and
                ord(buffer[buffer_index]) <= ord('\r')) or
                    buffer[buffer_index] == ' '):
                if len(token) > 0:  # shouldn't be nessesary, but..
                    add_to_count(k, history, token, n_gram_counts)
                    history.append(token)
                    if len(history) >= k:
                        history.popleft()
                    token = ''
            else:
                token += buffer[buffer_index]
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


def linear_interpolation_smoothing(max_k_gram, probabilities, lambda_array):
    if len(lambda_array) < max_k_gram:
        return probabilities
    rec_linear_interpolation(max_k_gram, 0, probabilities, lambda_array, [])
    return probabilities


def rec_linear_interpolation(max_k_gram, depth, current_dict, lambda_array, probability_history):
    if depth >= max_k_gram:
        return
    for token, (probability, token_dict) in current_dict.items():
        # prevent probability_history refference from being changed
        new_history = probability_history + [probability]
        new_probability = 0
        lambda_sum = 0
        for i in range(depth+1):
            lambda_sum += lambda_array[i]
            # rounding to prevent float errors
            new_probability += round(lambda_array[i] * new_history[i], 15)
        new_probability = round(new_probability / lambda_sum, 15)
        current_dict.update({token: ((new_probability), token_dict)})
        rec_linear_interpolation(max_k_gram, depth+1, token_dict, lambda_array, new_history)


def preprocess(corpus):
    get_eof = open(corpus, "a")
    eof = get_eof.tell()
    get_eof.close()
    corpus = open(corpus, 'r')

    vocabulary = dict()
    build_vocabulary(corpus, vocabulary, eof)

    # set translation of words in vocabulary
    for word, count in vocabulary.items():
        if count > 2:
            vocabulary.update({word: word})
        else:
            vocabulary.update({word: '<UNK>'})

    # create a new temporary version of the corpus
    #   with the words translated
    corpus.seek(0)

    return translate_corpus(corpus, vocabulary, eof)


def build_vocabulary(corpus, vocabulary, eof):
    total = 0
    file_index = 0
    token = ''
    # count occurences of all words
    while file_index < eof:
        buffer = corpus.read(512)
        for buffer_index in range(len(buffer)):
            if ((ord(buffer[buffer_index]) >= ord('\t') and
                ord(buffer[buffer_index]) <= ord('\r')) or
                    buffer[buffer_index] == ' '):
                if len(token) > 0:
                    count = vocabulary.get(token)
                    if not count:
                        count = 0
                    vocabulary.update({token: count+1})

                    total += 1
                    token = ''
            else:
                # make all words same case
                token += buffer[buffer_index].upper()
        file_index += 512
    return total


def translate_corpus(corpus, vocabulary, eof):
    processed_corpus = open('alskdjf_temp_01928374.txt', 'w')
    file_index = 0
    output_buffer = ''
    token = ''
    # count occurences of all words
    while file_index < eof:
        input_buffer = corpus.read(512)
        for buffer_index in range(len(input_buffer)):
            if ((ord(input_buffer[buffer_index]) >= ord('\t') and
                ord(input_buffer[buffer_index]) <= ord('\r')) or
                    input_buffer[buffer_index] == ' '):
                if len(token) > 0:
                    output_buffer += ' ' + vocabulary.get(token)

                    token = ''
            else:
                # make all words same case
                token += input_buffer[buffer_index].upper()
        file_index += 512
        processed_corpus.write(output_buffer)
        output_buffer = ''
    corpus.close()

    processed_corpus.close()
    return 'alskdjf_temp_01928374.txt'


def compute_perplexity(k_gram, training_probabilities, text):
    l = compute_l(k_gram, training_probabilities, text)
    return np.exp2(-l)


def compute_l(k_gram, training_probabilities, text):
    # setup
    history = deque([])
    needed_history = k_gram - 1
    sum = 0
    num_k_grams = 0
    get_eof = open(text, "a")
    eof = get_eof.tell()
    get_eof.close()
    text = open(text, 'r')
    file_index = 0
    # read (512 bytes) characters from the text
    current_token = ''
    while file_index < eof:
        buffer = text.read(512)
        for buffer_index in range(len(buffer)):
            # for each word in the text, add it to the history/calculate probability
            if ((ord(buffer[buffer_index]) >= ord('\t') and
                ord(buffer[buffer_index]) <= ord('\r')) or
                    buffer[buffer_index] == ' '):
                if len(current_token) > 0:  # shouldn't be nessesary, but..
                    if needed_history <= 0:
                        dictionary = training_probabilities
                        for history_token in history:
                            history_value = dictionary.get(history_token.upper())
                            if history_value:
                                _, dictionary = history_value
                            else:
                                _, dictionary = dictionary.get('<UNK>')
                        current_value = dictionary.get(current_token.upper())
                        if current_value:
                            probability, _ = current_value
                        else:
                            probability, _ = dictionary.get('<UNK>')
                        sum += np.log2(probability)
                        num_k_grams += 1
                    history.append(current_token)
                    if len(history) >= k_gram:
                        history.popleft()
                    current_token = ''
                    needed_history -= 1
            else:
                current_token += buffer[buffer_index]
        file_index += 512
    text.close()
    return sum/num_k_grams


def test_uni_gram(corpus):
    k_gram = 1
    start_time = perf_counter()

    probabilities = compute_probabilities(k_gram, corpus)
    perplexity = compute_perplexity(k_gram, probabilities, 'val.txt')

    end_time = perf_counter()
    return (perplexity, end_time - start_time)


def test_add_k(corpus, smoothing_k):
    k_gram = 2
    start_time = perf_counter()

    probabilities = compute_probabilities(k_gram, corpus, smoothing='add-k',
                                          smoothing_k=smoothing_k)
    perplexity = compute_perplexity(k_gram, probabilities, 'val.txt')

    end_time = perf_counter()
    return (perplexity, end_time - start_time)


def test_linear_interpolation(corpus, lambda_array):
    k_gram = 2
    start_time = perf_counter()

    probabilities = compute_probabilities(k_gram, corpus,
                                          smoothing='linear-interpolation',
                                          lambda_array=lambda_array)
    perplexity = compute_perplexity(k_gram, probabilities, 'val.txt')

    end_time = perf_counter()
    return (perplexity, end_time - start_time)


def evaluate():  # may take upwards of an hour
    corpus = preprocess('train.txt')

    # test uni-grams
    sum_time = 0
    for _ in range(100):
        perplexity, time = test_uni_gram(corpus)
        sum_time += time
    print("UNI-GRAM PERPLEXITY:", perplexity)
    print("AVG TIME:", (sum_time/100))
    print("ELAPSED TIME:", sum_time)

    # test bi-grams
    # add-k smoothing
    sum_time = 0
    best_perplexity = inf
    for smoothing_k in np.arange(0.005, ((100 * 0.0001) + 0.005), 0.0001):
        perplexity, time = test_add_k(corpus, smoothing_k)
        sum_time += time
        if perplexity < best_perplexity:
            best_perplexity = perplexity
    print("\nBI-GRAM ( ADD-K:", smoothing_k, ") PERPLEXITY:", best_perplexity)
    print("AVG TIME:", (sum_time/100))
    print("ELAPSED TIME:", sum_time)

    # linear inperpolation
    sum_time = 0
    best_lambda = []
    best_perplexity = inf
    for uni_lambda in np.arange(0.4, ((100 * 0.001) + 0.4), 0.001):
        lambda_array = [uni_lambda, (1-uni_lambda)]
        perplexity, time = test_linear_interpolation(corpus, lambda_array)
        sum_time += time
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_lambda = lambda_array
    print("\nBI-GRAM ( LINEAR-INTERPOLATION:", best_lambda, ") PERPLEXITY:",
          best_perplexity)
    print("AVG TIME:", (sum_time/100))
    print("ELAPSED TIME:", sum_time)

    # clean up temp corpus file
    if path.exists('alskdjf_temp_01928374.txt'):
        remove('alskdjf_temp_01928374.txt')


def example():
    corpus = preprocess('train.txt')

    # test uni-grams
    perplexity, time = test_uni_gram(corpus)
    print("UNI-GRAM PERPLEXITY:", perplexity)
    print("ELAPSED TIME:", (time))

    # test bi-grams
    # add-k smoothing
    smoothing_k = 0.01
    perplexity, time = test_add_k(corpus, smoothing_k)
    print("\nBI-GRAM ( ADD-K:", smoothing_k, ") PERPLEXITY:", perplexity)
    print("ELAPSED TIME:", (time))

    # linear inperpolation
    lambda_array = [0.445, 0.555]

    perplexity, time = test_linear_interpolation(corpus, lambda_array)
    print("\nBI-GRAM ( LINEAR-INTERPOLATION:", lambda_array, ") PERPLEXITY:",
          perplexity)
    print("ELAPSED TIME:", (time))

    # clean up temp corpus file
    if path.exists('alskdjf_temp_01928374.txt'):
        remove('alskdjf_temp_01928374.txt')


example()
# evaluate()
