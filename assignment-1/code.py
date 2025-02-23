from collections import deque


# compute n-gram probabilities
# k = int > 0, corput = file path to corpus
# return: [[n=1 probabilities list], [n=2], ..., [n=k]]
def compute_probabilities(k, corpus):
    # setup
    history = deque([])
    n_gram_counts = []
    total_counts = []
    for _ in range(k):
        n_gram_counts.append(dict())
        total_counts.append(0)
    get_eof = open(corpus, "a")
    eof = get_eof.tell()
    get_eof.close()
    corpus = open(corpus, 'r')
    file_index = 0
    # read (512 bytes) characters from the corpus
    while file_index < eof:
        buffer = corpus.read(512)
        current_token = ''
        for buffer_index in range(len(buffer)):
            # for each word in the corpus add to its count
            if buffer[buffer_index] == ' ':
                n_gram = current_token
                current_count = n_gram_counts[0].get(n_gram)
                if not current_count:
                    current_count = 0
                n_gram_counts[0].update({n_gram: current_count+1})
                for j in range(1, k):
                    if len(history) - j >= 0:
                        n_gram += ' ' + history[len(history)-j]
                        current_count = n_gram_counts[j].get(n_gram)
                        if not current_count:
                            current_count = 0
                        n_gram_counts[j].update({n_gram: current_count+1})
                history.append(current_token)
                if len(history) >= k:
                    history.popleft()
                current_token = ''
            else:
                current_token += buffer[buffer_index]
        file_index += 512
    corpus.close()
    probabilities = n_gram_counts
    return probabilities


probabilities = compute_probabilities(3, 'train.txt')
print(probabilities)
