# compute n-gram probabilities
# return: [[n=1 probabilities list], [n=2], ..., [n=k]]
def compute_probabilities(k, corpus):
    # setup
    history = []  # check if history is long enough to compute the n-gram
    probabilities = []
    n_gram_counts = []
    total_counts = []
    for _ in range(k):
        probabilities.append([])
        n_gram_counts.append([])
        total_counts.append(0)
    # read a chunk (512 bytes)
    get_eof = open(corpus, "a")
    eof = get_eof.tell()
    get_eof.close()
    corpus = open(corpus, 'r')
    file_index = 0
    while file_index < eof:
        buffer = corpus.read(512)
        current_token = ''
        for buffer_index in range(len(buffer)):
            # for each word in the chunk add to its count
            if buffer[buffer_index] == ' ':
                print(current_token)
                current_token = ''
            else:
                current_token += buffer[buffer_index]
        file_index += 512
    print(current_token)

    corpus.close()
    return probabilities


compute_probabilities(3, 'train.txt')
