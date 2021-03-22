import numpy as np

NUCLEOTIDES = 4

# Process next CLAMP sequence data: get sequence and binding affinity
# A CLAMP line looks like: SCORE SEQUQNCE
def process_clamp(clamp_file):   
    data = clamp_file.readline()
    if not data:
        return None
    line = data.strip().split()
    score = float(line[0])
    seq = line[1]
    return (seq, score)


def read_sequence(sequences_path, max_seq_len):
    with open(sequences_path, 'r') as sequences:
        data = list()
        lengths = list()
        labels = list()
        counter = 0
        while True:
            counter += 1
            seq_data = process_clamp(sequences)
            if not seq_data:
                return np.array(data), np.array(lengths), np.array(labels)
            # Compute a matrix of SEQ_LEN X RNA_ALPHABET for decoding the sequence bases
            labels.append(seq_data[1])
            seq_matrix = list()
            #print(seq_data[0].upper())

            for base in seq_data[0].upper():
                if base == 'A':
                    base_encoding = [1, 0, 0, 0]
                elif base == 'C':
                    base_encoding = [0, 1, 0, 0]
                elif base == 'G':
                    base_encoding = [0, 0, 1, 0]
                elif base == 'U':
                    base_encoding = [0, 0, 0, 1]
                else:
                    raise ValueError
                seq_matrix.append(base_encoding)
            seq_matrix = np.array(seq_matrix)

            curr_seq_len = seq_matrix.shape[0]
            lengths.append(curr_seq_len)
            padd_len = max_seq_len - curr_seq_len
            assert (padd_len  >= 0)
            if padd_len > 0:
                padding_matrix = np.zeros((padd_len, NUCLEOTIDES))
                seq_matrix = np.concatenate((seq_matrix, padding_matrix), axis=0)
            data.append(seq_matrix)




