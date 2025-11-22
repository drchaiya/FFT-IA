def get_all_butterfly_indices(seq_len):
    """Returns list of (stage, list_of_pairs) for full factorization"""
    if not (seq_len & (seq_len - 1) == 0):
        raise ValueError("seq_len must be power of 2")
    stages = []
    for stage in range(seq_len.bit_length() - 1):
        pairs = []
        stride = 1 << stage
        for i in range(0, seq_len, stride * 2):
            for j in range(stride):
                a = i + j
                b = i + j + stride
                if b < seq_len:
                    pairs.append((a, b))
        stages.append(pairs)
    return stages
