def chunk_dataset(tok, dataset, text_column: str, seq_len: int, n_samples: int):
    """
    Tokenize examples from the dataset, concatenate into a flat token stream,
    and slice into non-overlapping seq_len chunks.
    Returns a list of n_samples lists of token ids.
    """
    all_ids = []
    need    = n_samples * seq_len

    for ex in dataset:
        text = ex[text_column].strip() if isinstance(ex[text_column], str) else ""
        if not text:
            continue
        all_ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        if len(all_ids) >= need:
            break

    chunks = [
        all_ids[i : i + seq_len]
        for i in range(0, len(all_ids) - seq_len + 1, seq_len)
    ]
    return chunks[:n_samples]
