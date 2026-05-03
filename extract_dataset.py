def _is_wiki_header(s: str) -> bool:
    s = s.strip()
    return s.startswith("=") and s.endswith("=")

def chunk_dataset(tok, dataset, text_column, seq_len: int, n_samples: int):
    """
    Tokenize examples from the dataset, concatenate into a flat token stream,
    and slice into non-overlapping seq_len chunks.
    Returns a list of n_samples lists of token ids.

    text_column: str, or list of str to concatenate with a newline separator.
    """
    columns   = [text_column] if isinstance(text_column, str) else text_column
    separator = " "
    all_ids   = []
    need      = n_samples * seq_len

    for ex in dataset:
        parts = [ex[col].strip() for col in columns if isinstance(ex.get(col), str)]
        text  = separator.join(p for p in parts if p and not _is_wiki_header(p))
        if not text:
            continue
        all_ids.extend(tok(text, add_special_tokens=False, truncation=False)["input_ids"])
        if len(all_ids) >= need:
            break

    chunks = [
        all_ids[i : i + seq_len]
        for i in range(0, len(all_ids) - seq_len + 1, seq_len)
    ]
    return chunks[:n_samples]
