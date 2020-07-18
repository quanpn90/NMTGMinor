import numpy as np


# from .fast_extensions import


def _is_oversized(cur_batch, new_sent_size, cur_batch_sizes, batch_size_words, batch_size_sents):
    # cur_batch_size = sum(cur_batch_sizes)

    if len(cur_batch) == 0:
        return False

    if len(cur_batch) >= batch_size_sents:
        return True

    if max(max(cur_batch_sizes), new_sent_size) * (len(cur_batch) + 1) > batch_size_words:
        return True

    return False


def allocate_batch_slow(indices, lengths,
                        src_sizes, tgt_sizes,
                        batch_size_words, batch_size_sents, batch_size_multiplier,
                        max_src_len, max_tgt_len, cleaning=True):
    batches = list()
    batch = list()
    cur_batch_size = 0
    cur_batch_sizes = []

    idx = 0
    full_size = len(indices)

    while idx < full_size:
        i = indices[idx]

        sent_length = lengths[i]
        src_size = src_sizes[i] if src_sizes is not None else 0
        tgt_size = tgt_sizes[i] if tgt_sizes is not None else 0

        if cleaning:
            if not (0 < src_size < max_src_len and 2 < tgt_size < max_tgt_len):
                idx = idx + 1
                continue

        oversized = _is_oversized(batch, sent_length, cur_batch_sizes, batch_size_words, batch_size_sents)

        if oversized:
            current_size = len(batch)
            scaled_size = max(
                batch_size_multiplier * (current_size // batch_size_multiplier),
                current_size % batch_size_multiplier)

            batch_ = batch[:scaled_size]
            batches.append(batch_)  # add this batch into the batch list
            batch = batch[scaled_size:]  # reset the current batch
            cur_batch_sizes = cur_batch_sizes[scaled_size:]
            cur_batch_size = sum(cur_batch_sizes)

        batch.append(i)
        cur_batch_size += sent_length
        cur_batch_sizes.append(sent_length)

        idx = idx + 1

    if len(batch) > 0:
        batches.append(batch)

    return batches


def allocate_batch(indices, lengths,
                   src_sizes, tgt_sizes,
                   batch_size_words, batch_size_sents, batch_size_multiplier,
                   max_src_len, max_tgt_len, cleaning=True):

    try:
        import pyximport
        cython_available = True
    except ModuleNotFoundError as e:
        cython_available = False

    if not cython_available or (tgt_sizes is None or src_sizes is None):
        return allocate_batch_slow(indices, lengths, src_sizes, tgt_sizes,
                                   batch_size_words, batch_size_sents, batch_size_multiplier,
                                   max_src_len, max_tgt_len, cleaning)

    pyximport.install(setup_args={"include_dirs": np.get_include()},
                      inplace=True)
    from .fast_extensions import fast_batch_allocate

    cleaning = int(cleaning)

    if isinstance(indices, list):
        indices = np.asarray(indices)
    # convert to np int64

    return fast_batch_allocate(indices, lengths,
                               src_sizes, tgt_sizes,
                               batch_size_words, batch_size_sents, batch_size_multiplier,
                               max_src_len, max_tgt_len, cleaning)
