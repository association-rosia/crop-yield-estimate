import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def _convert_text_to_tabular_data(text: tp.List[str], columns: tp.List[str]) -> pd.DataFrame:
    generated = []

    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns, "placeholder")

        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" is ")
            if values[0] in columns and td[values[0]] == "placeholder":
                try:
                    td[values[0]] = values[1]
                except IndexError:
                    pass
        generated.append(td)
    df_gen = pd.DataFrame(generated)
    df_gen.replace("None", None, inplace=True)

    return df_gen


def _encode_row_partial(row, shuffle=True):
    num_cols = len(row.index)
    if not shuffle:
        idx_list = np.arange(num_cols)
    else:
        idx_list = np.random.permutation(num_cols)

    lists = ", ".join(
        sum(
            [
                [f"{row.index[i]} is {row[row.index[i]]}"]
                if not pd.isna(row[row.index[i]])
                else []
                for i in idx_list
            ],
            [],
        )
    )
    return lists


def _get_random_missing(row):
    nans = list(row[pd.isna(row)].index)
    return np.random.choice(nans) if len(nans) > 0 else None


def _partial_df_to_promts(partial_df: pd.DataFrame):
    encoder = lambda x: _encode_row_partial(x, True)
    res_encode = list(partial_df.apply(encoder, axis=1))
    res_first = list(partial_df.apply(_get_random_missing, axis=1))

    res = [
        ((enc + ", ") if len(enc) > 0 else "")
        + (fst + " is" if fst is not None else "")
        for enc, fst in zip(res_encode, res_first)
    ]
    return res
