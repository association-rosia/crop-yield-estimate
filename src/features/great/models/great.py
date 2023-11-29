import sys
import os
import warnings
import json
import typing as tp
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.curdir)

from src.features.great.models.great_utils import (
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    _partial_df_to_promts
)


class GReaT:
    def __init__(self, llm: str):

        # Load Model and Tokenizer from HuggingFace
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None

    def great_sample(
            self,
            starting_prompts: tp.Union[str, list[str]],
            temperature: float = 0.7,
            max_length: int = 100,
            device: str = 'cuda',
    ) -> pd.DataFrame:

        self.model.to(device)
        starting_prompts = (
            [starting_prompts]
            if isinstance(starting_prompts, str)
            else starting_prompts
        )
        generated_data = []

        # Generate a sample for each starting point
        if len(starting_prompts) > 1:
            loop_iter = tqdm(starting_prompts)
        else:
            loop_iter = starting_prompts

        for prompt in loop_iter:
            start_token = torch.tensor(self.tokenizer(prompt)['input_ids']).to(device)

            # Generate tokens
            gen = self.model.generate(
                input_ids=torch.unsqueeze(start_token, 0),
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=50256,
            )

            generated_data.append(torch.squeeze(gen))

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(decoded_data, self.columns)

        return df_gen

    def impute(
            self,
            df_miss: pd.DataFrame,
            temperature: float = 1.0,
            max_length: int = 100,
            max_retries: int = 100,
            device: str = 'cuda',
    ) -> pd.DataFrame:

        # Check DataFrame passed.
        if set(df_miss.columns) != set(self.columns):
            raise ValueError(
                'The column names in the DataFrame passed to impute do not match the columns of the GReaT model.'
            )

        self.model.to(device)
        start_temperature = temperature
        df_list = []

        for index in tqdm(range(len(df_miss))):
            temperature = start_temperature
            is_complete = False
            retries = 0
            df_curr = df_miss.iloc[[index]]
            org_index = df_curr.index  # Keep index in new DataFrame

            while not is_complete:
                # Generate text promt from current features.
                start_num_nan = df_curr.isna().any().sum()
                starting_prompts = _partial_df_to_promts(df_curr)
                df_curr = self.great_sample(starting_prompts, temperature, max_length, device=device)

                # Convert numerical values to float, flawed numerical values to NaN
                for i_num_cols in self.num_cols:
                    df_curr[i_num_cols] = pd.to_numeric(df_curr[i_num_cols], errors='coerce')

                df_curr[self.num_cols] = df_curr[self.num_cols].astype(float)
                current_num_nan = df_curr.isna().any().sum()

                # Check for missing values
                if not df_curr.isna().any().any():
                    is_complete = True
                    df_list.append(df_curr.set_index(org_index))

                elif retries == max_retries:
                    warnings.warn('Max retries reached.')
                    is_complete = True
                    df_list.append(df_curr.set_index(org_index))

                else:
                    if start_num_nan == current_num_nan:
                        temperature += 0.1

                    retries += 1

                index += 1

        return pd.concat(df_list, axis=0)

    @classmethod
    def load_from_dir(cls, path: str):
        assert os.path.isdir(path), f'Directory {path} does not exist.'

        # Load attributes
        with open(path + '/config.json', 'r') as f:
            attributes = json.load(f)

        # Create new be_great model instance
        great = cls(attributes['llm'])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + '/model.pt', map_location='cpu'))

        return great
