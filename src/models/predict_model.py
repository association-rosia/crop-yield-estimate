import os
import sys
import argparse
from argparse import Namespace

sys.path.append(os.curdir)

import pandas as pd

from src.features.preprocessing import CYEDataPreProcessor

from xgboost.sklearn import XGBRegressor

from src.constants import get_constants

cst = get_constants()

def main():
    args = parse_args()
    preprocessor = load_preprocessor(args.name)
    model = load_model(args.name)
   
   # Pre-process data
    raw_data = pd.read_csv(cst.file_data_test)
    input_data = preprocessor.preprocess(raw_data)
    input_data = preprocessor.transform(input_data)
    
    preds = model.predict(input_data)

    # Create submisiion file to be uploaded to Zindi for scoring
    submission = pd.DataFrame({'ID': input_data.index, 'Yield': preds})
    file_submission = os.path.join(cst.path_submissions, f'{args.name}.csv')
    submission.to_csv(file_submission, index=False)
    
def load_model(name_run: str) -> XGBRegressor:
    file_model = os.path.join(cst.path_models, name_run, 'model.json')
    xgbr = XGBRegressor()
    xgbr.load_model(fname=file_model)
    
    return xgbr 

 
def load_preprocessor(name_run: str) -> CYEDataPreProcessor:
    file_preprocessor = os.path.join(cst.path_models, name_run, 'preprocessor.json')
    
    return CYEDataPreProcessor.load(file_preprocessor)


def parse_args() -> Namespace:
    # Define the parameters
    parser = argparse.ArgumentParser(description=f'Make {cst.project_name} submission')
    
    # Run name
    parser.add_argument('-n', '--name', type=str, help='Name of run to use for submission')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()