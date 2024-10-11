import pandas as pd
import matplotlib as plt
from clean import Cleaner

PATH_TO_CSV = './16k_Movies.csv'

def openFile():
    file = PATH_TO_CSV
    data = pd.read_csv(file)
    return data

def get_info(df):
    df.info()

def get_count_of_duplicates(df):
    count_duplicates = len(df) - len(df.drop_duplicates())
    print(f"Очищено записей: {count_duplicates}")

def main():
    df = openFile()
    cleaner = Cleaner()
    df = cleaner.clean_duration(df)
    print(df)

if __name__ == '__main__':
    main()