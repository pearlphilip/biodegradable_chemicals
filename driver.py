'''
Main driver for running the tool
'''

import pandas as pd

import format_data

def main():
    '''
    Entry point for all code
    '''
    print("starting up")
    df = format_data.read_data('data/biodeg.csv')


if __name__ == "__main__":
    main()
