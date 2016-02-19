#!/usr/bin/env python3
'''
Main driver for running the tool
'''

import pandas as pd

import format_data
import transform

def main():
    '''
    Entry point for all code
    '''
    print("starting up")
    df = format_data.read_data('data/biodeg.csv')
    df = transform.transform(df)

if __name__ == "__main__":
    main()
