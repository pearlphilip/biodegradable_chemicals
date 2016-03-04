#!/usr/bin/env python2.7
'''
Main driver for running the tool
'''

import pandas as pd

import format_data
import nn_model
import transform

def main():
    '''
    Entry point for all code
    '''
    print("starting up")
    df = format_data.read_data('data/biodeg.csv')
    df = transform.transform(df, 'class')
    nn_model.build_nn(df, 'class')

if __name__ == "__main__":
    main()
