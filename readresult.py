#!/usr/bin/python

'''
Little python module to pickle.load results
'''

import pickle

def readresult(file):
    with open(file, 'r') as f_result:
        result = pickle.load(f_result)
    return result