#!/usr/bin/python

'''
Little python program to pickle.load appended files recursively
'''

import pickle
import sys

def main():
    if len(sys.argv) >= 2:
        file = sys.argv[1]
        output = {}
        n = 0
        with open(file, 'r') as f:
            while 1:
                try:
                    n += 1
                    output.update({n : pickle.load(f)})
                except EOFError:
                    break
        for i in output:
            print output[i], '\n'
    else:
        print 'Wrong file name!'

if __name__ == '__main__':
  main()