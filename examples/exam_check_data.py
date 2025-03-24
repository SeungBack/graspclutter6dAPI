import os
from graspclutter6dAPI import GraspClutter6D

# GraspClutter6D example for checking the data completeness.

if __name__ == '__main__':

    if 'GC6D_ROOT' not in os.environ:
        print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
        exit(0)
    else:
        gc6d_root = os.environ['GC6D_ROOT']


    g = GraspClutter6D(gc6d_root, split='all')
    if g.checkDataCompleteness():
        print('Check data completeness: PASSED')
    else:
        print('Check data completeness: FAILED')