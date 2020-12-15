#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import run_shell


def cmd_test_1():
    print('test 1')


def cmd_test_2():
    print('test 2')


globals_params = {}
if __name__ == '__main__':
    params_options = {'param1': ['p1', 'p2', 'p3', 'p4']}
    run_shell(globals_params=globals_params, params_options=params_options)
