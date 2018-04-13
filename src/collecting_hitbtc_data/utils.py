import sys

import requests
from bs4 import BeautifulSoup
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline

import time as tm
import datetime as dtm
import math
# from decimal import *

# import asyncio  # later


class Colors(object):
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'
    RESET   = '\033[39m'


def ping(url, attempts=10, logs=False, logs_final=True):
    session = requests.Session()
    session.get(url).json()
    pings = []

    for index in range(attempts):
        start = tm.time()
        req = session.get(url).json()
        finish = tm.time()
        pings.append(1000*(finish - start))
        if logs:
            print('ping ', index+1, ': ', round(pings[-1], 3), ' ms', sep='')

    pings = np.array(pings)
    ping_mean, ping_std, ping_max = round(pings.mean(), 3), round(pings.std(), 3), round(pings.max(), 3)
    if logs_final:
        if logs:
            print()
        print('address: ' + url)
        print('avr/std/max: ', ping_mean, '/', ping_std, '/', ping_max, ' ms', sep='')
    return ping_mean, ping_std, ping_max


def timestamp_to_microseconds(string):
    return int(dtm.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000000)


def timestamp_to_milliseconds(string):
    return int(dtm.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000000) // 1000


def now_to_microseconds():
    return int(dtm.datetime.now().timestamp() * 1000000)


def now_to_milliseconds():
    return int(dtm.datetime.now().timestamp() * 1000000) // 1000
