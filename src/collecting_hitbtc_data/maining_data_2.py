import sys
import gc
import pickle
import requests

#from bs4 import BeautifulSoup
import json

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# %matplotlib inline

import time as tm
import datetime as dtm
import math
# from decimal import *

from utils import *
from market import Currency, Pair, OrderBook


def write_log(file_logs, log):
    return
    try:
        with open(file_logs, 'a') as f:
            timestamp = now_to_milliseconds()
            f.write(str(timestamp) + '(' + str(dtm.datetime.fromtimestamp(timestamp//1000)) + ') : ' + str(log) + '\n')
    except Exception as e:
        # sys.stdout.write(Colors.RED)
        # print(e)
        # sys.stdout.write(Colors.RESET)
        pass
    return

def updating(columns, pairs, session, period, quantile, file_logs, correct=False):
    def update_data(pairs, session, timeout=10000):
        def get_row(pairs, time_now):
            columns = ['timestamp']
            row = [time_now]
            for pair_id in pairs.keys():
                pair = pairs[pair_id]
                columns += [pair_id + '_ask', pair_id + '_bid', pair_id + '_timestamp']
                row += [pair.orderbook_.ask_, pair.orderbook_.bid_, pair.timestamp_]
            return pd.DataFrame([row], columns=columns)

        ticker_json = session.get(url+'ticker', timeout=timeout / 1000).json()
        tickers = {ticker['symbol']: ticker for ticker in ticker_json if ticker['symbol'] in pairs.keys()}

        if tickers.keys() == pairs.keys():
            for pair_id in pairs.keys():
                pairs[pair_id].update_from_ticker(tickers[pair_id])
        else:
            raise Exception('tickers.keys() != pairs.keys()\nexit')

        time_now = now_to_milliseconds()
        row = get_row(pairs, time_now)
        # data = data.append(row, ignore_index=True)
        return row, time_now

    try:
        quantile = min(quantile, period // 5)
        deadline = (now_to_milliseconds() // period + 1.0) * period if correct else now_to_milliseconds()

        while True:
            while now_to_milliseconds() < round(deadline):
                tm.sleep(quantile*0.001)
            if now_to_milliseconds() > (deadline + period):
               deadline += period
               continue

            deadline += period

            try:
                start = now_to_milliseconds()
                row, timestamp = update_data(pairs, session, timeout=period)
                ping = timestamp - start
            except Exception as e:
                write_log(file_logs, e)
                sys.stdout.write(Colors.RED)
                print('                Connection is lost... (' + type(e).__name__ + ')', sep='', end='\n')
                sys.stdout.write(Colors.RESET)
                continue

            yield row, timestamp, ping

    except Exception as e:
        write_log(file_logs, e)
        if e in (KeyboardInterrupt, SystemExit):
            raise e
        return None, None, None


if __name__ == '__main__':
    try:
        config = json.load(open('config.json'))
        url, url_mount = config['url'], config['url_mount']
        currencies_ids = config['currencies_ids']
        file_data, file_backup, file_logs, file_pickle = config['file_data'], config['file_backup'], config['file_logs'], config['file_pickle']
        rows_to_write, period, ping_limit, quantile = config['rows_to_write'], config['period'], config['ping_limit'], config['quantile']

        with open(file_logs, 'w') as f:
            pass
        file_backup = None if not isinstance(file_backup, str) else file_backup
        # url = 'https://api.hitbtc.com/api/2/public/'
        # url_mount = 'https://api.hitbtc.com/'
        session = requests.Session()
        session.get(url_mount)

        currencies_json = session.get(url+'currency').json()
        # currencies_ids = ['BTC', 'ETH', 'BCH', 'USD', 'DASH', 'XRP', 'XMR', 'LTC', ]
        currencies_json = [currency for currency in currencies_json if currency['id'] in currencies_ids]
        currencies = {currency['id']: Currency(currency['id'], currency['fullName']) for currency in currencies_json}
        # currencies

        pairs_json = session.get(url+'symbol').json()
        pairs_json = [pair for pair in pairs_json if (pair['baseCurrency'] in currencies.keys()) and\
                                                     (pair['quoteCurrency'] in currencies.keys())]
        pairs = {pair['id'] : Pair(pair, currencies) for pair in pairs_json}

        for currency_id in currencies_ids:
            currencies[currency_id].init_pairs_info(pairs)

        pickle.dump((currencies, pairs), open(file_pickle, 'wb'), protocol=3)
        # pairs

        columns = ['timestamp']
        for pair_id in pairs.keys():
            columns += [pair_id + '_ask', pair_id + '_bid', pair_id + '_timestamp']
        #data = pd.DataFrame([np.zeros(len(columns), dtype=int)], columns=columns)
        data = pd.DataFrame(columns=columns)
        # data

        # file_data, file_backup = 'data.csv', 'backup.csv'
        # rows_to_write, period, ping_limit, quantile = 10, 1000, 800, 1
        initial_timestamp = now_to_milliseconds()
        write_log(file_logs, 'Initial log')
    except Exception as e:
        sys.stdout.write(Colors.RED)
        print('Initialization failed!')
        print(e)
        sys.stdout.write(Colors.RESET)

    try:
        index, number = 0, 0
        for row, timestamp, ping in updating(columns, pairs, session, period, quantile, file_logs):
            if isinstance(row, type(None)):
                break
            index += 1
            data = data.append(row, ignore_index=True)

            print(timestamp, ' : Successful request [', index, '], ping: ', sep='', end='')
            sys.stdout.write(Colors.GREEN if ping < ping_limit / 4 else Colors.CYAN if ping < ping_limit / 2 else\
                             Colors.YELLOW if ping < ping_limit else Colors.RED)
            if ping >= ping_limit:
                write_log(file_logs, 'High ping (' + format(ping / 1000, '.3f') + ' s)')
            print(format(ping / 1000, '.3f'), sep='', end='')
            sys.stdout.write(Colors.RESET)
            print(' s, lifetime: ' +\
                  str(dtm.timedelta(seconds=(timestamp - initial_timestamp) // 1000)), sep='', end='')

            if not (index % rows_to_write):
                backup_flag, memory_flag = False, False
                try:
                    print('\n                Writing ' + file_data + '...', sep='', end='')
                    if not number:
                        data.to_csv(file_data)
                    else:
                        data.to_csv(file_data, mode='a', header=False)
                    sys.stdout.write(Colors.GREEN)
                    print(' Done!', sep='', end='')
                    sys.stdout.write(Colors.RESET)
                    memory_flag = True if not isinstance(file_backup, str) else False
                except Exception as e:
                    write_log(file_logs, e)
                    backup_flag = True
                    sys.stdout.write(Colors.RED)
                    if isinstance(file_backup, str):
                        print(' Cannot write ' + file_data + '! Moving to backup!', sep='', end='')
                        sys.stdout.write(Colors.RESET)
                        try:
                            print('\n                Writing ' + file_backup + '...', sep='', end='')
                            if not number:
                                data.to_csv(file_backup)
                            else:
                                data.to_csv(file_backup, mode='a', header=False)
                            sys.stdout.write(Colors.GREEN)
                            print(' Done!', sep='', end='')
                            sys.stdout.write(Colors.RESET)
                        except Exception as e:
                            write_log(file_logs, e)
                            sys.stdout.write(Colors.RED)
                            print(' Cannot write ' + file_backup + ' too! Check access rights!', sep='', end='')
                            sys.stdout.write(Colors.RESET)
                    else:
                        print(' Cannot write ' + file_data + '!', sep='', end='')

                if not backup_flag and isinstance(file_backup, str):
                    try:
                        print('\n                Writing ' + file_backup + '...', sep='', end='')
                        if not number:
                            data.to_csv(file_backup)
                        else:
                            data.to_csv(file_backup, mode='a', header=False)
                        sys.stdout.write(Colors.GREEN)
                        print(' Done!', sep='', end='')
                        sys.stdout.write(Colors.RESET)
                    except Exception as e:
                        write_log(file_logs, e)
                        sys.stdout.write(Colors.RED)
                        print(' Cannot write ' + file_backup + '! Is it using right now?', sep='', end='')
                        sys.stdout.write(Colors.RESET)

                if memory_flag:
                    data = pd.DataFrame(columns=columns)
                    gc.collect()
                    number += 1

            print()

        print('Some crash just happened! Check ' + file_data + ' and ' + file_backup + '! Exit.')
    except Exception as e:
        write_log(file_logs, e)
