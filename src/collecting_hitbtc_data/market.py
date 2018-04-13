import sys

import numpy as np

import time as tm
import datetime as dtm
import math

from utils import *


class Currency(object):
    """
    Class Currency contains info about a cryptocurrency, including id, name and lists of pairs, which are connected with the currency.\n
    Method __init__ accepts currency id and it's name. 
    """
    def __init__(self, id_str, name):
        self.id_ = id_str
        self.name_ = name
        self.base_to_ = []
        self.quote_to_ = []


    def __repr__(self):
        return self.id_ + ' (' + self.name_ + ')\n'


    def init_pairs_info(self, pairs):
        """
        Initialization of pairs, which are connected with the currency.
        Parameter pairs is a dict of Pair objects.
        """
        self.base_to_ = [pairs[pair_id] for pair_id in pairs.keys() if pairs[pair_id].base_.id_ == self.id_]
        self.quote_to_ = [pairs[pair_id] for pair_id in pairs.keys() if pairs[pair_id].quote_.id_ == self.id_]
        return


class OrderBook(object):
    """
    Class OrderBook contains actual info about the orderbook, prices and volumes are assumed to be scaled.
    While we're not using orderbook info, it's just a wrapper above normalized ask and bid values.\n
    Method __init__ accepts no parameters.
    """
    def __init__(self):
        self.ask_, self.bid_, self.spread_ = 0, 0, 0
        # self.asks_, self.bids_ = np.array([], dtype=np.int), np.array([], dtype=np.int)
        # self.asks_volume_, self.bids_volume_ = np.array([], dtype=np.int), np.array([], dtype=np.int)
        # self.order_book_info_ = False


    # def __repr__(self):
    #     return None


    def update_from_ticker(self, ticker, step_price):
        """
        Setting ask and bid values from a .json ticker;
        step_price parameter is the step_price_ variable from class Pair.\n
        Ticker .json HitBTC format:
        {
            "ask": "0.050043",
            "bid": "0.050042",
            "last": "0.050042",
            "open": "0.047800",
            "low": "0.047052",
            "high": "0.051679",
            "volume": "36456.720",
            "volumeQuote": "1782.625000",
            "timestamp": "2017-05-12T14:57:19.999Z",
            "symbol": "ETHBTC"
        }
        """
        self.ask_, self.bid_ = round(float(ticker['ask']) / step_price),\
                               round(float(ticker['bid']) / step_price)
        self.spread_ = self.ask_ - self.bid_
        # self.asks_, self.bids_ = np.array([], dtype=np.int), np.array([], dtype=np.int)
        # self.asks_volume_, self.bids_volume_ = np.array([], dtype=np.int), np.array([], dtype=np.int)
        # self.order_book_info_ = False
        return


    # def update_from_orderbook(self, orderbook, depth=50):
    #    self.order_bok_info_ = True
    #    return


class Pair(object):
    """
    Class Pair contains info about the exchange market for two currencies (including orderbook).\n
    Method __init__ accepts two parameters: symbol (pair info) .json file and currency .json file in the HitBTC format both:\n
    Symbol:
    {
        "id": "ETHBTC",
        "baseCurrency": "ETH",
        "quoteCurrency": "BTC",
        "quantityIncrement": "0.001",
        "tickSize": "0.000001",
        "takeLiquidityRate": "0.001",
        "provideLiquidityRate": "-0.0001",
        "feeCurrency": "BTC"
    }\n
    Currency:
    {
        "id": "BTC",
        "fullName": "Bitcoin",
        "crypto": true,
        "payinEnabled": true,
        "payinPaymentId": false,
        "payinConfirmations": 2,
        "payoutEnabled": true,
        "payoutIsPaymentId": false,
        "transferEnabled": true,
        "delisted": false,
        "payoutFee": "0.00958"
   }
    """
    def __init__(self, symbol, currencies):
        self.timestamp_ = 0
        self.id_ = symbol['id']
        self.base_, self.quote_ = currencies[symbol['baseCurrency']], currencies[symbol['quoteCurrency']]

        self.fee_ = currencies[symbol['feeCurrency']]
        self.fees_ = float(symbol['takeLiquidityRate']), float(symbol['provideLiquidityRate'])

        self.step_price_ = float(symbol['tickSize'])
        self.step_volume_ = float(symbol['quantityIncrement'])

        self.orderbook_ = OrderBook()


    def __repr__(self):
        return self.id_ + ' (' + self.base_.id_ + '/' + self.quote_.id_ +\
               ')' + (('\nask: ' + str(self.orderbook_.ask_*self.step_price_) +\
                       '\nbid: ' + str(self.orderbook_.bid_*self.step_price_) + '\n') if self.timestamp_ else '')


    def update_from_ticker(self, ticker):
        """
        Updating the orderbook info from a .json ticker file in the HitBTC format:
        {
            "ask": "0.050043",
            "bid": "0.050042",
            "last": "0.050042",
            "open": "0.047800",
            "low": "0.047052",
            "high": "0.051679",
            "volume": "36456.720",
            "volumeQuote": "1782.625000",
            "timestamp": "2017-05-12T14:57:19.999Z",
            "symbol": "ETHBTC"
        }
        """
        self.timestamp_ = timestamp_to_milliseconds(ticker['timestamp'])
        self.orderbook_.update_from_ticker(ticker, self.step_price_)


    # def update_from_orderbook(self, orderbook, depth=50):
    #     return
