class MovingAverageBot():
    def __init__(self, mu, threshold_buy, threshold_sell, max_ticks, comprasion = 'buying_rate'):
        self.mu = mu
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.accumulated = None
        self.action_now = 'free'
        self.num_trades = 0
        self.buying_rate = None
        self.max_ticks = max_ticks
        self.ticks = None
        self.comprasion = comprasion
        
    def get_action(self, low_now, high_now, first, second):
        #print(first, second)
        if self.accumulated is None:
            self.accumulated = low_now
        else:
            self.accumulated = self.mu * self.accumulated + (1.0 - self.mu) * low_now
        
        if (self.action_now == 'free'):
            if high_now < self.threshold_buy * self.accumulated:
                self.num_trades += 1
                self.buying_rate = high_now
                self.action_now = 'wait'
                #print("high now:", high_now)
                self.ticks = 0
                return ['to_second', INF]
        if (self.action_now == 'wait'):
            self.ticks += 1
            if self.ticks >= self.max_ticks:
                self.action_now = 'free'
                return ['to_first', INF]
            
            if self.comprasion == 'buying_rate':
                reference = self.buying_rate
            else:
                reference = self.accumulated
            
            if (low_now / reference > self.threshold_sell):
                #print("low_now:", low_now)
                self.action_now = 'free'
                return ['to_first', INF]
            
