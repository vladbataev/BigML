class CrossectionBot():
    def __init__(self, mu_big, mu_small, threshold_sell, max_ticks, trigger_num):
        self.mu_big = mu_big
        self.mu_small = mu_small
        self.threshold_sell = threshold_sell
        self.trigger_num = trigger_num
        self.max_ticks = max_ticks
        self.accumulated_big = None
        self.accumulated_small = None
        self.num_trades = 0
        self.ticks = None
        self.accumulated_big_history = []
        self.accumulated_small_history = []
        self.state = 'free'
        
    def update(self, accumulated, mu, new):
        if accumulated is None:
            return new
        return accumulated * mu + (1.0 - mu) * new
    
    def triggered(self):
        if len(self.accumulated_big_history) < 3 * self.trigger_num:
            return False
        
        all_bigger = True
        for i in range(len(self.accumulated_big_history) - 3 * self.trigger_num, \
                      len(self.accumulated_big_history) - 2 * self.trigger_num):
            if self.accumulated_big_history[i] < self.accumulated_small_history[i]:
                all_bigger = False
                
        all_smaller = True
        for i in range(len(self.accumulated_big_history) - self.trigger_num, len(self.accumulated_big_history)):
            if self.accumulated_big_history[i] > self.accumulated_small_history[i]:
                all_smaller = False
        
       
        return all_bigger and all_smaller
    
    
    def get_action(self, low_now, high_now, first, second):
        self.accumulated_big = self.update(self.accumulated_big, self.mu_big, low_now)
        self.accumulated_small = self.update(self.accumulated_small, self.mu_small, low_now)
        self.accumulated_big_history.append(self.accumulated_big)
        self.accumulated_small_history.append(self.accumulated_small)
        
        if (self.state == 'free'):
            if (self.triggered()):
                self.num_trades += 1
                self.ticks = 0
                self.state = 'wait'
                return ['to_second', INF]
            
        
        if (self.state == 'wait'):
            self.ticks += 1
            if self.ticks >= self.max_ticks:
                self.state = 'free'
                return ['to_first', INF]

            '''  if self.comprasion == 'buying_rate':
                reference = self.buying_rate
            else:
                reference = self.accumulated'''
            reference = self.accumulated_big
            if (low_now / reference > self.threshold_sell):
                #print("low_now:", low_now)
                self.state = 'free'
                return ['to_first', INF]
            
