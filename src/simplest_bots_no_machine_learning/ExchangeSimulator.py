#simple exchange simulator for only two currencies trading
INF = 1e10
class ExchangeSimulator():
    def __init__(self, bot, commission):
        self.bot = bot
        self.first = 1.0
        self.second = 0.0
        self.commission = commission
        
    def from_first_to_second(self, amount, rate):
        actual_amount = min(self.first, amount)
        self.first -= actual_amount
        self.second += actual_amount / rate * self.commission
        
    def from_second_to_first(self, amount, rate):
        actual_amount = min(self.second, amount)
        self.second -= actual_amount
        self.first += actual_amount * rate * self.commission
    
    def modulate(self, low_rates, high_rates):
        for i in range(low_rates.shape[0]):
            action_now = self.bot.get_action(low_rates[i], high_rates[i], self.first, self.second)
            if action_now is not None:
                if action_now[0] == 'to_second':
                    self.from_first_to_second(action_now[1], high_rates[i])
                if action_now[0] == 'to_first':
                    self.from_second_to_first(action_now[1], low_rates[i])
        self.from_second_to_first(INF, low_rates[-1])
        return self.first
