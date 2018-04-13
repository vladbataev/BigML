#just for checking
class RandomBot():
    def __init__(self):
        pass
    def get_action(self, low_now, high_now, first, second):
        if (np.random.rand() < 0.5):
            return ['to_second', INF]
        else:
            return ['to_first', INF]
    
