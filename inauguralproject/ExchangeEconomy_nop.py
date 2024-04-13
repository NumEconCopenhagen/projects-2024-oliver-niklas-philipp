from types import SimpleNamespace

class ExchangeEconomyClass:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self,x1B,x2B):
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def demand_A(self,p1):
        p2 = 1
        demand_x1A = self.par.alpha * ((p1 * self.par.w1A + p2 * self.par.w2A) / p1)
        demand_x2A = (1 - self.par.alpha) * ((p1 * self.par.w1A + p2 * self.par.w2A) / p1)
        return demand_x1A, demand_x2A

    def demand_B(self,p1):
        p2 = 1
        demand_x1B = self.par.beta * ((p1 * self.par.w1B + p2 * self.par.w2B) / p1)
        demand_x2B = (1 - self.par.beta) * ((p1 * self.par.w1B + p2 * self.par.w2B) / p1)
        return demand_x1B, demand_x2B