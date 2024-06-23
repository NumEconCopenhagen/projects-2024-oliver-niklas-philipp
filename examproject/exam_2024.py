import numpy as np
from scipy.optimize import minimize_scalar, minimize
from types import SimpleNamespace

class EconomicsModel:
    def __init__(self):
        self.par = SimpleNamespace()
        # firms
        self.par.A = 1.0
        self.par.gamma = 0.5
        
        # households
        self.par.alpha = 0.3
        self.par.nu = 1.0
        self.par.epsilon = 2.0
        
        # government
        self.par.tau = 0.0
        self.par.T = 0.0
        
        # Question 3
        self.par.kappa = 0.1

    def firm_labor_demand(self, w, p, A, gamma):
        "Firms' labor demand - given wage w, price p, productivity A, and output elasticity gamma."
        "Labour demand is here the optimal labour demand, given the wage, output elasticity and the price. For each of the two firms"
        return ((p * A * gamma) / w) ** (1 / (1 - gamma))

    def firm_output(self, l, A, gamma):
        "Firms' output - given labor l, productivity A, and output elasticity gamma."
        "The output is the production of the firm, given the labour input. For each of the two firms."
        return A * l ** gamma

    def firm_profit(self, w, p, A, gamma):
        "Firms' profit - given wage w, price p, productivity A, and output elasticity gamma."
        l = self.firm_labor_demand(w, p, A, gamma)
        return p * self.firm_output(l, A, gamma) - w * l

    def consumer_utility(self, c1, c2, alpha, nu, ell, epsilon):
        "Consumer utility - given consumption of good 1 c1 and good 2 c2, preference parameter of good 1 alpha, disutility of labor nu, labor supplied by consumers ell, and elasticity of labor supply epsilon."
        return np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * (ell ** (1 + epsilon)) / (1 + epsilon)

    def market_clearing(self, p1, p2, w):
        "Question 1"
        "For each of the two firms and the consumer, we find the optimal behavior and check the market clearing conditions."
        "This is firstly done by finding the optimal labour and output for a given price and then calculating the profit."
        
        # Firm 1's behaviour given prices, wage, labor, technology, and output elasticity
        l1_star = self.firm_labor_demand(w, p1, self.par.A, self.par.gamma)
        y1_star = self.firm_output(l1_star, self.par.A, self.par.gamma)
        
        # Firm 2's behaviour given prices, wage, labor, technology, and output elasticity
        l2_star = self.firm_labor_demand(w, p2, self.par.A, self.par.gamma)
        y2_star = self.firm_output(l2_star, self.par.A, self.par.gamma)

        # Firms' profit given prices, wage, technology and output elasticity
        pi1_star = self.firm_profit(w, p1, self.par.A, self.par.gamma)
        pi2_star = self.firm_profit(w, p2, self.par.A, self.par.gamma)
        
        # Consumer's optimal behavior
        def utility_maximization(ell, w, p1, p2):
            "Consumers utility maximization problem. Given the wage, labor supplied by households, optimal firm profits, taxes, and preference parameter of good 1."
            "Here we return it negatively, as to use minimize to find the lowest value of consumer disutility"
            pi1_star = self.firm_profit(w, p1, self.par.A, self.par.gamma)
            pi2_star = self.firm_profit(w, p2, self.par.A, self.par.gamma)
            
            c1 = self.par.alpha * ((w * ell + self.par.T + pi1_star + pi2_star) / p1)
            c2 = (1 - self.par.alpha) * ((w * ell + self.par.T + pi1_star + pi2_star) / (p2 + self.par.tau))
            
            utility = self.consumer_utility(c1, c2, self.par.alpha, self.par.nu, ell, self.par.epsilon)
            return -utility  # Ensure this is a scalar value for minimization

        #minimize the utility maximization problem
        res = minimize(utility_maximization, x0=1.0, args=(w, p1, p2))

        #lowest value of disultility
        ell_star = res.x[0]
        
        # Consumer's optimal consumption given the optimal disutility of labor. Hence we find what labor will be supplied at the optimal consumption given the prices.
        c1_star = self.par.alpha * ((w * ell_star + self.par.T + pi1_star + pi2_star) / p1)
        c2_star = (1 - self.par.alpha) * ((w * ell_star + self.par.T + pi1_star + pi2_star) / (p2 + self.par.tau))
        
        # Market clearing conditions, which calculates whether the labor market and the goods markets clear.
        labor_market = ell_star - (l1_star + l2_star)
        good_market_1 = c1_star - y1_star
        good_market_2 = c2_star - y2_star
        
        return labor_market, good_market_1, good_market_2, y2_star, ell_star, c1_star, c2_star

    def find_best_market_clearing(self, market_clearing_results):
        "Question 2"
        "To find the market clearing prices, from question 1, we simply iterate through all possible results to find the one that is closest to market clearing."
        
        # We start by initializing variables storing the best values."
        best_p1 = None
        best_p2 = None
        #this is simply an infinitely large number, such that all improvements can be store here.
        min_distance = float('inf')
        best_labor_market = None
        best_good_market_1 = None
        best_good_market_2 = None
        
        # Loop going through all the market clreaing prices
        for result in market_clearing_results:
            p1 = result[0]
            p2 = result[1]
            labor_market = result[2][0]
            good_market_1 = result[2][1]
            good_market_2 = result[2][2]
            
            # Calculation the distance from 0, hence how far the market is from clearing.
            # Calculate sum of squares of deviations from zero
            distance = good_market_1 ** 2 + good_market_2 ** 2
            
            # For each result, the model will now check if the distance is smaller than the previous smallest distance.
            # If this is the case, the model will update the best values to the current values.
            # Which means that the function should converge to the market clearing prices.
            if distance < min_distance:
                min_distance = distance
                best_p1 = p1
                best_p2 = p2
                best_labor_market = labor_market
                best_good_market_1 = good_market_1
                best_good_market_2 = good_market_2
        
        return best_p1, best_p2, best_labor_market, best_good_market_1, best_good_market_2


    def find_optimal_tau(self):
        "Question 3"
        "Takes the resulting optimal tau from the social wefare function and calculates the corresponding consumer behaviour and market clearing values."
        "Given these values, the optimal lump sum transfer can be calculated"

        #the optimal tax rate tau is minimized for in the social welfare function
        result = minimize_scalar(lambda tau: self.calculate_swf(tau), bounds=(0.0, 1.0), method='bounded')
        optimal_tau = result.x

        #the equilibrium values are found for the consumer and market given the optimal tau
        optimal_equilibrium = self.find_equilibrium_for_tau(optimal_tau)

        #the lump sum transfer is calculated based on  the equilibrium values of the consumer
        optimal_T = optimal_tau * optimal_equilibrium['c2_star'] if optimal_equilibrium else None
        
        return optimal_tau, optimal_T, optimal_equilibrium

    def find_equilibrium_for_tau(self, tau):
        "Find the equilibirum values, which afterwards will be used to optimize tau to find the best SWF"        
        self.par.tau = tau
        p1_values = np.linspace(0.1, 2.0, 10)
        p2_values = np.linspace(0.1, 2.0, 10)
        w = 1.0
        
        #initializing the storgae variables
        best_p1 = None
        best_p2 = None
        #again initializing the value to be infinitely large
        min_distance = float('inf')
        best_results = None
        
        # Using a loop to find the values for the market clearing prices
        for p1 in p1_values:
            for p2 in p2_values:
                labor_market, good_market_1, good_market_2, y2_star, ell_star, c1_star, c2_star = self.market_clearing(p1, p2, w)
                distance = good_market_1 ** 2 + good_market_2 ** 2
                
                #finding the proces which are the closest to market clearing
                if distance < min_distance:
                    min_distance = distance
                    best_p1 = p1
                    best_p2 = p2
                    best_results = {
                        'p1': best_p1,
                        'p2': best_p2,
                        'labor_market': labor_market,
                        'good_market_1': good_market_1,
                        'good_market_2': good_market_2,
                        'y2_star': y2_star,
                        'ell_star': ell_star,
                        'c1_star': c1_star,
                        'c2_star': c2_star
                    }
        
        return best_results

    def calculate_swf(self, tau):
        "This function calculates the social wefare from chainging the tax rate and the lump sum transer to the consumer."
        
        # Using the previously defined welfare functions
        equilibrium_results = self.find_equilibrium_for_tau(tau)

        # Defining the value which the government balances its budget around and transfers back to the consumer
        self.par.T = tau * equilibrium_results['c2_star']  

        # Now it is essential to recalculate the consumer utility given the update in taxes and lump sum transfers.
        w = 1.0
        p1 = equilibrium_results['p1']
        p2 = equilibrium_results['p2']
        pi1_star = self.firm_profit(w, p1, self.par.A, self.par.gamma)
        pi2_star = self.firm_profit(w, p2, self.par.A, self.par.gamma)
        
        # Finding the new utilities attached to consuming the two goods
        def utility_maximization(ell, w, p1, p2, par, pi1_star, pi2_star):
                c1 = par.alpha * ((w * ell + par.T + pi1_star + pi2_star) / p1)
                c2 = (1 - par.alpha) * ((w * ell + par.T + pi1_star + pi2_star) / (p2 + par.tau))
                return -self.consumer_utility(c1, c2, par.alpha, par.nu, ell, par.epsilon)
        

        #minimize the utility maximization problem and finding the amount of labor supplied
        res = minimize(utility_maximization, x0=1.0, args=(w, p1, p2, self.par, pi1_star, pi2_star))
        ell_star = res.x[0]
        
        # Finding the optimal consumption of each good
        c1_star = self.par.alpha * (w * ell_star + self.par.T + pi1_star + pi2_star) / p1
        c2_star = (1 - self.par.alpha) * (w * ell_star + self.par.T + pi1_star + pi2_star) / (p2 + self.par.tau)
        
        # Calculating the actual social wefare function
        utility = self.consumer_utility(c1_star, c2_star, self.par.alpha, self.par.nu, ell_star, self.par.epsilon)
        swf = utility - self.par.kappa * equilibrium_results['y2_star']
        return -swf  
