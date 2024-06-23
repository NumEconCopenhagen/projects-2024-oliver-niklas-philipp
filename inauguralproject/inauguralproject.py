import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pandas as pd

class InauguralProjectClass:
    def __init__(self, alpha=1/3, beta=2/3, w1_A=0.8, w2_A=0.3, num_points=75, start=0.5):
        "Start by initializing all the parameters needed for the class"
        self.par = SimpleNamespace()
        self.par.alpha = alpha # Define alpha for Cobb-Douglas utility function (consumer A)
        self.par.beta = beta  # Define beta for Cobb-Douglas utility function (consumer B)
        self.par.w1_A = w1_A # Endowment of good 1 for consumer A
        self.par.w2_A = w2_A # Endowment of good 2 for consumer A
        self.par.w1_B = 1 - w1_A # Endowment of good 1 for consumer B
        self.par.w2_B = 1 - w2_A # Endowment of good 2 for consumer B
        self.par.num_points = num_points # Define number of points
        self.par.start = start # Start value for the price vector which we set to 0.5
        self.par.p1 = np.array([start + 2*i/num_points for i in range(0, num_points+1)]) # Price vector for good 1
        self.par.p2 = 1 # Price of good 2, set as the numeraire

        #Specifically for question 5B
        self.initial_utility_B = self.cobb_douglas_utility(self.par.w1_B, self.par.w2_B, self.par.beta)

        #Specifically for question 6A
        self.num_points = num_points

    def cobb_douglas_utility(self, x1, x2, alpha):
            """
            Parameters:
            x1: Quantity of good 1.
            x2: Quantity of good 2.
            alpha: Exponent parameter.
        
            Returns:
            Utility value.
            """    
            return (x1 ** alpha) * (x2 ** (1 - alpha))
    
    def x1_A(self, p1, p2, w1_A, w2_A):
        "Calculate the demand for good 1 by consumer A."
        x1_A = self.par.alpha * ((p1 * w1_A + p2 * w2_A) / p1)  # Demand for good 1 by consumer A
        return x1_A 
    
    def x2_A(self, p1, p2, w1_A, w2_A):
        "Calculate the demand for good 2 by consumer A."
        x2_A = (1 - self.par.alpha) * ((p1 * w1_A + p2 * w2_A) / p2)  # Demand for good 2 by consumer A
        return x2_A
    
    def x1_B(self, p1, p2, w1_B, w2_B):
        "Calculate the demand for good 1 by consumer B."
        x1_B = self.par.beta * ((p1 * w1_B + p2 * w2_B) / p1)  # Demand for good 1 by consumer B
        return x1_B
    
    def x2_B(self, p1, p2, w1_B, w2_B):
        "Calculate the demand for good 2 by consumer B."
        x2_B = (1 - self.par.beta) * ((p1 * w1_B + p2 * w2_B) / p2)  # Demand for good 2 by consumer B
        return x2_B
    
    def plot_points(self):
        "Question 1"
        "Edgeworth box illustrating pareto improvements for both consumers based on their initial endowments"
        par = self.par
        
        # Demand points for consumer A
        x1_values_A = np.linspace(0, 1, par.num_points)
        x2_values_A = np.linspace(0, 1, par.num_points)
        
        # Calculating at each point the utility for consumer A via the cobbdouglas utility function
        utilities_A = np.zeros((par.num_points, par.num_points))
        for i in range(par.num_points):
            for j in range(par.num_points):
                utilities_A[i, j] = self.cobb_douglas_utility(x1_values_A[i], x2_values_A[j], par.alpha)
        
        # Demand points for consumer A
        x1_values_B = 1 - x1_values_A
        x2_values_B = 1 - x2_values_A

        # Calculating at each point the utility for consumer B via the cobbdouglas utility function
        utilities_B = np.zeros((par.num_points, par.num_points))
        for i in range(par.num_points):
            for j in range(par.num_points):
                utilities_B[i, j] = self.cobb_douglas_utility(x1_values_B[i], x2_values_B[j], par.beta)
        
        # Plot settings 
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plots for the endowment poionts
        ax.scatter(par.w1_A, par.w2_A, color='blue', label='Endowment A', zorder=5, alpha=0.5)
        ax.scatter(par.w1_B, par.w2_B, color='red', label='Endowment B', zorder=5, alpha=0.5)

        # Annotation for the endowment points
        ax.annotate('(x1A, x2A)', (par.w1_A, par.w2_A), textcoords="offset points", xytext=(5,5), ha='center')
        ax.annotate('(x1B, x2B)', (par.w1_B, par.w2_B), textcoords="offset points", xytext=(5,5), ha='center')

        # X and Y axis labels as well as title
        ax.set_xlabel('$x_{1A}$')
        ax.set_ylabel('$x_{2A}$')
        ax.set_title('Edgeworth Box Plot (Cobb-Douglas Utility)')
        ax.legend()

        # Add secondary x and y axis for consumer B
        ax2 = ax.secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))
        ax2.set_xlabel('$x_{1B}$')

        ax3 = ax.secondary_yaxis('right', functions=(lambda y: 1 - y, lambda y: 1 - y))
        ax3.set_ylabel('$x_{2B}$')

        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')

        # Calculating the utility for given endowments for consumer A
        utility_endowment_A = self.cobb_douglas_utility(par.w1_A, par.w2_A, par.alpha)

        # Plot the points where consumer A's utility is greater than or equal to the endowment
        satisfactory_points_A = np.argwhere(utilities_A >= utility_endowment_A)
        satisfactory_x1_values_A = x1_values_A[satisfactory_points_A[:, 0]]
        satisfactory_x2_values_A = x2_values_A[satisfactory_points_A[:, 1]]
        ax.scatter(satisfactory_x1_values_A, satisfactory_x2_values_A, color='green', label='Satisfactory Points A', alpha=0.5)

        # Calculate the utility of the endowment for consumer B in transformed coordinates
        utility_endowment_B = self.cobb_douglas_utility(par.w1_B, par.w2_B, par.beta)

        # Plot the points where consumer B's utility is greater than or equal to the endowment
        satisfactory_points_B_x = 1 - satisfactory_x1_values_A
        satisfactory_points_B_y = 1 - satisfactory_x2_values_A
        ax.scatter(satisfactory_points_B_x, satisfactory_points_B_y, color='purple', label='Satisfactory Points B', alpha=0.5)

        plt.legend()
        plt.show()

    def calculate_demand_and_clearing_errors(self):
        "Question 2"
        "Calculating the demand for each consumer at each given p1 price, hereby also checking whether it cleras the market"
        par = self.par
        
        for i, price in enumerate(par.p1):
            # Looping through each p1 price and calculating the demand for each consumer
            demand_A_1 = np.sum(self.x1_A(price, par.p2, par.w1_A, par.w2_A))  
            demand_A_2 = np.sum(self.x2_A(price, par.p2, par.w1_A, par.w2_A))  
            demand_B_1 = np.sum(self.x1_B(price, par.p2, par.w1_B, par.w2_B))  
            demand_B_2 = np.sum(self.x2_B(price, par.p2, par.w1_B, par.w2_B))  
            
            # Calculating market clearing errors
            error_1 = demand_A_1 + demand_B_1 - par.w1_A - par.w1_B  
            error_2 = demand_A_2 + demand_B_2 - par.w2_A - par.w2_B  

            # Printing the result at each given price point
            print(f"At price {price:.2f}:")
            print(f"Demand A for good 1: {demand_A_1:.2f}")
            print(f"Demand A for good 2: {demand_A_2:.2f}")
            print(f"Demand B for good 1: {demand_B_1:.2f}")
            print(f"Demand B for good 2: {demand_B_2:.2f}")
            print(f"Market Clearing Error for good 1: {error_1:.2f}")
            print(f"Market Clearing Error for good 2: {error_2:.2f}")
            print()



    def find_market_clearing_price(self):
        "Question 3"
        "Given the above market clearing errors, the price is found at which the market clears"

        price_range = self.par.p1
        #minimum error is set to infinity, so as to find the minimum error
        min_error = float('inf')
        #storing the error prices  
        min_error_price = None  
        #storing the corresponding demands
        corresponding_demands = None  

        for p1 in price_range:
            "Looping through the price range to find the minimum error"
            p2 = 1.0
            demand_x1A = self.x1_A(p1, p2, self.par.w1_A, self.par.w2_A)
            demand_x2A = self.x2_A(p1, p2, self.par.w1_A, self.par.w2_A)
            demand_x1B = self.x1_B(p1, p2, self.par.w1_B, self.par.w2_B)
            demand_x2B = self.x2_B(p1, p2, self.par.w1_B, self.par.w2_B)

            total_demand_x1 = demand_x1A + demand_x1B
            total_demand_x2 = demand_x2A + demand_x2B

            error_x1 = abs(total_demand_x1 - (self.par.w1_A + self.par.w1_B))
            error_x2 = abs(total_demand_x2 - (self.par.w2_A + self.par.w2_B))

            total_error = error_x1 + error_x2

            if total_error < min_error:
                "Finding the price that minimizes the market clearing errors"
                min_error = total_error
                min_error_price = p1
                corresponding_demands = (demand_x1A, demand_x2A, demand_x1B, demand_x2B)

        return min_error_price, min_error, corresponding_demands

    
    def optimize_utility(self):
        "Question 4a"
        "Finding the allocation that maximizes consumer A's utility"

        #again setting the optimal utility to negative infinity
        optimal_utility = -np.inf

        #storing the optimal prices and demands
        optimal_price_p1 = None 
        optimal_x1_A = None  
        optimal_x2_A = None  
        optimal_x1_B = None  
        optimal_x2_B = None  

        for p1 in self.par.p1:
            #for each price the objective function is now minimized 
            result = minimize_scalar(self.objective, bounds=(p1, p1 + 2), method='bounded') 
            if result.success:
                current_optimal_price_p1 = result.x
                #calculating the optimal demand for each good for consumer A
                x1_A = 1 - self.x1_B(current_optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B)  
                x2_A = 1 - self.x2_B(current_optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B)  

                #utility achived by this demand
                current_optimal_utility = self.cobb_douglas_utility(x1_A, x2_A, self.par.alpha)  

                if current_optimal_utility > optimal_utility:
                    "given the caluculation of the utility, the optimal values are stored"
                    optimal_utility = current_optimal_utility
                    optimal_price_p1 = current_optimal_price_p1
                    optimal_x1_A = x1_A
                    optimal_x2_A = x2_A
                    optimal_x1_B = self.x1_B(optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B)
                    optimal_x2_B = self.x2_B(optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B)

        # Next we print these values at 3 decimals
        #starting wuth the optimal price, and the given price of good 2
        print("Utility-maximizing price p1:", round(optimal_price_p1, 3))
        print("Price of good 2 (p2):", self.par.p2)

        # Printing optimal demand values and utility
        print("Optimal values of x1A and x2A using the utility-maximizing price p1:")
        print("x1A:", round(optimal_x1_A, 3))
        print("x2A:", round(optimal_x2_A, 3))
        print("Optimal utility:", round(optimal_utility, 3))

        # Printing the demand values for consumer B
        print("\nOptimal demand for consumer B:")
        print("x1B:", round(optimal_x1_B, 3))
        print("x2B:", round(optimal_x2_B, 3))

    def objective(self, p1):
        "Question 4a"
        "this is the objective function of consumer A, to maximize utility given her demand for goods"
        x1_A = 1 - self.x1_B(p1, self.par.p2, self.par.w1_B, self.par.w2_B)  
        x2_A = 1 - self.x2_B(p1, self.par.p2, self.par.w1_B, self.par.w2_B)  

        #the utilitiy is here set as negative to minimize for the negative utility
        return -self.cobb_douglas_utility(x1_A, x2_A, self.par.alpha)  

    
    

    def optimize_utility_external(self):
        "Question 4b"
        "Next we check if the above excersize is possible given any price p1"
        
        #same as in 4a
        optimal_utility = -np.inf
        optimal_price_p1 = None  
        optimal_x1_A = None  
        optimal_x2_A = None  
        optimal_x1_B = None  
        optimal_x2_B = None  

        for p1 in self.par.p1:
            "same procedure as in 4a except we change the bounds to be p1 and p1 + 10"
            result = minimize_scalar(self.objective, bounds=(p1, p1 + 10), method='bounded') 
            if result.success:
                current_optimal_price_p1 = result.x
                x1_A = 1 - self.x1_B(current_optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B) 
                x2_A = 1 - self.x2_B(current_optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B) 
                current_optimal_utility = self.cobb_douglas_utility(x1_A, x2_A, self.par.alpha)

                if current_optimal_utility > optimal_utility:
                    optimal_utility = current_optimal_utility
                    optimal_price_p1 = current_optimal_price_p1
                    optimal_x1_A = x1_A
                    optimal_x2_A = x2_A
                    optimal_x1_B = self.x1_B(optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B)
                    optimal_x2_B = self.x2_B(optimal_price_p1, self.par.p2, self.par.w1_B, self.par.w2_B)

        # Return the optimal price p1, optimal demands for both consumers, and the corresponding utility
        return optimal_price_p1, optimal_x1_A, optimal_x2_A, optimal_x1_B, optimal_x2_B, optimal_utility

    def calculate_utilities(self):
        "Question 5A"
        "A is now the market maker, and we calculate the utilities for both consumers"
        "To find which allocation consumer A chooses we start by caluculating for all possible demands"
        par = self.par
        
        # Demand for consumer A
        x1_values_A = np.linspace(0, 1, par.num_points)
        x2_values_A = np.linspace(0, 1, par.num_points)
        
        # Calculate utilities for each point in the grid for consumer A
        utilities_A = np.zeros((par.num_points, par.num_points))
        for i in range(par.num_points):
            for j in range(par.num_points):
                utilities_A[i, j] = self.cobb_douglas_utility(x1_values_A[i], x2_values_A[j], par.alpha)
        
        # now the same is done for consumer B
        x1_values_B = 1 - x1_values_A
        x2_values_B = 1 - x2_values_A
        
        utilities_B = np.zeros((par.num_points, par.num_points))
        for i in range(par.num_points):
            for j in range(par.num_points):
                utilities_B[i, j] = self.cobb_douglas_utility(x1_values_B[i], x2_values_B[j], par.beta)
        
        # Hence we get the utilities for both consumers at each given demand possibility
        return utilities_A, utilities_B

    def optimize_utility1(self):
        "question 5A"
        "Given all possible utilities for demand, We need to find the allocation given that the choice set is restricted to $C$"
        "This is done by iterating through all possible utilities for consumer A"
        par = self.par
        
        optimal_x1_A = None
        optimal_x2_A = None
        max_utility_A = -np.inf
        
        optimal_x1_B = None
        optimal_x2_B = None
        max_utility_B = -np.inf
        
        utilities_A, utilities_B = self.calculate_utilities()
        
        #A. Iterate over the grid of possible allocations for consumer A
        for i in range(par.num_points):

            #b. Inner loop over all the possible allocations
            for j in range(par.num_points):

                #c. Calculate utility for consumer A
                utility_A = utilities_A[i, j]

                #d. Calculate corresponding x1 and x2 for consumer B
                x1_B = 1 - np.linspace(0, 1, par.num_points)[i]
                x2_B = 1 - np.linspace(0, 1, par.num_points)[j]

                #e. Calculate utility for consumer B
                utility_B = self.cobb_douglas_utility(x1_B, x2_B, par.beta)
                
                #f. Check if utilities for both consumers are not lower than their initial endowments
                if utility_A >= self.cobb_douglas_utility(par.w1_A, par.w2_A, par.alpha) and \
                   utility_B >= self.cobb_douglas_utility(par.w1_B, par.w2_B, par.beta):
                    
                    #g. Update optimal allocation and utility for consumer A if utility is higher
                    if utility_A > max_utility_A:
                        max_utility_A = utility_A
                        optimal_x1_A = np.linspace(0, 1, par.num_points)[i]
                        optimal_x2_A = np.linspace(0, 1, par.num_points)[j]
                        
                        #h. Update optimal allocation for consumer B
                        optimal_x1_B = x1_B
                        optimal_x2_B = x2_B
        
        # printing the optimal demand for consumer A and B
        print("Optimal allocation for consumer A:")
        print("x1_A:", round(optimal_x1_A, 2))
        print("x2_A:", round(optimal_x2_A, 2))
        print("Maximum utility for consumer A:", round(max_utility_A, 2))
        
        print("\nCorresponding allocation for consumer B:")
        print("x1_B:", round(optimal_x1_B, 2))
        print("x2_B:", round(optimal_x2_B, 2))


    def optimize_utility_and_prices(self):
        "question 5B"
        "We need to make maximize the utility of A given the budget constraint that the utility of B cannot be lower than the utility of their initial endowments."
        par = self.par

        # Initialize variables to store optimal values
        optimal_utility_A = -np.inf
        optimal_x1A = None
        optimal_x2A = None
        optimal_utility_B = None
        optimal_x1B = None
        optimal_x2B = None

        #a. Loop through different combinations of x1A
        for x1A in np.linspace(0, 1, 101):

            #b. Loop through different combinations of x1A
            for x2A in np.linspace(0, 1, 101):

                #c. Calculate corresponding values of x1B and x2B
                x1B = 1 - x1A
                x2B = 1 - x2A
                
                #d. Calculate utility of consumer B
                utility_B = self.cobb_douglas_utility(x1B, x2B, par.beta)

                #e. Ensure utility of consumer B is not lower than initial endowment
                if utility_B < self.initial_utility_B:
                    continue
                
                #f. Calculate utility of consumer A
                utility_A = self.cobb_douglas_utility(x1A, x2A, par.alpha)

                #g. Update optimal values if utility of consumer A is higher
                if utility_A > optimal_utility_A:
                    optimal_utility_A = utility_A
                    optimal_x1A = x1A
                    optimal_x2A = x2A
                    optimal_utility_B = utility_B
                    optimal_x1B = x1B
                    optimal_x2B = x2B
        
        # Print optimal values for both consumers rounded to 2 decimals
        print("Optimal values for consumer A:")
        print("Optimal x1A:", round(optimal_x1A, 2))
        print("Optimal x2A:", round(optimal_x2A, 2))
        print("Utility for consumer A:", round(optimal_utility_A, 2))

        print("\nOptimal values for consumer B:")
        print("Optimal x1B:", round(optimal_x1B, 2))
        print("Optimal x2B:", round(optimal_x2B, 2))
        print("Utility for consumer B:", round(optimal_utility_B, 2))

        # Calculate the price of good 1 (p1) using the ratios of utilities and endowments
        p1 = (par.w1_A * optimal_utility_A + par.w2_A * optimal_utility_A) / (par.w1_A * optimal_x1A + par.w2_A * optimal_x2A)

        print("\nOptimal price for good 1 (p1):", round(p1, 2))



    def find_optimal_allocations_and_prices6a(self):
        "question 6A"
        "Maximizing in terms of the social planner to get the highest possible utility for both consumers"
        par = self.par

        # Initialize variables to store optimal values
        optimal_aggregate_utility = -np.inf
        optimal_x1A = None
        optimal_x2A = None
        optimal_x1B = None
        optimal_x2B = None
        optimal_utility_A = None
        optimal_utility_B = None

        # #a. Loop through different combinations of x1A
        for x1A in np.linspace(0, 1, 101):
           #b. Loop through different combinations of x1A
            for x2A in np.linspace(0, 1, 101):

                #c. Calculate corresponding values of x1B and x2B
                x1B = 1 - x1A
                x2B = 1 - x2A

                #d. Ensure x1B and x2B are within [0, 1]
                if 0 <= x1B <= 1 and 0 <= x2B <= 1:

                    #e. Calculate utility of consumer A
                    utility_A = self.cobb_douglas_utility(x1A, x2A, par.alpha)

                    #f. Calculate utility of consumer B
                    utility_B = self.cobb_douglas_utility(x1B, x2B, par.beta)

                    #g. Calculate aggregate utility
                    aggregate_utility = utility_A + utility_B

                    #h. Update optimal values if aggregate utility is higher
                    if aggregate_utility > optimal_aggregate_utility:
                        optimal_aggregate_utility = aggregate_utility
                        optimal_x1A = x1A
                        optimal_x2A = x2A
                        optimal_x1B = x1B
                        optimal_x2B = x2B
                        optimal_utility_A = utility_A
                        optimal_utility_B = utility_B

        # Calculate the price of good 1 (p1) using the ratio of utilities and endowments
        p1 = (par.w1_A * optimal_utility_A + par.w2_A * optimal_utility_A) / (par.w1_A * optimal_x1A + par.w2_A * optimal_x2A)

        # Print optimal values rounded to 2 decimals
        print("Optimal values:")
        print("Optimal x1A:", round(optimal_x1A, 2))
        print("Optimal x2A:", round(optimal_x2A, 2))
        print("Optimal x1B:", round(optimal_x1B, 2))
        print("Optimal x2B:", round(optimal_x2B, 2))
        print("Optimal utility for consumer A:", round(optimal_utility_A, 2))
        print("Optimal utility for consumer B:", round(optimal_utility_B, 2))
        print("Optimal aggregate utility:", round(optimal_aggregate_utility, 2))
        print("Optimal price for good 1 (p1):", round(p1, 2))

        
        return optimal_x1A, optimal_x2A, optimal_x1B, optimal_x2B, optimal_utility_A, optimal_utility_B, optimal_aggregate_utility, p1


    def generate_random_draws(self, num_draws):
        "Question 7"
        "This function simply generates random draws for 50 possible endowmnets"
        np.random.seed(42)  # Set the seed for reproducibility
        w1A = np.random.uniform(0, 1, num_draws)
        w2A = np.random.uniform(0, 1, num_draws)
        return w1A, w2A

    def random_draws_for_endowment(self, num_draws):
        "Question 7"
        "Here we store the random draws for the endowments"
        w1A_draws, w2A_draws = self.generate_random_draws(num_draws)
        return w1A_draws, w2A_draws
    

    def plot_consumers(self, w1_A, w2_A, num_points):
        "Question 8"
        "Given the the possible endowments in the previous question, we plot the market equilibirum in an edngeworthe box"
        
        # Given trouble with implementing this as a class, cobdouglas utility was defined again
        def cobb_douglas_utility2(x1, x2, alpha=0.5):
            return x1**alpha * x2**(1 - alpha)

        num_draws = 50

        # draws are generated using the same seed
        np.random.seed(42)  
        w1A_values = np.random.uniform(0, 1, num_draws)
        w2A_values = np.random.uniform(0, 1, num_draws)

        # Calculating x1A, x2A, x1B, and x2B for each random draw
        x1A_values = w1A_values
        x2A_values = 1 - w1A_values
        x1B_values = 1 - w1A_values
        x2B_values = w1A_values

        # Plot for the 50 endowments
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plotting the points for consumer A
        ax.scatter(x1A_values, x2A_values, color='blue', label='Consumer A (Primary Axis)', alpha=1)
        ax.set_xlabel('x1A')
        ax.set_ylabel('x2A')
        ax.set_title('Consumer A and B Points')

        # Create a secondary axis for consumer B
        ax2 = ax.secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))
        ax2.set_xlabel('x1B')

        ax3 = ax.secondary_yaxis('right', functions=(lambda y: 1 - y, lambda y: 1 - y))
        ax3.set_ylabel('x2B')

        # Plot consumer B points
        ax.scatter(x1B_values, x2B_values, color='red', label='Consumer B (Secondary Axis)', alpha=1)

        # Calculate the utility of the endowment for consumer A
        utility_endowment_A = cobb_douglas_utility2(w1_A, w2_A, self.par.alpha)

        # Generate points for x1A and x2A
        x1_values_A = np.linspace(0, 1, num_points)
        x2_values_A = np.linspace(0, 1, num_points)

        # Recalculating the utilities for the grid of points
        utilities_A = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                utilities_A[i, j] = cobb_douglas_utility2(x1_values_A[i], x2_values_A[j], self.par.alpha)

        # Plotting the points where consumer As utility is greater or equal than the initidal endowment
        satisfactory_points_A = np.argwhere(utilities_A >= utility_endowment_A)
        satisfactory_x1_values_A = x1_values_A[satisfactory_points_A[:, 0]]
        satisfactory_x2_values_A = x2_values_A[satisfactory_points_A[:, 1]]
        ax.scatter(satisfactory_x1_values_A, satisfactory_x2_values_A, color='green', label='Satisfactory Points A', alpha=0.3)

        # By finding the best points for consumer A the best points for consumer B are found
        utility_endowment_B = cobb_douglas_utility2(1 - w1_A, 1 - w2_A, self.par.alpha)

        # Plot the points where consumer B's utility is greater than or equal to the endowment
        satisfactory_points_B_x = 1 - satisfactory_x1_values_A
        satisfactory_points_B_y = 1 - satisfactory_x2_values_A
        ax.scatter(satisfactory_points_B_x, satisfactory_points_B_y, color='purple', label='Satisfactory Points B', alpha=0.3)

        ax.legend()

        plt.show()
    
   