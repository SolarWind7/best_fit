# best_fit
Given an estimation method, such as OLS, GLS, LOGIT, or PROBIT, searches function specification space up to three interaction levels from variables specified in a formula to find best fitting selections of variables based on a given information criteria, such as AIC and BIC.  Use it to examine polynomial and interaction terms up to 3rd degree.  Implemented in Python 3.  Tested with Python 3.8, 3.10.

Help on module best_fit:

NAME
    best_fit

FUNCTIONS

    expand_vars(formula, level)
        returns all possible combination of the independent variables in the formula based on the level of interaction,
        where 1 means no interaction, 2 means chose two combinations of all variables are added, and 3 means chose three 
        combinations of all variables are added.  Square and cube terms are also added at the appropriate levels.
    
    find_best_fit_genetic(formula, data, level=1, criteria='bic', classifier='ols', pop_size=100, number_iterations=50, crossover_rate=0.9, mutation_rate=0.1)
        Searches for specification with best fit using genetic algorithm and criteria.
         Uses statsmodels OLS, GLS, Logit, or Probit, depending on the classigier parameter ('ols', 'gls', 'logit', 'probit')
        The formula is in the form 'y ~ a + b + c', data must be a pandas dataframe with the variables 
        referred to in formula as column names.
        When level == 2, also adds interaction terms and squared terms on all variables.
        When level == 3, adds level 3 interaction terms and cubed terms on all variables.
        Arguments (hyperparameters) for the genetic algorithm: pop_size, number_iterations, crossover_rate, mutation_rate
    
    problem_domain_size(formula, minvars=1, maxvars=None, level=1)
        Calculates and prints the problem domain size that is the total number of models
        that we could attempt to estimate and the total number of variables to chose from
        if calling search_for_best_fit with these arguments.
        
    search_for_best_fit(formula, data, minvars=1, maxvars=None, level=1, criteria='bic', classifier='ols')
        Searches for specification with best fit using criteria.  Fits all combinations
        with the minimum size of variables, given by minvars, to the maximum size of mvariables, given by maxvars.
        Uses statsmodels OLS, GLS, Logit, or Probit, depending on the classigier parameter ('ols', 'gls', 'logit', 'probit').
        The formula is in the form 'y ~ a + b + c', data must be a pandas dataframe with the variables 
        referred to in formula as column names.
        When level == 2, also adds interaction terms and squared terms on all variables.
        When level == 3, adds level 3 interaction terms and cubed terms on all variables.

<meta name="google-site-verification" content="ZYgaXP-d4anw82tkJOdTCXCWTwOd3BM2r-zoe9wSP5A" />
