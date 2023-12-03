import sys
import re
import itertools
import math
import statsmodels.formula.api as smf
from operator import attrgetter
from operator import methodcaller
from numpy.random import randint, rand

import numpy as np
import patsy
import numpy.linalg

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

def expand_vars(formula, level):
    """returns all possible combination of the independent variables in the formula based on the level of interaction,
    where 1 means no interaction, 2 means chose two combinations of all variables are added, and 3 means chose three 
    combinations of all variables are added.  Square and cube terms are also added at the appropriate levels."""
    dep_var, vars = re.split(r'\s*\~\s*', formula)
    vars = re.split(r'\s*\+\s*', vars)
    
    if level == 2:
        squares = ['np.power(' + a + ',2)' for a in vars]
        interactions = [a + ':' + b for a,b in itertools.combinations(vars, level)]
        vars.extend(squares)
        vars.extend(interactions)

    if level == 3:
        squares = ['np.power(' + a + ',2)' for a in vars]
        cubes = ['np.power(' + a + ',3)' for a in vars]
        interactions1 = [a + ':' + b for a,b in itertools.combinations(vars + squares, 2) if not (a in squares and b in squares) and not re.search('np\.power\('+a+'\,2\)', b) and not re.search('np\.power\('+b+'\,2\)', a)]
        interactions2 = [a + ':' + b + ':' + c for a,b,c in itertools.combinations(vars, 3)]
        vars.extend(squares)
        vars.extend(cubes)
        vars.extend(interactions1)
        vars.extend(interactions2)

    return dep_var, vars


def problem_domain_size(formula, minvars = 1, maxvars = None, level = 1):
    """Calculates and prints the problem domain size that is the total number of models
     that we could attempt to estimate and the total number of variables to chose from
     if calling search_for_best_fit with these arguments."""
    dep_var, vars = expand_vars(formula, level)

    if maxvars is None:
        maxvars = len(vars)

    modelcount = 0
    n = len(vars)
    for r in range(minvars, maxvars+1):
        modelcount +=  math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

    print('We are looking at estimating ' + str(modelcount) + ' models from ' + str(n) + ' variables.')
    return modelcount, n


def find_best_fit_genetic(formula, data, level = 1, criteria = 'bic', classifier = 'ols', \
                    pop_size = 100, number_iterations = 50, crossover_rate = .9, mutation_rate = 0.1):
    """Searches for specification with best fit using genetic algorithm and criteria.
 Uses statsmodels OLS.
 The formula is in the form 'y ~ a + b + c', data must be a pandas dataframe with the variables 
 referred to in formula as column names.
 When level == 2, also adds interaction terms and squared terms on all variables.
 When level == 3, adds level 3 interaction terms and cubed terms on all variables.
 Arguments (hyperparameters) for the genetic algorithm: pop_size, number_iterations, crossover_rate, mutation_rate
"""

    dep_var, vars = expand_vars(formula, level)
    
    modelcount = 0
    n = len(vars)
    for r in range(1, n+1):
        modelcount +=  math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

    print('We are going to attempt genetic search to find a best fit from ' + str(len(vars)) + ' variables.  The search space consists of a total of ' + str(modelcount) + ' distinct equations.')
    print('Method: ' + str(classifier) + '.  Criteria: ' + str(criteria) + '.  Level: ' + str(level))

    if classifier in ('ols', 'gls', 'logit', 'probit'):
        classifier = attrgetter(classifier)
        if classifier in ('ols',):
            fitmethodcaller = methodcaller('fit')
        else:
            fitmethodcaller = methodcaller('fit', disp=0)
    else:
        raise ValueError(str(classifier) + " is not a suppoted classifier method.")

    def generate_model(bits):
        myvars = [vars[i] for i in range(len(bits)) if bits[i] == 1]
        myformula = dep_var + '~' + '+'.join(myvars)
        model = fitmethodcaller(classifier(smf)(formula=myformula, data=data))
        return myformula, model
    
    def selection(pop, k=3):
        selected = randint(len(pop))
        try:
            selected_aic = generate_model(pop[selected])[1].bic
        except patsy.PatsyError as e:
            print(str(e))
            del pop[selected]
            return selection(pop)
        except numpy.linalg.LinAlgError as e:
            #print(str(e))
            del pop[selected]
            return selection(pop)

        for i in randint(0, len(pop), k-1):
            try:
                i_aic = generate_model(pop[i])[1].bic
            except patsy.PatsyError as e:
                print(str(e))
                del pop[i]
                i -= 1
            except numpy.linalg.LinAlgError as e:
                #print(str(e))
                del pop[i]
                i -= 1
            else:
               if i_aic < selected_aic:
                    selected = i

        sys.stdout.write('.')
        sys.stdout.flush()

        return pop[selected]

    
    def crossover(parent1, parent2):
        if rand() < crossover_rate:
            crossover_point = randint(1, len(parent1)-2)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        return (child1, child2)


    def mutation(bits):
        for i in range(len(bits)):
            if rand() < mutation_rate:
                bits[i] = 1 - bits[i]


    def initialize(pop):
        try:
            return generate_model(pop[0])
        except patsy.PatsyError as e:
            print(str(e))
            del pop[0]
            return initialize(pop)
        except numpy.linalg.LinAlgError as e:
            #print(str(e))
            del pop[0]
            return initialize(pop)


    number_bits = len(vars)
    pop = [randint(0, 2, number_bits).tolist() for i in range(pop_size)]
    
    bestformula, bestmodel = initialize(pop)

    print("Initialized best:")
    print(bestformula)
    print(bestmodel.bic)
    print()

    try:
        for gen in range(number_iterations):
            print()
            print('Generation ' + str(gen) + '. ' + str(100*gen/number_iterations) + '% done.')
            for i in range(len(pop)):
                try:
                    myformula, mymodel = generate_model(pop[i])
                except patsy.PatsyError as e:
                    print(str(e))
                except numpy.linalg.LinAlgError as e:
                    #print(str(e))
                    pass
                else:
                    if mymodel.bic < bestmodel.bic:
                        bestformula, bestmodel = myformula, mymodel
                        print()
                        print("Generation " + str(gen) + " new best:")
                        print(myformula)
                        print(mymodel.bic)
                        print()
                finally:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            
            sys.stdout.write(' ')
            sys.stdout.flush()
            selected = [selection(pop) for i in range(pop_size)]

            children = []
            for i in range(0, len(selected), 2):
                parent1, parent2 = selected[i], selected[i+1]
                for child in crossover(parent1, parent2):
                    mutation(child)
                    children.append(child)

            pop = children
    except KeyboardInterrupt as ki:
        print(ki)
        print("Found best model: " + bestformula)
        print("BIC: " + str(bestmodel.bic))
        return bestformula, bestmodel
    except Exception as err:
        print("Found best model: " + bestformula)
        print("BIC: " + str(bestmodel.bic))
        raise err
    else:
        print("Found best model: " + bestformula)
        print("BIC: " + str(bestmodel.bic))
        return bestformula, bestmodel
        

def search_for_best_fit(formula, data, minvars = 1, maxvars = None, level = 1, criteria = 'bic', classifier = 'ols'):
    """Searches for specification with best fit using criteria.  Fits all combinations
 with the minimum size of variables, given by minvars, to the maximum size of mvariables, given by maxvars.
 Uses statsmodels OLS.
 The formula is in the form 'y ~ a + b + c', data must be a pandas dataframe with the variables 
 referred to in formula as column names.
 When level == 2, also adds interaction terms and squared terms on all variables.
 When level == 3, adds level 3 interaction terms and cubed terms on all variables.
"""

    dep_var, vars = expand_vars(formula, level)
    
    if maxvars is None:
        maxvars = len(vars)
        
    modelcount = 0
    n = len(vars)
    for r in range(minvars, maxvars+1):
        modelcount +=  math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

    print('We are going to attempt to estimate ' + str(modelcount) + ' models by exhaustive searching through combinations of ' + str(n) + ' variables.')
    print('Method: ' + str(classifier) + '.  Criteria: ' + str(criteria) + '.')

    if classifier in ('ols', 'gls', 'logit', 'probit'):
        classifier = attrgetter(classifier)
        if classifier in ('ols',):
            fitmethodcaller = methodcaller('fit')
        else:
            fitmethodcaller = methodcaller('fit', disp=0)
    else:
        raise ValueError(str(classifier) + " is not a suppoted classifier method.")

    j = 0
    bestmodel = None
    bestformula = None
    for i in range(minvars-1, maxvars):
        for combination in itertools.combinations(vars, i+1):
            j += 1
            print('Iteration ' + str(j))
            myformula = dep_var + '~' + '+'.join(combination)
            print(myformula)
            try:
                model = fitmethodcaller(classifier(smf)(formula=myformula, data=data))
            except patsy.PatsyError as e:
                print(str(e))
            except numpy.linalg.LinAlgError as e:
                print(str(e))
            else:
                print(model.bic)
            
                if bestmodel is None:
                    bestmodel = model
                    bestformula = myformula
                elif model.bic < bestmodel.bic:
                    bestmodel = model
                    bestformula = myformula
            
    print("Found best model: " + bestformula)
    print("BIC: " + str(bestmodel.bic))
    return bestformula, bestmodel
    




