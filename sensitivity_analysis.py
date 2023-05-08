#%%
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

import warnings
warnings.simplefilter(action="default") # Make sure warning is printed in VSCode
from probability_integral_transform import ProbabilityIntegralTransformation
import doe
import helpers

#%%
""" Generate an example use case and apply (validate) the transformation. """
N = 200 # Num samples
f = 2   # Num dimensions (features)

X = np.empty(shape = (N, f))
# Random variable (bi-modal)
X[:, 0] = np.concatenate((np.random.normal(-2, 0.5, int(0.7 * N)), np.random.normal(2, 0.1, int(0.3 * N))))
X[:, 1] = np.concatenate((np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N))))

# Construct a 3D function
def fun(x, y):
    #return 3 * (1 - x) ** 2 * np.exp( -(x ** 2) - (y + 1) ** 2) \
    #    - 10*(x / 5 - x ** 3 - y ** 5) * np.exp(- x ** 2 - y ** 2) \
    #    - 1/3 * np.exp(-(x + 1) ** 2 - y** 2)
    return -x + y

""" Apply transformation """
tf      = ProbabilityIntegralTransformation()   # Instantiate
Xtrans  = tf.fit_transform(X)                   # Transform
Xrev    = tf.inverse_transform(Xtrans)          # Inverse
y       = fun(X[:, 0], X[:, 1]).reshape(-1, 1)  # Make target

#plotTransformation(X, Xtrans, Xrev, dim = 0)
#plotSurface(X, fun)

#%%
""" Run a regression analysis. By observing the coefficient values and standard errors, one 
    can understand if a certain variable can be removed. 
"""
X = sm.add_constant(Xtrans)
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

#%%

# TODO: Scale all design matrices using the equation at the bottom of page 71 (footnote 5)
# TODO: Mimic these: 
#       https://pythonhosted.org/pyDOE/factorial.html#fractional-factorial, 
#       https://nl.mathworks.com/help/stats/design-of-experiments-1.html?s_tid=CRUX_lftnav
#       https://tirthajyoti.github.io/DOE/DOE.html
# TODO: The two functions below should distinguish between lowercase and uppercase
full = doe.fullFactorialDoE(levels = [2, 2])
frac = doe.twoLevelFractionalFactorialDoE('a b c abc')
# TODO: Write unit tests -> https://www.itl.nist.gov/div898/handbook/pri/section3/pri3347.htm
#print(full)
print(frac)



# %%
import doe
import numpy as np
from itertools import combinations
from typing import Tuple
import warnings

def makeInelligibleEffects(terms: list) -> list:
    """ Generates the ineligible effects set from the factors in the requirements set. """

    interactions = [getInteraction(t1, t2) for t1, t2 in combinations(terms, 2)]
    iEffects     = terms + interactions

    return sorted(set(iEffects))

def getInteraction(t1: str, t2: str) -> str: 
    """ Computes the generalised interaction between the two terms. """

    if t1 == t2: return '1' # Mean effect
    
    # Grab unique elements from the two terms
    l = set(t1).symmetric_difference(set(t2))

    # Sort and join
    return "".join(sorted(l))

def getMinSampleSize(terms: list, resolution: int) -> int:
    """ Step 2: Computes the minimum required sample size for the 
        given term list and resolution number.
    """

    # Get highest interaction number in the requirements set
    termSize = [len(e) for e in terms]
    n        = len(terms)
    minSize  = max(np.ceil( np.log2(n + 1) ), max(termSize)) # Valid for a resolution 3 design

    # For resolution 5 find all terms up to 2-factor interactions
    if resolution >= 5:   minSize = max(np.ceil(np.log2(1 + n + n * (n - 1) / 2)), minSize)

    # For resolution 4, the number of terms should at least equal to a resolution 3 design
    elif resolution == 4: minSize = max(np.ceil(np.log2(2 * n)), minSize)

    return minSize.astype(int)

def getNumBase(terms: list) -> int:
    """ Returns the number of base factors in the model. """
    return sum([len(t) == 1 for t in terms])

def getBase(terms: list) -> list:
    """ Returns base factors in the model. """
    return [t for t in terms if len(t) == 1]

def iterateAlphabet(lowercase: bool = True):
    """ Generator that iterates through the English alphabet 
        yielding each letter one at a time. 
    """
    
    if lowercase:
        for letter in range(ord('a'), ord('z') + 1): yield chr(letter)
    else:
        for letter in range(ord('A'), ord('Z') + 1): yield chr(letter)

def checkNumTerms(numFactors: int, sampleSize: int) -> list:
    """ Checks if the current settings result in a valid number of terms to 
        be estimated. Raises warning if not and returns the generators for a 
        full factorial design.
    """

    if numFactors - sampleSize <= 0:
        warnings.warn("Full factorial design is required.")
        
        # Iterate through the alphabet for an appropriate number of times
        # and return the terms for a full-factorial design
        gen = []
        tmp = min(sampleSize, numFactors)
        for idx, letter in enumerate(iterateAlphabet()):

            if idx > tmp: break
            gen.append(letter)
        return gen

    else:
        return None

def makeSampleSize(terms: list, sampleSize: int, resolution: int) -> Tuple[np.array, list]:
    """ Computes the sample size (number of runs) required for the design, given
        a list of names for the terms <terms>, the sample size of the design <sampleSize>,
        and the required resolution <resolution>.
        Raises value error for invalid specified sample sizes, or invalid resolution.
    """

    numFactors    = getNumBase(terms)
    minSampleSize = getMinSampleSize(terms, resolution)
    fullFactGen   = [] # Generator list is populated in case a full factorial is needed
    notSpecified  = sampleSize is None
    notValid      = not isinstance(sampleSize, int) or sampleSize < minSampleSize
    
    if notSpecified: # All designs will be searched, starting from the smallest possible one
        sampleSizeVec = np.arange(minSampleSize, numFactors)

    elif notValid: # Invalid user-specified sample size
        msg = f'Sample size must be an integer higher than or equal to {minSampleSize}'
        raise ValueError(msg)

    else: # Valid number of runs
        # Convert to numpy array for consistency with the previous case
        sampleSizeVec  = np.array([sampleSize])

        # Check if a valid number of model terms to be estimated arises
        fullFactGen = checkNumTerms(numFactors, sampleSize)

    # Check if a valid sample size vector has been obtained.
    if sampleSizeVec.size == 0:
        msg = f'Resolution must be an integer lower than {resolution}.'
        raise ValueError(msg);

    return sampleSizeVec, fullFactGen

def makeBasicTerms(terms: list, base: list, sampleSize: int):
    """ Selects the basic factors of the model from the list of terms. """

    # Get the term with the highest level of interaction (i.e. highest character count
    termSize = [len(t) for t in terms]
    maxInter = terms[np.argmax(termSize)]

    # Make base terms
    idx, n = 0, len(base)
    basic  = list(maxInter)

    while len(basic) < sampleSize and idx < (n - 1):
        b = base[idx]
        if b not in basic: basic.append(b)
        idx +=1

    return sorted(basic)

def makeAddedTerms(base: list, basic: list) -> list:
    """ Returns the added factors of the model given its base and basic terms. """

    added = list(set(base).difference(set(basic)))
    return sorted(added)

def makeBasicEffectsGroup(terms: list, basic: list, resolution: int) -> list:
    """ Generates the basic effects of the model, i.e. all combinations of the basic 
        terms according to the given resolution and the model,
    """

    basicEffects = []
    for term in basic:

        if not basicEffects: # Empty list
            # Add the new term in the basicEffects list
            basicEffects.append(term)
        else:
            # Evaluate all interactions between the new term and 
            # all terms in the basicEffects list
            interactions = [getInteraction(e, term) for e in basicEffects]

            # Add the new term in the basic effects, and then all interactions
            basicEffects.append(term)
            basicEffects.extend(interactions)
    
    # Remove model terms
    for t in terms:
        if t in basicEffects: basicEffects.remove(t)

    return basicEffects

def makeBasicGroup(terms: list, sampleSize: int, resolution: int) -> Tuple[list, list]:
    """ Step 3: Selects a set of basic factors and forms the basic effects group, given
        a list of names for the terms <terms>, the sample size of the design <sampleSize>,
        and the required resolution <resolution>.
     """

    baseTerms    = getBase(terms)
    basicTerms   = makeBasicTerms(terms, baseTerms, k)
    addedTerms   = makeAddedTerms(baseTerms, basicTerms)
    basicEffects = makeBasicEffectsGroup(terms, basicTerms, resolution)
    
    return baseTerms, addedTerms, basicEffects

def makeEligibleEffects(basicEffects: list, addedTerms: list) -> np.array:
    """ Step 4: Generates table of eligible effects. """

    tab  = np.empty(
            shape = (len(basicEffects), len(addedTerms)), 
            dtype = 'U52'
            )

    for i, effect in enumerate(basicEffects):
        for j, term in enumerate(addedTerms):

            interaction = getInteraction(effect, term)
            if interaction in inelligible: interaction = ''

            tab[i, j] = interaction

    return tab

def evaluateInteractions(
    generator: str, contrasts: list, inelligibleSet: list) -> Tuple[bool, list]:
    """ Step 8: Checks whether all generalized interactions between the current generator
        and the defining contrasts group are eligible
    """
    
    empty        = all([e is None for e in contrasts])
    eligible     = True
    interactions = []

    if not empty:
        for c in contrasts:
            if c is not None:
                interaction = getInteraction(c, generator)
                interactions.append(interaction)
                if interaction in inelligibleSet: 
                    eligible = False
                    break
            else:
                interactions.append(None)
    
    return eligible, interactions




def addGenerator(
    gens: list, col: int, curSelection: list, addedTerms: list,
    baseTerms:list, basicEffects: list) -> list:
    
    idx = addedTerms[col]
    val = basicEffects[curSelection[col]]
    gens[baseTerms.index(idx)] = val

    return gens

def addContrast(contrasts: list, index: int, generator: str, interactions: list) -> list:
                
    contrasts[index] = generator
    for j in range(index): 
        contrasts[index + j + 1] = interactions[j]
    
    return contrasts

def removeContrast(contrasts: list, index: int) -> list:

    contrasts[index] = None
    for j in range(index): contrasts[index + j + 1] = None
    
    return contrasts

##################################################
def makeGeneratorString(gens, baseTerms, contrasts):
    print('----- Design found -----')
    for idx, val in enumerate(baseTerms):
        if gens[idx] is None:
            gens[idx] = baseTerms[idx]
    print(' '.join(gens))
    r = min([len(e) for e in contrasts])
    print(f'Resolution found: {r}')
    #print(f'Contrasts: {contrasts}')
    print('------------------------')

termStr = 'a b c d e f g h ab be bc bd bf be';
#termStr = 'a b c d ab';
#termStr = 'a b c d';
#termStr = 'a b c d e ad de';
#termStr = 'a b c d e f ab be';
#termStr = 'a b c d e f g h ad de';
#termStr = 'a b c d e f g h i gh gf ad de abc';

# Default values
k = None
resolution = 3

# Check input.
if not isinstance(resolution, int) or resolution < 3: 
    raise ValueError('Resolution must be an integer higher than or equal to 3.')

# TODO: Ensure that all base factors are added in the string from the interactions
# for instance, a b c d ae should be converted to: a b c d ->e<- ae

tParsed     = doe._parse(termStr)                   # Step 0
terms       = [t['name'] for t in tParsed]
inelligible = makeInelligibleEffects(terms)         # Step 1
kVec, ffGen = makeSampleSize(terms, k, resolution)  # Step 2
mVec        = getNumBase(terms) - kVec
# if ffGen: return ffGen


for k, m in zip(kVec, mVec):    # Searh one or more designs as required
    baseTerms, addedTerms, basicEffects = makeBasicGroup(terms, k, resolution) # Step 3
    tab = makeEligibleEffects(basicEffects, addedTerms) # Step 4

    break

# Step 5. Initialize for search through table
nrows, ncols = tab.shape
cursel    = [nrows] * m                 # Current selection of generators from table
gens      = [None] * getNumBase(terms)  # Generator from each column
contrasts = [None] * (2 ** m-1)         # Defining contrasts group
col, resfound = 0, 0 # Current column and best resolution found so far

while resfound < resolution:
    #print()

    cursel[col] -= 1
    if cursel[col] >= 0: # Step 7: Select the next available effect in this column

        gen = tab[cursel[col]][col]
        nident = 2 ** col - 1
        #print(col, cursel, gen, end = ' ')
        if gen == '': continue # Ineligible effect on this cell. Go back to step 7
        eligible, interactions = evaluateInteractions(gen, contrasts, inelligible) # Step 8
        
        if not eligible: # If not eligible, return to step 7
            #print(f'Interaction {gen} x {c} = {getInteraction(gen, c)} is not eligible. Moving on')
            continue

        else: # Eligible.
            # Perform Step 9: Extend the defining contrasts group
            #print(f'All interactions eligible. Adding generator {gen}')
            contrasts = addContrast(contrasts, nident, gen, interactions)
            gens      = addGenerator(gens, col, cursel, addedTerms, baseTerms, basicEffects)

            if col == ncols - 1: # Last column has been reached
                makeGeneratorString(gens.copy(), baseTerms, contrasts)
                contrasts = removeContrast(contrasts, nident)
                continue # Go back to step 7
            
            else: # Not last column
                # Perform step 6
                #print('Moving forward')
                #print(f'Contrasts: {contrasts}')
                col += 1
                cursel[col] = nrows

    else: # Step 10
        if col == 0: break # Current column is the first. Perform Step 11 (i.e. exit)
        else:
            col -= 1 # Move to the previous column without changing the pointer, and perform Step 7
            nident = 2 ** col - 1
            #print('Backing up')
            contrasts = removeContrast(contrasts, nident)
            #print(f'Contrasts: {contrasts}')

""" 
print()
print()
print('Table')
for gg, t in enumerate(addedTerms):
    if gg == 0: print('_\t', end = '')
    print(f'{t}', end = '\t')
print()

for i in range(len(basicEffects)):
    print(basicEffects[i], end = '\t')
    for j in range(len(addedTerms)):
        print(tab[i][j], end = '\t')
    print()

"""

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
