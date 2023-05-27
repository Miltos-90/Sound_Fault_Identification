""" Collection of functions common to all scales-related modules. """


import numpy as np
from typing import Literal
from .mel   import filterbank as melBank
from .bark  import filterbank as barkBank

def makeEinsumNotation(matrix1Dims: int, matrix2Dims: int, axis: int) -> str:
    """ Dynamic generation of the einsum notation string based on the 
        dimensions of the input matrices and the axis of the first matrix
        that will be multipled with the weights of the Mel filterbank.
        Inputs:
            matrix1Dims: Number of dimensions of the first matrix
            matrix2Dims: Number of dimensions of the second matrix
            axis       : Axis of the first matrix that will be operated upon
        Outputs:
            notation: Einsum notation string
    """

    def makeMatrixStr(nDims: int, alphabet: list) -> list:
        """ Generates the subscript notation for a <nDims>-dimensional matrix. """

        ix, sub = 0, []
        while ix < nDims and alphabet:
            # Grab the first available letter of the (remaining) 
            # alphabet and assign it to the next dimension.
            sub.append(alphabet.pop(0))
            ix += 1

        return sub

    if axis >= matrix1Dims: raise ValueError(f'Input axis should be lower than {matrix1Dims}.')

    alphabet = list('abcdefghijklmnopqrstuvwxyz')

    # Check matrix shape
    maxLen = len(alphabet) - 2
    if matrix1Dims > maxLen: # Reserve one letter for the weights matrix
        raise RuntimeError(f'Input matrix should have lower than {maxLen} dimensions.')

    # Make LHS
    # Make strings for the two matrices
    str1 = makeMatrixStr(matrix1Dims, alphabet)
    str2 = makeMatrixStr(matrix2Dims, alphabet)

    # Assign the same subscript to the axes that will be multiplied
    commonLetter = str1[axis]
    str2[0]      = commonLetter

    # Make right-hand-side: Start with ther subscripts of the first matrix, up to the subscript that will be multiplied upon
    # Subscript of the second matrix that is not the common letter should substitute the subsript of the RHS appearing on element <axis>.
    # All trailing elements should be moved to the right by 1 element

    if axis == 0:
        head = ''.join([s for s in str2 if s != commonLetter])
        tail = ''.join([s for s in str1 if s != commonLetter])
        
    else: # axis > 0
        head = ''.join(str1[:axis])
        tail = ''.join(str2[axis:] + str1[axis + 1:])

    # Convert to strings and make left-hand-side of the einsum notation string
    str1 = ''.join(str1)
    str2 = ''.join(str2)
    lhs  = ', '.join([str1, str2])
    rhs  = ''.join([head, tail])
    notation = ' -> '.join([lhs, rhs])

    return notation

def energy(
    amplitudes: np.array, sampleFrequency: int, numFilters: int, scale: Literal["mel", "bark"], axis: int) -> np.array:
    """ Computes the critical band energy in the Mel scale.
        Inputs:
            amplitudes     : Matrix of the amplitudes of the power spectrum
                NOTE: DIMS can be any arbitrary number of dimensions, so long as the input axis <axis>
                      contains <Num. frequencies> elements.
            sampleFrequency: Sampling frequency of the signal
            numFilters     : Number of Mel bands
            scake          : Scale to be used
            axis           : Axis along which to compute the critical energy
        Outputs:
            Critical energy of the Mel bands [DIMS]. 
                Dimensions are exactly the same as the dimensions of the input <amplitudes> matrix, with the
                exception of the axis <axis>. The latter will contain <numBands> elements instead of
                <Num. frequencies> elements.
    """

    numFFT = amplitudes.shape[axis] * 2

    if   scale == 'mel' : filters = melBank( numFFT, numFilters, sampleFrequency)
    elif scale == 'bark': filters = barkBank(numFFT, numFilters, sampleFrequency)

    notation = makeEinsumNotation(amplitudes.ndim, filters.ndim, axis = axis)
    return np.einsum(notation, amplitudes, filters)