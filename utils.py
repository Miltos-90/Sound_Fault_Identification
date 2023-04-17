import os
import numpy as np 
import pandas as pd


def makeIndex(filePath):
    """ Indexes all files and extracts some metadata and the target. """

    data = list()

    for root, dirs, files in os.walk(filePath):

        for filename in files:
            nm, ext = os.path.splitext(filename)

            if ext.lower().endswith('.wav'):
                
                fullpath = os.path.join(os.path.abspath(root), filename)
                rootPart = root.split('\\')
                target   = rootPart[-1]                     # Normal / abnormal
                machType = rootPart[-3]                     # Machine type
                machId   = int(rootPart[-2].strip('id_'))   # Machine ID
                noise    = rootPart[-4].strip(              # Background noise level
                    '_'.join(machType))

                data.append((fullpath, machId, machType, noise, target))

    cols = ['filepath', 'machine_id', 'machine_type', 'background_noise', 'target']
    return pd.DataFrame(data, columns = cols)