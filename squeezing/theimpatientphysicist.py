""" Module to store calculated data and find it again, for an
impatient physicist

"""
import pickle
import os
import csv


def hash_dict(params):
    """ Calculate the hash value corresponding to the tuple of values
    in a dictionary

    Parameters
    ----------
    params : dict
        A dictionary containing all model parameters

    Returns
    -------
    hash : float
        The hash value the tuple containing the dictionary values

    """
    return hash(tuple(params.values()))


def is_pickled(params):
    """ Checks if Pickeled object with matching parameters exits

    Parameters
    ----------
    params : dict
        A dictionary containing all model parameters

    Returns
    -------
    : obj
        Pickled object matching the hash of the dictionary

    """
    hash_value = hash_dict(params)
    pickle_path = "/Users/emanuel/Uni/Projects/" \
        + "Parametric_quantum_amplification/" \
        + "Codebase/QuPulses/squeezing/Pickled/"
    for filename in os.listdir(pickle_path):
        if filename == str(hash_value):
            with open(f"{pickle_path}/{filename}", 'rb') as file:
                object_tmp = pickle.load(file)
                print("I found a pickeled object called "
                      f"\"{filename}\" matching your description!")
                return object_tmp
    return None


def save_data(fname, header, data, comment=""):
    """ Saves data to CSV file in folder Data

    Parameters
    ----------
    fname : str
        name of file
    header : str
        Header to put into CSV file
    data : 2D list
        2D Data array to be saved
    comment : str, optional
        Comment to put into CSV file. Defaults to empty string.

    """
    with open("Data/{}.csv".format(fname), 'w', newline='') as file:
        file.write("# "+comment+'\n')
        file.write("{}\n".format(header))
        writer = csv.writer(file, lineterminator='\n')
        for i in range(len(data)):
            writer.writerow(data[i])
