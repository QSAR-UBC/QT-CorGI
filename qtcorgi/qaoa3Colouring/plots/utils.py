def get_inds(x_values, params):
    """
    Gets indexes of data to include

    Args:
        data_keys (list): list of values to be plotted on x-axis
        params (list): list of parameters to be plotted on current itteration

    Returns:
        inds (list): indexes of the desired params in data keys
    """
    inds = []
    for ind, param in enumerate(x_values):
        if param in params:
            inds.append(ind)
    return inds
