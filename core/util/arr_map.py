from numpy import array

def arr_map(func, *arrs):
    return array(list(map(func, *arrs)))
