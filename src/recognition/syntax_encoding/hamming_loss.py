# TODO: Implement Hamming Loss

"""

L(w1`, w2`) = [max(margin * dh(w1, w2) - d(w1`, w2`), 0)] + [max(d(w1`, w2`) - margin * (dh(w1, w2) + 1), 0)]

where 
    w` - output of neural network with given "w" word
    d(w1`, w2`) = (w1` - w2`) ** 2, 
    dh(w1, w2) - Hamming distance between words w1, w2

"""