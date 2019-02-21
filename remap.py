

def remap(x, out_min, out_max):
    return (x - x.min()) * (out_max - out_min) / (x.max() - x.min()) + out_min
