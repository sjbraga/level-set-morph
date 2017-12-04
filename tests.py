import morphsnakes
import levelset
import visual

import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl
from PIL import Image

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def initial_phi(shape, w1=4, w2=4):
    c0=4
    nrow, ncol=shape
    phi=c0*np.ones((nrow,ncol))
    phi[w1+5:-w1-1, w2+1:-w2-1]=-c0
    return phi




def test_trad_levelset_obj():
    img = ppl.imread("testimages/twoObj.bmp")
    
    g = levelset.border(img)

    ls = levelset.Levelset(g)
    ls.levelset = initial_phi(img.shape)

    ppl.figure()
    return visual.evolve_visual(ls, num_iters=150, background=img)

def test_trad_nodule():
    img = ppl.imread("testimages/mama07ORI.bmp")
    imgbw = rgb2gray(img)

    g = levelset.border(imgbw, sigma=50.0, h=8)

    ls = levelset.Levelset(g, step=1, v=-1)
    ls.levelset = initial_phi(imgbw.shape, 120, 110)

    ppl.figure()
    visual.evolve_visual(ls, num_iters=400, background=imgbw, save_every_iter=50, save_img=True, show_axe3=True)


def test_morph_nodule():
    # Load the image.
    img = imread("testimages/mama07ORI.bmp")[...,0]/255.0
    
    # g(I)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
    mgac.levelset = circle_levelset(img.shape, (100, 126), 20)
    
    # Visual evolution.
    ppl.figure()
    visual.evolve_visual(mgac, num_iters=45, background=img, save_every_iter=5, save_img=True, show_axe3=False)

def test_morph_obj():
    img = imread("testimages/twoObj.bmp")/255.0

    gI = morphsnakes.gborders(img, alpha=1000, sigma=1)

    # ppl.hist(gI, normed=True)

    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.28, balloon=-1)
    mgac.levelset = circle_levelset(img.shape, (40,40), 40)

    ppl.figure()
    return visual.evolve_visual(mgac, num_iters=40, background=img, save_every_iter=10)


if __name__ == '__main__':
    print("""""")
    # test_trad_levelset_obj()
    # test_morph_obj()

    # test_trad_nodule()
    test_morph_nodule()


    ppl.show()
