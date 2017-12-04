import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl

def evolve_visual(msnake, levelset=None, num_iters=20, background=None, save_img=False, save_every_iter=20, show_axe3=False):
    """
    Visual evolution of a morphological snake.
    
    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    background : array-like, optional
        If given, background will be shown behind the contours instead of
        msnake.data.
    """
    
    if levelset is not None:
        msnake.levelset = levelset
    
    # Prepare the visual environment.
    fig = ppl.gcf()
    fig.clf()

    fig.canvas.set_window_title(msnake.name)

    ax1 = fig.add_subplot(1,2,1)
    if background is None:
        ax1.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax1.imshow(background, cmap=ppl.cm.gray)
    ax1.contour(msnake.levelset, [0.5], colors='r')
    
    ax2 = fig.add_subplot(1,2,2)
    ax_u = ax2.imshow(msnake.levelset)

    if show_axe3 == True:
        ax3 = fig.add_subplot(1,3,2)
        ax_b = ax3.imshow(msnake.data)

    ppl.pause(0.001)
    
    # Iterate.
    for i in range(num_iters):
        # Evolve.
        msnake.step()
        
        # Update figure.
        del ax1.collections[0]
        ax1.contour(msnake.levelset, [0.5], colors='r')
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()

        if save_img == True:
            if i % save_every_iter == 0:
                fig = ppl.gcf()
                fig.savefig('runs/' + msnake.name + '_' + str(i) + '.png')

        #ppl.pause(0.001)
    
    fig = ppl.gcf()
    fig.savefig('runs/' + msnake.name + '_final' + '.png')

    # Return the last levelset.
    return msnake.levelset


def evolve_visual_gif(msnake, levelset=None, num_iters=20, background=None):
    if levelset is not None:
        msnake.levelset = levelset
    
    # Prepare the visual environment.
    fig = ppl.gcf()
    fig.clf()

    ax1 = fig.plot()
    if background is None:
        ax1.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax1.imshow(background, cmap=ppl.cm.gray)
    ax1.contour(msnake.levelset, [0.5], colors='r')

