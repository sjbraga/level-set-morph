import numpy as np
import scipy.signal as signal


def gauss_kern():
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h1 = 8
    h2 = 8
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 10.0
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) )
    return g / g.sum()

def border(img):
    """Funcao de parada das bordas"""
    img = img[::-1]
    g = gauss_kern()
    img_smooth = signal.convolve(img, g, mode='same')
    Iy, Ix = np.gradient(img_smooth)
    absGradI=np.sqrt(Ix**2+Iy**2)
    return 1 / (1+absGradI**2)


class Levelset(object):
    """
    Traditional levelset implementation
    """
    def __init__(self, borderFunc, step=1, num_reinit=10, max_iter=150):
        """
        Create traditional levelset solver

        Parameters
        ----------

        :border: border function
        :step: step size
        :num_reinit: number of iterations to reset levelset function
        :max_iter: max number of iterations for contour evolution
        """
        self._u = None #contorno C => levelset
        self.borders = borderFunc
        self.step_size = step
        self.num_reinit = num_reinit
        self.max_iter = max_iter

    def set_levelset(self, u):
        #self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0

    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")

    def step(self):
        phi = self._u #contorno
        g = self.borders #funcao de borda
        gy, gx = np.gradient(g)
        dt = self.step_size

        if phi is None:
            raise ValueError("levelset not set")

        # gradient of phi
        gradPhiY, gradPhiX = np.gradient(phi)    
        # magnitude of gradient of phi
        absGradPhi=np.sqrt(gradPhiX**2+gradPhiY**2)
        # normalized gradient of phi - eliminating singularities
        normGradPhiX=gradPhiX/(absGradPhi+(absGradPhi==0))
        normGradPhiY=gradPhiY/(absGradPhi+(absGradPhi==0))
        
        divYnormGradPhiX, divXnormGradPhiX=np.gradient(normGradPhiX)
        divYnormGradPhiY, divXnormGradPhiY=np.gradient(normGradPhiY)

        # curvature is the divergence of normalized gradient of phi
        K = divXnormGradPhiX + divYnormGradPhiY
        tmp1 = g * K * absGradPhi
        tmp2 = g * absGradPhi
        tmp3 = gx * gradPhiX + gy*gradPhiY
        dPhiBydT =tmp1 + tmp2 + tmp3    

        #curve evolution
        phi = phi + (dt * dPhiBydT)

        self._u = phi

        