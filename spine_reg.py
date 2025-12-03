import numpy as np
import matplotlib.pyplot as plt
import torch
# import sys
# sys.path.append('/home/abenneck/Desktop/emlddmm/')
# import emlddmm

def draw(I,xI=None,fig=None,function=np.sum,**kwargs):
    """
    Generate a plot of 'I' along the 3 cardinal planes - Coronal, sagittal, and transverse.

    Parameters:
    -----------
    I : array
        A 3D image volume
    xI : list of array
        A list of the coordinates along each dimension of I
    fig : matplotlib.figure
        A figure on which new plots will be generated
    function : Python function
        Default - np.sum(), will produce a Maximum Intensity Projection; The function used to generate the 3 views of I.

    Returns:
    --------
    axs : np.array of matplotlib.axes
        Each element of axs contains 1 of the 3 cardinal views of I        
    """
    if xI is None:
        nI = I.shape[-3:]
        xI = [np.arange(n) - (n-1)/2 for n in nI]
    if fig is None:
        fig = plt.figure()
    
    # Initialize figure
    dI = [x[1] - x[0] for x in xI]    
    I = np.asarray(I)
    fig.clf()
    axs = []

    # Generate plot along coronal plane
    ax = fig.add_subplot(3,1,1)
    ax.imshow(function(I,-3).squeeze(),extent=(xI[-1][0]-dI[-1]/2, xI[-1][-1]+dI[-1]/2, xI[-2][-1]+dI[-2]/2, xI[-2][0]-dI[-2]/2),**kwargs)
    axs.append(ax)
    
    # Generate plot along ?transverse? plane
    ax = fig.add_subplot(3,1,2)
    ax.imshow(function(I,-2).squeeze(),extent=(xI[-1][0]-dI[-1]/2, xI[-1][-1]+dI[-1]/2, xI[-3][-1]+dI[-3]/2, xI[-3][0]-dI[-3]/2),**kwargs)
    axs.append(ax)
    
    # Generate plot along ?sagittal? plane
    ax = fig.add_subplot(3,1,3)
    ax.imshow(function(I,-1).squeeze(),extent=(xI[-2][0]-dI[-2]/2, xI[-2][-1]+dI[-2]/2, xI[-3][-1]+dI[-3]/2, xI[-3][0]-dI[-3]/2),**kwargs)
    axs.append(ax)

    # 12/03/25: Changed return from np.array(axs) to fig, axs. This was done so that a user could save the figure as a file if desired
    return fig, axs


def getslice(I,ax):
    """
    Return a 2D slice of the 3D image volume 'I'

    Parameters:
    -----------
    I : array
        A 3D image volume
    ax : int
        Options : -1, -2, -3; The axis of I from which a slice should be extracted

    Returns:
    --------
    A subset of I along the desired axis
    """
    if ax == -1:
        return I[...,I.shape[-1]//2]
    elif ax == -2:
        return I[...,I.shape[-2]//2,:]
    elif ax == -3:
        return I[...,I.shape[-3]//2,:,:]    


def interp(xI,I,Xs,**kwargs):
    """
    Interpolate ...

    Parameters:
    -----------
    xI : list of array
        A list of the coordinates along each dimension of I
    I : array
        A 3D image volume
    Xs : ...
        ...

    Returns:
    --------
    output : torch.Tensor
        ...
    """
    Xs = Xs - torch.stack([x[0] for x in xI])
    Xs = Xs / torch.stack([x[-1] - x[0] for x in xI])
    Xs = Xs *2 - 1

    return torch.nn.functional.grid_sample(I[None],Xs[None].flip(-1),align_corners=True,**kwargs)[0]


def interp1d(xI,squish,Xs,**kwargs):
    """
    Hack for 1D interpolation

    Parameters:
    -----------
    xI : list of array
        A list of the coordinates along each dimension of I
    squish : ...
        ...
    Xs : ...
        ...

    Returns:
    --------
    output : ...
        ...
    """
    # set up a hack for 1d interplation    
    # grid sample supports 2D
    # so we will make it 2d
    # use the slice coordinate as the first coordinate, and zeros as the second fake coordinate    
    samples = torch.stack([Xs[...,0].squeeze(),torch.zeros_like(Xs[...,0].squeeze())],-1)
    #print(squish.shape)
    # for the input, we keep the channel dimension, keep the first coordinate, and add a fake second coordinate
    squishin = squish[:,:,None]
    out = interp([xI[0],torch.tensor([-0.5,0.5],**dd)],squishin,samples.reshape(-1,1,2),**kwargs)
    
    return out.reshape((squish.shape[0],)+samples.shape[:-1])    


# we need integration of v, note there is NO batch dimension
def phii_from_v(xv,v):
    """
    ...

    Parameters:
    -----------
    xv : ...
        ...
    v : ...
        ...

    Returns:
    --------
    phii : torch.Tensor
        ...
    """
    dt = 1.0/v.shape[0]
    XV = torch.stack(torch.meshgrid(xv,indexing='ij'),-1)
    phii = XV.clone()#.repeat((v.shape[0],1,1,1))
    for t in range(v.shape[0]):
        Xs = XV - v[t]*dt # Xs should have a batch dimension        
        phii = interp(xv,(phii-XV).permute(3,0,1,2),Xs,padding_mode='border').permute(1,2,3,0) + Xs
    return phii


def toblocks(Jd,blocksize):
    """
    ...

    Parameters:
    -----------
    Jd : ...
        ...
    blocksize : int
        ...

    Returns:
    --------
    Jdpp : ...
        ...
    """
    nblocks = torch.ceil(torch.tensor(Jd.shape[1:],**dd)/blocksize ).to(int)
    topad = nblocks*blocksize - torch.tensor(Jd.shape[1:],device=device)
    topadlist = [topad[-1],0,topad[-2],0,topad[-3],0] # this pads the left
    Jdp = torch.nn.functional.pad(Jd,topadlist,mode='reflect')
    Jdpv = Jdp.reshape(Jdp.shape[0],nblocks[0],blocksize,nblocks[1],blocksize,nblocks[2],blocksize)
    Jdpp = Jdpv.permute(1,3,5,0,2,4,6)#.reshape(-1,Jdpv.shape[0],blocksize,blocksize,blocksize)
    return Jdpp


def fromblocks(fphiIpp,Jdsize):
    """
    ...

    Parameters:
    -----------
    fphiIpp : ...
        ...
    Jdsize : ...
        ...

    Returns:
    --------
    output : ...
        ...
    """
    # undo the permutation
    blocksize = fphiIpp.shape[-1]
    nblocks = torch.ceil(torch.tensor(Jdsize[-3:],**dd)/blocksize ).to(int)
    #print(nblocks)
    topad = nblocks*blocksize - torch.tensor(Jdsize[-3:],device=device)
    fphiIpv = fphiIpp.permute(3,0,4,1,5,2,6)
    # NOTE THIS SIZE 1 IS HARD CODED
    fphiIp = fphiIpv.reshape(1,nblocks[0]*blocksize,nblocks[1]*blocksize,nblocks[2]*blocksize)
    #return fphiIp[:,:Jdsize[-3],:Jdsize[-2],:Jdsize[-1]]
    return fphiIp[:,topad[0]:,topad[1]:,topad[2]:]


def measure_matching_dot(qIU,wIU,phiiQJU,wphiiQJU,SigmaQIU):
    """
    ...

    Parameters:
    -----------
    qIU : ...
        ...
    wIU : ...
        ...
    phiiQJU : ...
        ...
    wphiiQJU : ...
        ...
    SigmaQIU : ...
        ...

    Returns:
    --------
    output : ...
        ...
    """
    K = torch.exp( - torch.sum( (qIU[:,None] - phiiQJU[None,:])**2/2/SigmaQIU , -1) )*wIU[:,None]*wphiiQJU[None,:]
    return torch.sum(K)