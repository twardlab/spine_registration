import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import argparse
from spine_reg import * # TODO: Add path argument in final version
import gc

def main():
    """
    Command Line Arguments:
    =======================
    fname_I : str
        The path to the interpolated atlas .npz file to be used for registration
    fname_L : str
        The path to the interpolated atlas labels .npz file to be used for registration
    fname_J : str
        The path to the spine reflection .npz file to be registered
    fname_pointsJ : str
        The path to an .swc file used for more precise registration of J
    outdir : str
        The directory where all output files should be saved
    e_path : str
        The location of the custom Python library \'emlddmm\', which can be cloned from GitHub at https://github.com/twardlab/emlddmm
    device : str
        Default - cpu; The device where registration computations should occur
    dtype : torch.dtype
        Default - torch.float32; The data type used for the voxel values in image registration
    niter : int
        Default - 5000; The number of iterations to use for image registration
    down : list of int
        Default - [8,8,8]; The factor used to downsample along each axis of I, respectively
    blocksize : int
        Default - 50; TODO: ...
    verbose : bool
        Default - False; If present, print out several progress messages throughout the registration process
    saveAllFigs : bool
        Default - False; If present, save every intermediate figure generated before, during, and after registration
    saveIntermediateFigs : bool
        Default - false; If present, save intermediate figures at each iteration of the registration process
    saveFig0 : bool
        Default - False; If present, save a MIP of the interpolated atlas provided
    saveFig1 : bool
        Default - False; If present, save a MIP of the interpolated atlas labels provided
    saveFig2 : bool
        Default - False; If present, save a scatter plot of all the points labeled in the interpolated atlas porvided
    saveFig3 : bool
        Default - False; If present, save a standard view of the target data along all 3 cardinal axes
    saveFig4 : bool
        Default - False; If present, save a MIP of the target data along all 3 cardinal axes
    saveFig5 : bool
        Default - False; If present, save a standard view of the target data along all 3 cardinal axes
    saveFig6 : bool
        Default - False; If present, save a standard view of the target data along all 3 cardinal axes
    saveFig7 : bool
        Default - False; If present, save a save a scatter plot of all points along the AP axis
    saveFig8 : bool
        Default - False; If present, save a standard view of the qJU object along the AP axis
    saveFig9 : bool
        Default - False; If present, save a save a standard view of the qJU object along the AP axis with points superimposed
    saveFig10 : bool
        Default - False; If present, save a save a standard view of the downsampled target data along all 3 cardinal axes
    saveFig11 : bool
        Default - False; If present, save a save a plot of the xv*xId object
    saveFig12 : bool
        Default - False; If present, save a save a plot of the B object
    saveFig13 : bool
        Default - False; If present, save a color gradient
    saveFig14 : bool
        Default - False; If present, save a color gradient
    saveFig15 : bool
        Default - False; If present, save a color gradient
    saveFig16 : bool
        Default - False; If present, save a color gradient

    Raises:
    =======
    Exception
        If any of the 8 input files do not have the correct file extension
    Exception
        If a negative Jacobian arises during the registration loop
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('fname_I', type=str, help = 'The path to the interpolated atlas .npz file')
    parser.add_argument('fname_L', type=str, help = 'The path to the interpolated atlas labels .npz file')
    parser.add_argument('fname_J', type=str, help = 'The path to the spine reflection .npz file')
    parser.add_argument('fname_pointsJ', type=str, help = 'The input .swc file used for more precise registration')
    parser.add_argument('outdir', type=str, help = 'The directory where all output files should be saved')
    parser.add_argument('e_path', type=str, help= 'The location of the custom Python library \'emlddmm\', which can be cloned from GitHub at https://github.com/twardlab/emlddmm')
    
    parser.add_argument('-device', type=str, default = 'cpu', help = 'Default - cpu; The device where registration computations should occur')
    parser.add_argument('-dtype', type = torch.dtype, default = torch.float32, help = 'Default - torch.float32; The data type used for image registration')
    parser.add_argument('-niter', type = int, default = 5000, help = 'Default - 5000; The number of iterations to use for image registration')
    parser.add_argument('-down', type = int, nargs = 3, default = [8,8,8], help = 'The factor used to downsample along each axis of I, respectively')
    parser.add_argument('-bs', '--blocksize', type=int, default = 50, help = 'TODO: ...')

    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Default - False; If present, print out several progress messages throughout the registration process')
    
    parser.add_argument('-saveAllFigs', action = 'store_true', help = 'Default - False; If present, save every intermediate figure generated before, during, and after registration')
    parser.add_argument('-saveIntermediateFigs', action = 'store_true', help = 'Default - false; If present, save intermediate figures at each iteration of the registration process')
    parser.add_argument('-saveFig0', action = 'store_true', help = 'Default - False; If present, save a MIP of the interpolated atlas provided')
    parser.add_argument('-saveFig1', action = 'store_true', help = 'Default - False; If present, save a MIP of the interpolated atlas labels provided')
    parser.add_argument('-saveFig2', action = 'store_true', help = 'Default - False; If present, save a scatter plot of all the points labeled in the interpolated atlas porvided')
    parser.add_argument('-saveFig3',  action = 'store_true', help = 'Default - False; If present, save a standard view of the target data along all 3 cardinal axes')
    parser.add_argument('-saveFig4',  action = 'store_true', help = 'Default - False; If present, save a MIP of the target data along all 3 cardinal axes')
    parser.add_argument('-saveFig5',  action = 'store_true', help = 'Default - False; If present, save a standard view of the target data along all 3 cardinal axes')
    parser.add_argument('-saveFig6',  action = 'store_true', help = 'Default - False; If present, save a standard view of the target data along all 3 cardinal axes')
    parser.add_argument('-saveFig7',  action = 'store_true', help = 'Default - False; If present, save a scatter plot of all points along the AP axis')
    parser.add_argument('-saveFig8',  action = 'store_true', help = 'Default - False; If present, save a standard view of the qJU object along the AP axis')
    parser.add_argument('-saveFig9',  action = 'store_true', help = 'Default - False; If present, save a standard view of the qJU object along the AP axis with points superimposed')
    parser.add_argument('-saveFig10', action = 'store_true', help = 'Default - False; If present, save a standard view of the downsampled target data along all 3 cardinal axes')
    parser.add_argument('-saveFig11', action = 'store_true', help = 'Default - False; If present, save a plot of the xv*xId object')
    parser.add_argument('-saveFig12', action = 'store_true', help = 'Default - False; If present, save a plot of the B object')
    parser.add_argument('-saveFig13', action = 'store_true', help = 'Default - False; If present, save a color gradient')
    parser.add_argument('-saveFig14', action = 'store_true', help = 'Default - False; If present, save a color gradient')
    parser.add_argument('-saveFig15', action = 'store_true', help = 'Default - False; If present, save a color gradient')
    parser.add_argument('-saveFig16', action = 'store_true', help = 'Default - False; If present, save a color gradient')
    
    args = parser.parse_args()

    fname_I = args.fname_I
    fname_L = args.fname_L
    fname_J = args.fname_J
    fname_pointsJ = args.fname_pointsJ
    outdir = args.outdir
    e_path = args.e_path
    device = args.device
    dtype = args.dtype
    niter = args.niter
    down = args.down
    blocksize = args.blocksize
    verbose = args.verbose
    
    saveAllFigs = args.saveAllFigs
    saveIntermediateFigs = args.saveIntermediateFigs
    saveFig0  = args.saveFig0
    saveFig1  = args.saveFig1
    saveFig2  = args.saveFig2
    saveFig3  = args.saveFig3
    saveFig4  = args.saveFig4
    saveFig5  = args.saveFig5
    saveFig6  = args.saveFig6
    saveFig7  = args.saveFig7
    saveFig8  = args.saveFig8
    saveFig9  = args.saveFig9
    saveFig10 = args.saveFig10
    saveFig11 = args.saveFig11
    saveFig12 = args.saveFig12
    saveFig13 = args.saveFig13
    saveFig14 = args.saveFig14
    saveFig15 = args.saveFig15
    saveFig16 = args.saveFig16

    sys.path.append(e_path)
    import emlddmm

    # Create outdir if it doesn't already exist
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)

    # =================================================
    # ===== Perform validity checks on parameters =====
    # =================================================
    if '.npz' not in fname_I:
        raise Exception(f'{fname_I} should be a .npz file')
    if '.npz' not in fname_L:
        raise Exception(f'{fname_L} should be a .npz file')
    if '.npz' not in fname_J:
        raise Exception(f'{fname_J} should be a .npz file')
    if '.swc' not in fname_pointsJ:
        raise Exception(f'{fname_pointsJ} should be a .swc file')
        
    # ===================================
    # ===== Load interpolated atlas =====
    # ===================================
    data_I = np.load(fname_I,allow_pickle=True)
    I = data_I['I']
    xI = data_I['xI']
    I = I / I.max() # Normalize atlas values

    lowert = 0.15 # TODO: Decide if this should be a passable argument
    M = I<lowert # find a mask for the background
    I[M] = 1.0
    I = (1 - I)
    I = I - lowert
    I[I<0] = 0
    
    I = I/I.max()

    if saveAllFigs or saveFig0:
        fig, axs = draw(I,xI,function=getslice)
        fig.savefig(os.path.join(outdir, 'fig0_interp_atlas.png'))

    if verbose:
        print('Successfully loaded atlas file . . .')

    # ==========================================
    # ===== Load interpolated atlas labels =====
    # ==========================================
    data_L = np.load(fname_L,allow_pickle=True)
    L = data_L['I']%256
    xL = data_L['xI']

    if saveAllFigs or saveFig1:
        fig, axs = draw((L%256)==16,xI,)
        fig.savefig(os.path.join(outdir, 'fig1_interp_atlas_labels.png'))

    qIU = np.stack(np.meshgrid(*xI,indexing='ij'),-1)[(L[0]%256)==16]
    nqU = 1000 # TODO: Decide if this should be a passable argument
    qIU = qIU[np.random.permutation(qIU.shape[0])[:nqU]]

    if saveAllFigs or saveFig2:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*qIU.T)
        fig.savefig(os.path.join(outdir, 'fig2_scattered_labels.png'))

    SigmaQIU = []
    for j in range(3):
        d2 = (qIU[:,None,j] - qIU[None,:,j])**2
        d2i = []
        for i in range(d2.shape[0]):
            d2i.append( np.min( d2[i][d2[i]>0] ) )
        SigmaQIU.append( np.mean(d2i) )
    SigmaQIU = np.array(SigmaQIU)

    if verbose:
        print('Successfully loaded atlas labels file . . .')

    # =====================================================================
    # ===== Load the target image data to be registered + axis values =====
    # =====================================================================
    J = np.load(fname_J.replace('.npz','_I.npy'))
    xJ = [
        np.load(fname_J.replace('.npz','_xI0.npy')),
        np.load(fname_J.replace('.npz','_xI1.npy')),
        np.load(fname_J.replace('.npz','_xI2.npy')),
    ]

    # If necessary, add a 4th dimension to J
    if J.ndim == 3:
        J = J[None]

    # Readjust coordinate axes to be centered about 0
    xJ = [x - np.mean(x) for x in xJ]
    
    # Normalize J
    Jsave = J.copy()
    low = 0 # TODO: Decide if this should be a passable argument
    high = 1500 # TODO: Decide if this should be a passable argument
    J = Jsave.copy().clip(low,high)
    WJ = 1.0 - (J == high)
    J = J - low
    J = J / (high-low)
    J = J*WJ
    WJ = WJ*0+1
    sl = (slice(None),slice(24,-50,None),slice(50,-110,None),slice(200,-310,None))

    if saveAllFigs or saveFig3:
        fig, axs = draw((J*WJ)[sl],[x[s] for s,x in zip(sl[1:],xJ)],function=getslice)
        fig.savefig(os.path.join(outdir, 'fig3_target_slice0.png'))

    if saveAllFigs or saveFig4:
        fig, axs = draw((J*WJ)[sl],[x[s] for s,x in zip(sl[1:],xJ)])
        fig.savefig(os.path.join(outdir, 'fig4_target_MIP.png'))

    if saveAllFigs or saveFig5:
        fig, axs = draw(J*WJ,xJ,function=getslice)
        fig.savefig(os.path.join(outdir, 'fig5_target_slice1.png'))

    if saveAllFigs or saveFig6:
        fig, axs = draw(WJ,xJ,function=getslice)
        fig.savefig(os.path.join(outdir, 'fig6_target_slice2.png'))

    if verbose:
        print('Successfully loaded target image data and axes files . . .')

    # Initialzie qJU, which is TODO
    qJU = []
    with open(fname_pointsJ) as f:
        for line in f:
            if line.strip()[0] == '#': 
                continue
            items = line.split()[2:5] 
            qi = np.array([float(s) for s in items])
            qJU.append(qi)
    qJU = np.stack(qJU)
    qJU = qJU[np.random.randint(low=0,high=qJU.shape[0],size=nqU)]
    qJU = qJU + np.random.randn(*qJU.shape)*2

    if saveAllFigs or saveFig7:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*qJU.T)
        ax.set_aspect('equal')
        fig.savefig(os.path.join(outdir, 'fig7_qJU_scatter.png'))

    if saveAllFigs or saveFig8:
        fig,ax = plt.subplots()
        ax.imshow(J.sum((0,1)),extent=(xJ[2][0],xJ[2][-1],xJ[1][-1],xJ[1][0]))
        ax.plot(qJU[:,2],qJU[:,1])
        fig.savefig(os.path.join(outdir, 'fig8_qJU0.png'))
        
    if saveAllFigs or saveFig9:
        fig,ax = plt.subplots()
        ax.imshow(J[0,:,J.shape[2]//2],extent=(xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0]))
        ax.plot(qJU[:,2],qJU[:,0])
        fig.savefig(os.path.join(outdir, 'fig9_qJU1.png'))

    if verbose:
        print('Successfully loaded target image points file . . .')

    # =========================
    # ===== Preprocessing =====
    # =========================

    # Downsample the atlas image, I, and target iamge, J
    xId,Id = emlddmm.downsample_image_domain(xI,I,down)
    xJd,Jd,WJd = emlddmm.downsample_image_domain([x[s] for x,s in zip(xJ,sl[1:])],J[sl],down,W=WJ[0][sl[1:]])    

    if saveAllFigs or saveFig10:
        fig, axs = draw(Jd*WJd,xJd,function=getslice)
        fig.savefig(os.path.join(outdir, 'fig10_target_down.png'))

    # Convert input data from Numpy to torch format
    dd = {'device':device,'dtype':dtype}
    Id = torch.tensor(Id,**dd)
    Jd = torch.tensor(Jd,**dd)
    xId = [torch.tensor(x,**dd) for x in xId]
    xJd = [torch.tensor(x,**dd) for x in xJd]
    WJd = torch.tensor(WJd,**dd)
    
    XId = torch.stack(torch.meshgrid(xId,indexing='ij'),-1)
    XJd = torch.stack(torch.meshgrid(xJd,indexing='ij'),-1)
    
    nId = torch.tensor(Id.shape[-3:],device=device) # int
    
    qIU = torch.tensor(qIU,**dd)
    qJU = torch.tensor(qJU,**dd)
    SigmaQIU = torch.tensor(SigmaQIU,**dd)

    # Initialize parameters for deformation portion of registration
    # TODO: Decide if any of these should be passable arguments
    nt = 5
    dt = 1.0/nt
    a = 200.0
    p = 2.0
    expand = [1.02,1.2,1.2]
    dv = a*0.5
    vminmax = [(torch.min(x),torch.max(x)) for x in xId]
    vr = torch.stack( [(x[1] - x[0]) for x in vminmax] )
    vc = torch.stack( [(x[1]+x[0])/2 for x in vminmax] )
    vminmax = vc + (vr*torch.tensor(expand,**dd))*torch.tensor([-1,1],**dd)[...,None]*0.5
    xv = [torch.arange(vm[0],vm[1],dv) for vm in vminmax.T]
    XV = torch.stack(torch.meshgrid(xv,indexing='ij'),-1)
    nv = [len(x) for x in xv]
    fv = [torch.arange(n)/n/dv for n in XV.shape[:3]]
    FV = torch.stack(torch.meshgrid(fv,indexing='ij'),-1)
    LL = (1.0 + 2.0*a**2 * torch.sum(  (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2,   -1))**(2.0*p)
    K = 1.0/LL    
    v = torch.zeros(nt,*nv,3,requires_grad=True)
    theta = torch.zeros(nv[0],requires_grad=True,**dd)
    T = torch.zeros((nv[0],2),requires_grad=True,**dd)
    squish = torch.ones(nv[0],**dd)*(-0.4) # exponentiate it (-0.1 was good), -1 makes it very short and fat in the coronal plane
    squish.requires_grad = True
    B = torch.exp( -(xv[0][None,:] - xId[0][:,None])**2/2/5000**2)
    B = torch.exp( -(xv[0][None,:] - xId[0][:,None])**2/2/2000**2)
    B = B / torch.sum(B,1,keepdims=True)
    squishb = B@squish

    if saveAllFigs or saveFig11:
        fig,ax = plt.subplots()
        ax.plot(xv[0],squish.detach())
        ax.plot(xId[0],squishb.detach())
        fig.savefig(os.path.join(outdir, 'fig11_xv_xId.png'))
        

    if saveAllFigs or saveFig12:
        fig,ax = plt.subplots()
        ax.imshow(B)        
        fig.savefig(os.path.join(outdir, 'fig12_B.png'))

    # Initalize the final permutation
    P = torch.eye(4)[[1,2,0,3]]
    A = P
    A.requires_grad = True
    stretch = torch.tensor([0.2,-0.2,-0.75],requires_grad = True) # the third number, in the coronal plane, makes it grow left right if it is positive

    # Define a metric for affine, since coordinate system is centered at 0, my off diagonal terms will be 0 in the standard push forward approach
    # for two basis matrices, we need to evaluate:
    #     gij = int (Eix)^T   Ejx dx = int trace [  x^T Ei^T Ej x ] dx =  int trace [xx^T Ei^TEj]dx = trace [ int xx^T dx Ei^TEj ]
    XX = torch.sum(XId[...,None]*XId[...,None,:],(0,1,2))
    O = torch.sum(torch.ones_like(XId[...,0]))
    XXO = torch.diag(torch.concatenate((torch.diag(XX),O[None])))
    g = torch.zeros((12,12),**dd)
    count = 0
    for i in range(3):
        for j in range(4):
            Eij = (torch.arange(4,**dd)==i)[...,None]*(torch.arange(4,**dd)==j)[...,None,:]*1.0
            count_ = 0
            for i_ in range(3):
                for j_ in range(4):
                    Eij_ = (torch.arange(4,**dd)==i_)[...,None]*(torch.arange(4,**dd)==j_)[...,None,:]*1.0
                    g[count,count_] = torch.trace( XXO@(Eij.T@Eij_) )
    
                    count_ += 1 
            count += 1
    gi = torch.linalg.inv(g)
    dJd = [(x[1] - x[0]).item() for x in xJd]

    # ========================
    # ===== Registration ===== 
    # ========================
    figE,axE = plt.subplots(3,4)
    axE = axE.ravel()
    figI = plt.figure()
    figErr = plt.figure()
    figJ = plt.figure()
    draw(Jd.detach(),xJd,function=getslice,fig=figJ,aspect='auto')
    figJ.canvas.draw()
    
    figQ = plt.figure()
    axQ = figQ.add_subplot(projection='3d')
    axQ.cla()
    axQ.scatter(*qIU.T,label='qIU',alpha=0.1)
    axQ.legend()
    
    # Initialize output figures data structures for storing output data
    nrow = 6
    ncol = 4
    figS,axS = plt.subplots(nrow,ncol)
    figS.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
    axS = axS.ravel()        
    Esave = []
    Tsave = [] # the max
    squishsave = []
    thetasave = []
    vsave = []
    ALsave = []
    ATsave = []
    update_v = False
    vstart = -1
    
    # Initalize various step sizes
    # TODO: Decide if any of these should be passable arguments
    epv = 5e4
    eptheta = 5e-4
    epSquish = 2e-4
    epT = 2e2
    measureMatchingSigma = SigmaQIU*5**2
    sigmaM = (1e5)**0.5
    sigmaR = 1e5
    sigmaR = 2e5
    sigmaQU = 1e-1*50    
    wIU = torch.ones_like(qIU[...,0])#/qIU.shape[0]*qJU.shape[0]
    EQU0 = measure_matching_dot(qIU,wIU,qIU,wIU,measureMatchingSigma) # only compute once

    # Begin the registration loop over 'niter' iterations
    for it in range(0,niter):

        if it > vstart:
            update_v = True
        if it < 500:
            blocksize = 0
        elif it < 1000:
            blocksize = 50
        else:
            blocksize = 32

        # Clean up memory using garbage collection
        Xs = None
        Xs0 = None
        phii = None
        out = None
        gc.collect()
        
        # Apply blurs to barious data structures 
        thetab = B@theta
        Tb = B@T
        squishb = B@squish
    
        # First, compute the inverse affine (Ai)
        Ai = torch.linalg.inv(A)[:3]
        Ai = torch.diag((-stretch).exp())@Ai
        Xs = (Ai[:3,:3]@XJd[...,None])[...,0] + Ai[:3,-1]

        # Second, compute the squish
        tosample = torch.concatenate((Tb.T,squishb[None],thetab[None]))
        out = interp1d(xId,tosample,Xs,dd)
        Ts = out[0:2].permute(1,2,3,0)
        squishs = out[2]
        thetas = out[3]
        eye = torch.diag(torch.ones(3,**dd),)[None,None,None].repeat(Ts.shape[0],Ts.shape[1],Ts.shape[2],1,1)
        zo = torch.tensor([0.0,0.0,0.0,1.0],**dd)[None,None,None,None].repeat(Ts.shape[0],Ts.shape[1],Ts.shape[2],1,1)
        z = torch.zeros_like(Ts[...,0,None],)
        Tcat =  torch.concatenate((z,Ts),-1)    
        squishmat = torch.diag_embed(  torch.stack([torch.ones_like(squishs),(-squishs).exp(),(squishs).exp()] ,-1 ) )
        rotmat = torch.stack([
            torch.stack([torch.ones_like(thetas),torch.zeros_like(thetas),torch.zeros_like(thetas)],-1),
            torch.stack([torch.zeros_like(thetas),torch.cos(thetas),torch.sin(thetas)],-1),
            torch.stack([torch.zeros_like(thetas),-torch.sin(thetas),torch.cos(thetas)],-1)
            ],-2)
        
        Xs = Xs - Tcat    
        Xs = (rotmat@squishmat@Xs[...,None])[...,0]

        # Third, compute the diffeo
        phii = phii_from_v(xv,v)
        Xs = interp(xv,(phii-XV).permute(-1,0,1,2),Xs).permute(1,2,3,0) + Xs
        phiI = interp(xId,Id,Xs)
        phiiQJU = interp(xJd,Xs.permute(-1,0,1,2),qJU[None,None])[:,0,0].T
        dphii = torch.ones_like(phiiQJU[...,0])
        if torch.any(dphii)<=0:
            raise Exception('Negative jacobian')
        
        # contrast
        if blocksize > 0:
            Jdpp = toblocks(Jd,blocksize) # this only needs to be done once
            WJdpp = toblocks(WJd[None],blocksize) # only needs to be done once
            phiIpp = toblocks(phiI,blocksize) # only needs to be done once
        with torch.no_grad():
            if blocksize == 0:
                muI = torch.sum(phiI*WJd)/torch.sum(WJd)
                muJ = torch.sum(Jd*WJd)/torch.sum(WJd)
                varI = torch.sum((phiI-muI)**2*WJd)/torch.sum(WJd)
                covIJ = torch.sum((phiI-muI)*(Jd-muJ)*WJd)/torch.sum(WJd)
            else:
                muI = torch.sum(phiIpp*WJdpp,(-3,-2,-1),keepdims=True)/torch.sum(WJdpp,(-3,-2,-1),keepdims=True)
                muJ = torch.sum(Jdpp*WJdpp,(-3,-2,-1),keepdims=True)/torch.sum(WJdpp,(-3,-2,-1),keepdims=True)
                varI = torch.sum((phiIpp-muI)**2*WJdpp,(-3,-2,-1),keepdims=True)/torch.sum(WJdpp,(-3,-2,-1),keepdims=True)
                covIJ = torch.sum((phiIpp-muI)*(Jdpp-muJ)*WJdpp,(-3,-2,-1),keepdims=True)/torch.sum(WJdpp,(-3,-2,-1),keepdims=True)
        if blocksize > 0:
            fphiIpp = (phiIpp-muI)/(varI + 1e-6)*covIJ + muJ
            fphiI = fromblocks(fphiIpp,Jd.shape)
        else:
            fphiI = (phiI-muI)/(varI + 1e-6)*covIJ + muJ
        err = (fphiI-Jd)
        EM = (err**2*WJd).sum()/2.0*torch.prod(torch.stack([x[1] - x[0 ] for x in xJd])) /sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(-4,-3,-2)))**2,(0,-1))*K)/2.0/nt/v[0,:,:,:,0].numel()/sigmaR**2*torch.prod(torch.stack([x[1] - x[0 ] for x in xJd])) 
    
        # Point matching
        EQU = (EQU0 + -2*measure_matching_dot(qIU,wIU,phiiQJU,dphii,measureMatchingSigma) + measure_matching_dot(phiiQJU,dphii,phiiQJU,dphii,measureMatchingSigma))/2.0/sigmaQU**2
        E = EM + ER + EQU
        E.backward()
        Esave.append([E.item(),EM.item(),ER.item(),EQU.item()])
    
        # Draw the error figures
        draw(err.detach()*WJd,xJd,function=getslice,fig=figErr,aspect='auto')
        figErr.canvas.draw()
        draw(fphiI.detach(),xJd,function=getslice,fig=figI,aspect='auto')
        figI.canvas.draw()
        
        # Update parameters + set gradients to 0
        theta.data = theta.data - theta.grad*eptheta
        theta.grad.zero_()
        
        T.data = T.data - T.grad*epT
        T.grad.zero_()

        squish.data = squish.data - squish.grad*epSquish
        squish.grad.zero_()
        
        epStretch = 0
        stretch.data = stretch.data - stretch.grad*epStretch
        stretch.grad.zero_()
    
        if update_v:
            v.data = v.data - torch.fft.ifftn((torch.fft.fftn(v.grad,dim=(1,2,3))*K[...,None]),dim=(1,2,3)).real * epv
            v.grad.zero_()
        
        epA = 2e6
        Agrad = (gi@A.grad[:3].ravel()).reshape(3,4)
        A.data[:3] = A.data[:3] - epA*Agrad
        A.grad.zero_()
        ALsave.append(A[:3,:3].clone().detach().ravel().numpy())
        ATsave.append(A[:3,-1].clone().detach().ravel().numpy())
    
    
        # Generate plots showing all relevant registration variables
        axE[0].cla()
        axE[0].plot(Esave)
    
        thetasave.append(thetab.clone().detach().abs().max()*180/np.pi)
        Tsave.append(Tb.clone().detach().abs().max())
        squishsave.append(squishb.clone().detach().abs().max())
        vsave.append(v.detach().abs().max())
        axE[1].cla()
        axE[1].plot(Tsave)
        axE[1].set_title("max T")
        axE[2].cla()
        axE[2].plot(thetasave)
        axE[2].set_title("max theta")
        axE[3].cla()
        axE[3].plot(squishsave)
        axE[3].set_title("max squish")
    
        axE[4].cla()
        axE[4].plot(vsave)
        axE[4].set_title("max v")
    
        axE[5].cla()
        axE[5].plot(ALsave)
        axE[5].set_title("AL")
    
        axE[6].cla()
        axE[6].plot(ATsave)
        axE[6].set_title("AT")
    
    
        axE[7].cla()
        axE[7].plot(Tb.detach())
        axE[7].set_title("Tb")
        axE[8].cla()
        axE[8].plot(thetab.detach())
        axE[8].set_title("thetab")
        axE[9].cla()
        axE[9].plot(squishb.detach())
        axE[9].set_title("squishb")
        
        figE.canvas.draw()
    
        axQ.cla()
        axQ.scatter(*qIU.T,label='qIU',alpha=0.1)
        axQ.scatter(*phiiQJU.detach().cpu().T,label='phiiQJU',alpha=0.1)
        axQ.legend()
        figQ.canvas.draw()
        
        nshow = nrow*ncol            
        slices = np.round(np.linspace(0,Jd.shape[-1]-1,nshow+2)).astype(int)
        for i in range(nshow):
            Jshow = (Jd[0,:,:,slices[i]]*WJd[:,:,slices[i]]).numpy()
            Jshow = Jshow/np.max(Jshow)
            Ishow = fphiI[0,:,:,slices[i]].detach().numpy()/np.max(Jshow)
            axS[i].imshow(np.stack((Jshow,Ishow,Jshow),-1))
            axS[i].axis('off')
        
        figS.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
        if saveAllFigs or saveIntermediateFigs:
            figQ.savefig(os.path.join(outdir,f'out_Q_it_{it:06d}.png'))
            figErr.savefig(os.path.join(outdir,f'out_err_it_{it:06d}.png'))
            figS.savefig(os.path.join(outdir,f'out_S_it_{it:06d}.png'))
        
    # =====================================        
    # ===== Save all relevant outputs =====
    # =====================================

    # First, save the final registered versions of all relevant parameters
    np.savez(os.path.join(outdir,'saved_parameters.npz'),
             theta=theta.detach().cpu().numpy(),
             T=T.detach().cpu().numpy(),
             squish=squish.detach().cpu().numpy(),
             B=B.detach().cpu().numpy(),
             A=A.detach().cpu().numpy(),
             v=v.detach().cpu().numpy(),
             xv=np.array([x.detach().cpu().numpy() for x in xv],dtype=object),
             xId=np.array([x.detach().cpu().numpy() for x in xId],dtype=object),
             xJd=np.array([x.detach().cpu().numpy() for x in xJd],dtype=object)
            )
    
    # Second, save the forward and inverse transform
    with torch.no_grad():
         
        thetab = B@theta # blur
        Tb = B@T # blur T
        squishb = B@squish # blur Squish
    
        # Compute inverse affine
        Ai = torch.linalg.inv(A)[:3]
        Ai = torch.diag((-stretch).exp())@Ai
        Xs = (Ai[:3,:3]@XJd[...,None])[...,0] + Ai[:3,-1]

        # Since these operations do not change the z coordinate, I can interpolate them all at once
        tosample = torch.concatenate((Tb.T,squishb[None],thetab[None]))    
        out = interp1d(xId,tosample,Xs,dd)    
        Ts = out[0:2].permute(1,2,3,0)
        squishs = out[2]
        thetas = out[3]
    
        # These variables only need to be computed once
        eye = torch.diag(torch.ones(3,**dd),)[None,None,None].repeat(Ts.shape[0],Ts.shape[1],Ts.shape[2],1,1)
        zo = torch.tensor([0.0,0.0,0.0,1.0],**dd)[None,None,None,None].repeat(Ts.shape[0],Ts.shape[1],Ts.shape[2],1,1)
        z = torch.zeros_like(Ts[...,0,None],)
    
        # Remove the bottom row?
        Tcat =  torch.concatenate((z,Ts),-1)
        squishmat = torch.diag_embed(  torch.stack([torch.ones_like(squishs),(-squishs).exp(),(squishs).exp()] ,-1 ) )
        rotmat = torch.stack([
            torch.stack([torch.ones_like(thetas),torch.zeros_like(thetas),torch.zeros_like(thetas)],-1),
            torch.stack([torch.zeros_like(thetas),torch.cos(thetas),torch.sin(thetas)],-1),
            torch.stack([torch.zeros_like(thetas),-torch.sin(thetas),torch.cos(thetas)],-1)
            ],-2)
        
        Xs = Xs - Tcat    
        Xs = (rotmat@squishmat@Xs[...,None])[...,0]
    
        # Now compute the diffeo
        phii = phii_from_v(xv,v)
        Xs = interp(xv,(phii-XV).permute(-1,0,1,2),Xs).permute(1,2,3,0) + Xs

    # Save the inverse transform
    np.savez(os.path.join(outdir,'inverse_transform.npz'), phii=Xs.detach().cpu().numpy(),x=np.array([x.detach().cpu().numpy() for x in xJd],dtype=object))

    # Generate + save a high res version of the inverse
    XJd_ = torch.concatenate((XJd,torch.ones_like(XJd[...,0,None])),-1)
    Xs_ = torch.concatenate((Xs.detach(),torch.ones_like(XJd[...,0,None])),-1)
    fit = torch.linalg.solve( XJd_.reshape(-1,4).T@XJd_.reshape(-1,4), XJd_.reshape(-1,4).T @ Xs_.detach().reshape(-1,4) ).T
    FIT = (fit[:3,:3]@XJd_[...,:3,None])[...,0] + fit[:3,-1]

    fig,ax = plt.subplots(2,1)
    # hfig = display(fig,display_id=True)
    OUT = []
    for i in range(0,J.shape[1],1):
        # get the location
        thisx = [[xJ[0][i]],xJ[1],xJ[2]]
        thisX = np.stack(np.meshgrid(*thisx,indexing='ij'),-1)
        # interpolate Xs, so that now when we fill with zeros, the result will be appropriate
        thisXfit = (fit[:3,:3]@thisX[...,None])[...,0] + fit[:3,-1]    
        out = interp(xJd,(Xs-FIT).clone().detach().permute(-1,0,1,2),torch.tensor(thisX,**dd)).permute(1,2,3,0) + thisXfit.float()
        
        bad = out==0
        test = interp(xId,Id,out)
        ax[0].cla()
        ax[0].imshow((out[0] - out.min())/(out.max()-out.min()))
        ax[0].set_title(i)
        ax[1].cla()
        ax[1].imshow(test.squeeze())
        ax[1].set_title(i)
        # hfig.update(fig)
    
        # scale
        out = out - torch.tensor([xI[0][0],xI[1][0],xI[2][0]],**dd)
        out = out / ( torch.tensor([xI[0][1] - xI[0][0],xI[1][1] - xI[1][0],xI[2][1] - xI[2][0]],**dd))
        OUT.append(out)
    OUT = torch.concatenate(OUT)
    OUT = OUT.numpy()

    # Write it out
    np.save(os.path.join(outdir,'interpolated_atlas_to_spine_reflection_v04.npy'),OUT)

    if saveAllFigs or saveFig13:
        fig.savefig(os.path.join(outdir, 'fig13_interp_atlas_to_spine.png'))

    # Save the forward and inverse transform
    with torch.no_grad():
        # blur 
        thetab = B@theta
        # blur T
        Tb = B@T
        # blur Squish
        squishb = B@squish
    
        # forward phi
        phi = phii_from_v(xv,-v.flip(0))
    
        # start with XId
        Xs = XId.clone()
        
        # then apply phi
        Xs = interp(xv,(phi-XV).permute(-1,0,1,2),Xs).permute(1,2,3,0) + Xs
        
        # then we can compose the other transformations
        # since these operations do not change the z coordinate
        # I can interpolate them all at once
        tosample = torch.concatenate((Tb.T,squishb[None],thetab[None]))    
        out = interp1d(xId,tosample,Xs,dd)
        Ts = out[0:2].permute(1,2,3,0)
        squishs = out[2]
        thetas = out[3]
    
        # these guys only need to be computed once
        eye = torch.diag(torch.ones(3,**dd),)[None,None,None].repeat(Ts.shape[0],Ts.shape[1],Ts.shape[2],1,1)
        zo = torch.tensor([0.0,0.0,0.0,1.0],**dd)[None,None,None,None].repeat(Ts.shape[0],Ts.shape[1],Ts.shape[2],1,1)
        z = torch.zeros_like(Ts[...,0,None],)
        Tcat =  torch.concatenate((z,Ts),-1) # I'm leaving the sign here the same
        squishmat = torch.diag_embed(  torch.stack([torch.ones_like(squishs),(squishs).exp(),(squishs).exp()] ,-1 ) ) # note I deleted the minus sign as compared to above
        rotmat = torch.stack([
            torch.stack([torch.ones_like(thetas),torch.zeros_like(thetas),torch.zeros_like(thetas)],-1),
            torch.stack([torch.zeros_like(thetas),torch.cos(thetas),-torch.sin(thetas)],-1), # note I moved the minus sign as compared to above
            torch.stack([torch.zeros_like(thetas),torch.sin(thetas),torch.cos(thetas)],-1)
            ],-2)
        # before the order was translate, squish, rotate
        # so now we do, rotate, squish translate
        Xs = (squishmat@rotmat@Xs[...,None])[...,0]
        Xs = Xs + Tcat
    
        # last the affine, and the streching is part of it
        A_ = A[:3,:3]@torch.diag(stretch.exp()) + A[:3,-1]
        Xs = (A_[:3,:3]@Xs[...,None])[...,0] + A_[:3,-1]
    
        
    np.savez(os.path.join(outdir,'forward_transform.npz'), phi=Xs.detach().cpu().numpy(), x=np.array([x.detach().cpu().numpy() for x in xId],dtype=object))

    # Now generate + save a high res version of the inverse
    XId_ = torch.concatenate((XId,torch.ones_like(XId[...,0,None])),-1)
    Xs_ = torch.concatenate((Xs.detach(),torch.ones_like(XId[...,0,None])),-1)
    fit = torch.linalg.solve( XId_.reshape(-1,4).T@XId_.reshape(-1,4), XId_.reshape(-1,4).T @ Xs_.detach().reshape(-1,4) ).T
    FIT = (fit[:3,:3]@XId_[...,:3,None])[...,0] + fit[:3,-1]

    fig,ax = plt.subplots(2,1)
    # hfig = display(fig,display_id=True)
    OUT = []
    for i in range(0,I.shape[1],1):
        # get the location
        thisx = [[xI[0][i]],xI[1],xI[2]]
        thisX = np.stack(np.meshgrid(*thisx,indexing='ij'),-1)
        # interpolate Xs
        thisXfit = (fit[:3,:3]@thisX[...,None])[...,0] + fit[:3,-1]    
        out = interp(xId,(Xs-FIT).clone().detach().permute(-1,0,1,2),torch.tensor(thisX,**dd)).permute(1,2,3,0) + thisXfit.float()
        
        bad = out==0
        test = interp(xJd,Jd,out)
        ax[0].cla()
        ax[0].imshow((out[0] - out.min())/(out.max()-out.min()))
        ax[0].set_title(i)
        ax[1].cla()
        ax[1].imshow(test.squeeze())
        ax[1].set_title(i)
        # hfig.update(fig)
    
        # scale
        out = out - torch.tensor([xJ[0][0],xJ[1][0],xJ[2][0]],**dd)
        out = out / ( torch.tensor([xJ[0][1] - xJ[0][0],xJ[1][1] - xJ[1][0],xJ[2][1] - xJ[2][0]],**dd))
        out[bad] = -1
        OUT.append(out)
    OUT = torch.concatenate(OUT)
    OUT = OUT.numpy()

    if saveAllFigs or saveFig14:
        fig.savefig(os.path.join(outdir, 'fig14_high_res.png'))

    if saveAllFigs or saveFig15:
        fig,ax = plt.subplots()
        ax.imshow((out[0] - out.min())/(out.max()-out.min()))
        fig.savefig(os.path.join(outdir, 'fig15_high_res0.png'))

    if saveAllFigs or saveFig16:
        i = Xs.shape[0]-1
        fig,ax = plt.subplots()
        ax.imshow((Xs.detach()[i] - Xs.detach()[i].min())/(Xs.detach()[i].max()-Xs.detach()[i].min()))
        fig.savefig(os.path.join(outdir, 'fig16_high_res1.png'))

    if verbose:
        print(f'Successfully registered {fname_J} to {fname_I} and saved all outputs in {outdir}')

if __name__ == '__main__':
    main()