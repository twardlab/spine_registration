import argparse
from spine_reg import * # TODO: Add path argument in final version
import os


def main():
    """
    Command Line Arguments:
    =======================

    Raises:
    =======
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('fname_I', type=str, help = 'The path to the interpolated atlas .npz file')
    parser.add_argument('fname_L', type=str, help = 'The path to the interpolated atlas labels .npz file')
    parser.add_argument('fname_J', type=str, help = 'The path to the spine reflection .npz file')
    parser.add_argument('fname_pointsJ', type=str, help = 'The input .swc file used for more precise registration')
    parser.add_argument('outdir', type=str, help = 'The location where all output files should be saved')
    parser.add_argument('-saveFigAtlas', action = 'store_true', help = 'Default - False; If present, save a MIP of the interpolated atlas provided')
    parser.add_argument('-saveFigAtlasLabels', action = 'store_true', help = 'Default - False; If present, save a MIP of the interpolated atlas labels provided')
    parser.add_argument('-saveScatter', action = 'store_true', help = 'Default - False; If present, save a scatter plot of all the points labeled in the interpolated atlas porvided')

    args = parser.parse_args()

    fname_I = args.fname_I
    fname_L = args.fname_L
    fname_J = args.fname_J
    fname_pointsJ = args.fname_pointsJ
    outdir = args.outdir
    saveFigAtlas = args.saveFigAtlas
    saveFigAtlasLabels = args.saveFigAtlasLabels
    saveScatter = args.saveScatter

    # ===================================
    # ===== Load interpolated atlas =====
    # ===================================
    data_I = np.load(fname_I,allow_pickle=True)
    I = data_I['I']
    xI = data_I['xI']
    I = I / I.max() # Normalize atlas values

    lowert = 0.15
    M = I<lowert # find a mask for the background
    I[M] = 1.0
    I = (1 - I)
    I = I - lowert
    I[I<0] = 0
    
    I = I/I.max()

    if saveFigAtlas:
        fig, axs = draw(I,xI,function=getslice)
        fig.savefig(os.path.join(outdir, 'interp_atlas.png'))

    # ==========================================
    # ===== Load interpolated atlas labels =====
    # ==========================================
    data_L = np.load(fname_L,allow_pickle=True)
    L = data_L['I']%256
    xL = data_L['xI']

    if saveFigAtlasLabels:
        fig, axs = draw((L%256)==16,xI,)
        fig.savefig(os.path.join(outdir, 'interp_atlas_labels.png'))

    qIU = np.stack(np.meshgrid(*xI,indexing='ij'),-1)[(L[0]%256)==16]
    nqU = 1000
    qIU = qIU[np.random.permutation(qIU.shape[0])[:nqU]]

    if saveScatter:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*qIU.T)
        fig.savefig(os.path.join(outdir, 'scattered_labels.png'))

    SigmaQIU = []
    for j in range(3):
        d2 = (qIU[:,None,j] - qIU[None,:,j])**2
        d2i = []
        for i in range(d2.shape[0]):
            d2i.append( np.min( d2[i][d2[i]>0] ) )
        SigmaQIU.append( np.mean(d2i) )
    SigmaQIU = np.array(SigmaQIU)

    # ==========================================
    # ===== Load the spine reflection file =====
    # ==========================================
    J = np.load(fname_J.replace('.npz','_I.npy'))
    xJ = [
        np.load(fname_J.replace('.npz','_xI0.npy')),
        np.load(fname_J.replace('.npz','_xI1.npy')),
        np.load(fname_J.replace('.npz','_xI2.npy')),
    ]
    
    