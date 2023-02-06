#main data analysis packages
import numpy as np
import pandas as pd

#image processing packages
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

#out of memory computation
import dask.array as da
import dask


@dask.delayed
def get_shift(frm0, frm1):
    dxy, _, _ = phase_cross_correlation(np.squeeze(frm0),np.squeeze(frm1))
    return dxy

@dask.delayed
def apply_shift(frm, dyx):
    return ndi.shift(frm, [0,*dyx], cval=0, order=0)

def register_pos_calc(pos, ch=0):
    shift = da.zeros((pos.shape[0],2), chunks=(1, 2))
    for t in range(pos.shape[0]-1):
        dxy = get_shift(pos[t,ch,:,:], pos[t+1,ch,:,:])
        shift[t+1,:] = da.from_delayed(dxy, shape=(1, 2), dtype=np.float64)
    return np.cumsum(shift, axis=0)    

def register_pos_apply(pos, transform):
    reg_im = da.empty_like(pos)
    for t in range(pos.shape[0]):
        im = apply_shift(pos[t,:,:,:], transform[t,:])
        reg_im[t,:,:,:] = da.from_delayed(im, shape=(pos.shape[1:]), dtype=np.float64)
    return reg_im

def register_pos(pos):
    transform = register_pos_calc(pos).compute()
    return (register_pos_apply(pos, transform), transform)


def register_movie(raw_data, outpath, outname):
    reg_data = da.empty_like(raw_data)
    translation = np.zeros((raw_data.shape[0],raw_data.shape[1],4))
    for p in range(raw_data.shape[1]):
        pos = raw_data[:,p,:,:,:]
        reg_pos, trans = register_pos(pos)
        translation[:,p,:] = np.column_stack([np.arange(raw_data.shape[0]), p*np.ones(raw_data.shape[0]), trans])
        reg_data[:,p,:,:,:] = reg_pos 
        
        outname = outpath / (outname + ('_reg_p%03i.h5' % p))   
        reg_pos.to_hdf5(outname, '/images')

    # translation metadata
    translation_df = pd.DataFrame(data = translation.reshape((-1,4)), columns = ['frame', 'pos', 'dy', 'dx']).astype(int)
    outname = outpath / (outname + '_reg.csv') 
    translation_df.to_csv(outname)   
    
    return (reg_data,translation)