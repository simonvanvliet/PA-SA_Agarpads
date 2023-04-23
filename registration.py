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
    '''Calculate the translation to register two frames
    
        Parameters
        frm0: numpy array of shape (height, width)
        frm1: numpy array of shape (height, width)
        
        Output
        dxy: numpy array of shape (2,) with columns [dy, dx]
    
    '''
    dxy, _, _ = phase_cross_correlation(np.squeeze(frm0),np.squeeze(frm1))
    return dxy

@dask.delayed
def apply_shift(frm, dyx):
    '''Apply a translation to a frame
    
        Parameters
        frm: numpy array of shape (channels, height, width)
        dyx: numpy array of shape (2,) with columns [dy, dx]
        
        Output
        reg_frm: numpy array of shape (channels, height, width)
    '''
    return ndi.shift(frm, [0,*dyx], cval=0, order=0)

def register_pos_calc(pos, ch=0):
    '''Calculate the translation to register a position
    
        Parameters
        pos: dask array of shape (frames, channels, height, width)
        ch: int, the channel to use for the registration
        
        Output
        shift: numpy array of shape (frames, 2) with columns [dy, dx]
    '''
    shift = da.zeros((pos.shape[0],2), chunks=(1, 2))
    for t in range(pos.shape[0]-1):
        dxy = get_shift(pos[t,ch,:,:], pos[t+1,ch,:,:])
        shift[t+1,:] = da.from_delayed(dxy, shape=(1, 2), dtype=np.float64)
    return np.cumsum(shift, axis=0)    

def register_pos_apply(pos, transform):
    '''Apply a translation to a position
    
        Parameters
        pos: dask array of shape (frames, channels, height, width)
        transform: numpy array of shape (frames, 2) with columns [dy, dx]
        
        Output
        reg_im: dask array of shape (frames, channels, height, width)
    '''
    reg_im = da.empty_like(pos)
    for t in range(pos.shape[0]):
        im = apply_shift(pos[t,:,:,:], transform[t,:])
        reg_im[t,:,:,:] = da.from_delayed(im, shape=(pos.shape[1:]), dtype=np.float64)
    return reg_im

def register_pos(pos):
    '''Register a position and return the registered position and the translation metadata
    
        Parameters
        pos: dask array of shape (frames, channels, height, width)
        
        Output
        reg_pos: dask array of shape (frames, channels, height, width)
        transform: numpy array of shape (frames, 2) with columns [dy, dx]
    
    '''
    transform = register_pos_calc(pos).compute()
    return (register_pos_apply(pos, transform), transform)


def register_movie(raw_data, outpath, outname, save_images=True, save_metdata=True, max_frames=None):
    '''Register a movie and save the registered movie and the translation metadata
    
        Parameters
        raw_data: dask array of shape (frames, positions, channels, height, width)
        outpath: path to save the registered movie and metadata
        outname: name of the registered movie and metadata
        save_images: boolean to save the registered movie to disk
        save_metdata: boolean to save the translation metadata to disk
        max_frames: numpy array of shape (positions,) with the maximum number of frames to register
        
        Output
        reg_data: dask array of shape (frames, positions, channels, height, width)
        translation: numpy array of shape (frames, positions, 4) with columns [frame, pos, dy, dx]
        '''
    reg_data = da.zeros_like(raw_data)
    translation = np.zeros((raw_data.shape[0],raw_data.shape[1],4))
    for p in range(raw_data.shape[1]):
        
        max_idx = max_frames[p] if max_frames is not None else raw_data.shape[0]    
        max_idx = int(max_idx) if max_idx is not np.nan else raw_data.shape[0]   
                  
        pos = raw_data[:max_idx,p,:,:,:]

        reg_pos, trans = register_pos(pos)
        
        trans_full = np.full((raw_data.shape[0],2), np.nan)
        trans_full[:max_idx,:] = trans
        
        translation[:,p,:] = np.column_stack([np.arange(raw_data.shape[0]), p*np.ones(raw_data.shape[0]), trans_full])
        reg_data[:max_idx,p,:,:,:] = reg_pos 
        
        if save_images:
            outname = outpath / (outname + ('_reg_p%03i.h5' % p))   
            reg_pos.to_hdf5(outname, '/images')

    # translation metadata
    if save_metdata:
        translation_df = pd.DataFrame(data = translation.reshape((-1,4)), columns = ['frame', 'pos', 'dy', 'dx'])
        translation_df['frame'] = pd.to_numeric(translation_df['frame'], downcast='integer')
        translation_df['pos'] = pd.to_numeric(translation_df['pos'], downcast='integer')
        outname = outpath / (outname + '_reg.csv') 
        translation_df.to_csv(outname)   
    
    return (reg_data,translation)