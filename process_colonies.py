import numpy as np
import pandas as pd
import dask.array as da
import dask_image.ndmorph as damorph
import skimage.filters as filters
from skimage.measure import label, regionprops_table
from skimage import morphology

#dask cash
from dask.cache import Cache
cache = Cache(2e9)  # Leverage two gigabytes of memory
cache.register()    # Turn cache on globally


def process_seg(seg_prob, sigma=1, threshold=0.5, closing_radius=5, min_cell_area=20, max_hole_area=100, channel_axis=1, output_intermediate=False):
    """convert probability map from Ilastik into semantic segmentation

    Parameters
    ----------
    seg_prob : dask array
        probability map of shape (frames, chanels, height, width)
    sigma : int, optional
        sigma of gaussian filter, by default 1
    threshold : float, optional
        trheshold for segmentation, by default 0.5
    closing_radius : int, optional
        radius of disk for closing, by default 5
    min_cell_area : int, optional
        minimum area of colonies to keep, by default 20
    max_hole_area : int, optional
        maximum area of holes to fill, by default 100
    channel_axis : int, optional
        color channel axis, by default 1

    Returns
    -------
    dask array
        semantic segmentation of shape (frames, height, width)
    """
    
    if seg_prob.dtype == 'uint8':
        seg_prob = seg_prob.astype('float32')/255

    # 1. smooth probabilities
    prob_sm = da.map_blocks(filters.gaussian, seg_prob, sigma, channel_axis=channel_axis-1)

    # 2. threshold image
    mask = prob_sm > threshold

    # 3. close holes
    disk = np.expand_dims(morphology.disk(closing_radius), axis=0) # create structuring element
    mask_cl = damorph.binary_closing(mask, disk)

    # 4. clean mask
    mask_cl = da.map_blocks(morphology.remove_small_holes, mask_cl, max_hole_area) #remove small holes
    mask_cl = da.map_blocks(morphology.remove_small_objects, mask_cl, min_cell_area) #remove small objects

    # 5. Convert semantic segmentation into instance segmentation

    #convert binary markers into label markers:
    labels = da.map_blocks(label, mask_cl)
    
    if output_intermediate:
        return labels, mask, mask_cl
    else:
        return labels


#function to process single frame  
def extract_prop_slice(label_im, prop_list, image=None, metadata=None):
    '''extract region properties from a single frame
    
    Parameters
    ----------
    label_im : dask array
        the label image
    prop_list : list
        list of properties to extract
    image : dask array
        fluorescent image to extract intensity properties from
    metadata : dict
        dictionary of metadata to add to the table
    
    Returns
    -------
    pandas.DataFrame
        table of region properties
    
    '''
    
    label_im = label_im.compute() if isinstance(label_im, da.Array) else label_im
    
    if image is None:
        rp_table = regionprops_table(label_im, properties=prop_list) 
    else:
        #regionprops need color channel to be at end
        image = da.moveaxis(image, 0, -1)
        rp_table = regionprops_table(label_im, intensity_image=image.compute(), properties=prop_list) 
    
    df = pd.DataFrame(rp_table)
    #add the time index
    if metadata is not None:
        for key, val in metadata.items():
            df[key] = val
    
    return df


def track_extract_prop(label_im, prop_list, metadata=None):
    '''extract region properties from a stack of frames and track colonies
    
    Parameters
    ----------
    label_im : dask array
        the label image
    prop_list : list
        list of properties to extract
    metadata : dict
        dictionary of metadata to add to the table
    
    Returns
    -------
    pandas.DataFrame
        table of region properties
    
    '''
    
    df = pd.concat([extract_prop_slice(label, 
                                       prop_list, 
                                       metadata = {'frame':t, **metadata}) 
                    for t, label in enumerate(label_im)])
    
    df = track_colonies(df, direction='forward')
    
    return df


def track_colonies(df, direction='forward'):
    '''track colonies in a dataframe of region properties
            
    Parameters
    ----------
    df : pandas.DataFrame
        table of region properties
    direction : str, optional
        forward or backward tracking, by default 'forward'
    
    Returns
    -------
    pandas.DataFrame
        table of region properties with colony ids
    
    '''
    #add colony id to dataframe
    df = df.copy()
    df['colony_id']=-1

    #set col_idx for frame 0
    frm0 = df['frame']==0
    col_idx0 = np.arange(frm0.sum())
    df.loc[frm0, 'colony_id'] = col_idx0

    #forward tracking
    for frm in range(1, df['frame'].max()+1):
        col_idx_prev = df[df['frame']==frm-1]['colony_id'].values

        #calc distance between centroids
        x0 = df[df['frame']==frm-1]['centroid-0'].values
        y0 = df[df['frame']==frm-1]['centroid-1'].values

        x1 = df[df['frame']==frm]['centroid-0'].values
        y1 = df[df['frame']==frm]['centroid-1'].values

        if direction == 'forward':
            #forward tracking
            dx = np.atleast_2d(x0).T - np.atleast_2d(x1) #row is x0, col is x1
            dy = np.atleast_2d(y0).T - np.atleast_2d(y1) #row is y0, col is y1
        elif direction == 'backward':
            #backward tracking
            dx = np.atleast_2d(x1).T - np.atleast_2d(x0) #row is x1, col is x0
            dy = np.atleast_2d(y1).T - np.atleast_2d(y0) #row is y1, col is y0
        
        ds = np.sqrt(dx**2 + dy**2)

        #get column index of minimum distance between each cell in frame 0 and frame 1
        idx = np.argmin(ds, axis=1)

        #forward tracking
        if direction == 'forward':
            #init new col_idx
            col_idx_new = -1 * np.ones(np.sum(df['frame']==frm))
            
            #make sure each colony in frame t is only matched to one colony in frame t-1
            unique_idx = np.unique(idx)
            for id_new in unique_idx:
                if np.sum(idx==id_new) == 1: #unique match
                    id_old = np.where(idx==id_new)[0][0]
                    col_idx_new[id_new] = col_idx_prev[id_old]
        elif direction == 'backward':
            col_idx_new = col_idx_prev[idx]
        
        #assign new col_idx
        df.loc[df['frame']==frm, 'colony_id'] = col_idx_new
        
    return df


def df_track_to_lin(df):
    ''' convert dataframe of tracked colonies to linear array for napari
    
    Parameters
    ----------
    df : pandas.DataFrame
        table of region properties with colony ids
    
    Returns
    -------
    numpy.ndarray
        linear array for napari
    
    '''
    lin_data = np.vstack([
        df["colony_id"].to_numpy(dtype=int), 
        df["frame"].to_numpy(dtype=int), 
        df["centroid-0"].to_numpy(dtype=int), 
        df["centroid-1"].to_numpy(dtype=int)]).T

    return lin_data[lin_data[:,0]>=0,:]


def calc_min_dist(target, source):
    """ calculate distance to closest source point for each target colony
    
    Parameters
    ----------
    target : 2D numpy.ndarray or tuple of two 1D numpy.ndarrays
        centroids of target colonies
    source : 2D numpy.ndarray or tuple of two 1D numpy.ndarrays
        centroids of source colonies
    
    Returns
    -------
    numpy.ndarray
        distance to closest source point for each target colony
    
    """
    
    #extract x and y coordinates
    if isinstance(target, tuple):
        target_x = target[0]
        target_y = target[1]
    elif isinstance(target, np.ndarray):
        target_x = target[:,0]
        target_y = target[:,1]
    else:
        raise ValueError('target must be tuple or numpy.ndarray')
    
    if isinstance(source, tuple):
        source_x = source[0]
        source_y = source[1]
    elif isinstance(source, np.ndarray):
        source_x = source[:,0]
        source_y = source[:,1]
    else:
        raise ValueError('source must be tuple or numpy.ndarray')

    #distance from target to source
    dx = np.atleast_2d(target_x).T - np.atleast_2d(source_x) #row is target col is source
    dy = np.atleast_2d(target_y).T - np.atleast_2d(source_y) #row is target col is source
    ds = np.sqrt(dx**2 + dy**2)

    return np.min(ds, axis=1) 
        

def add_centrod_distance(df):
    """ add distance to closest PA centroid to SA1 and SA2 colonies
    
    Parameters
    ----------
    df : pandas.DataFrame
        table of region properties with colony ids
    
    Returns
    -------
    pandas.DataFrame
        table of region properties with colony ids and distance to closest PA centroid
    
    """
    
    #initialize dataframe
    df = df.copy()
    df['min_dist_PA_centroid'] = np.nan

    for frm in df['frame'].unique():

        pos_SA1 = df[(df['frame']==frm) & (df['strain']=='SA1')][['centroid-0', 'centroid-1']].values
        pos_SA2 = df[(df['frame']==frm) & (df['strain']=='SA2')][['centroid-0', 'centroid-1']].values
        pos_PA = df[(df['frame']==frm) & (df['strain']=='PA')][['centroid-0', 'centroid-1']].values

        df.loc[(df['frame']==frm) & (df['strain']=='SA1'), 'min_dist_PA_centroid'] = calc_min_dist(pos_SA1, pos_PA)
        df.loc[(df['frame']==frm) & (df['strain']=='SA2'), 'min_dist_PA_centroid'] = calc_min_dist(pos_SA2, pos_PA)
        
    return df


def add_edge2edge_distance(df,PA_labels,SA1_labels,SA2_labels):
    ''' add closest distance between edge of SA and PA colonies 
    
    Parameters
    ----------
    df : pandas.DataFrame
        table of region properties with colony ids
    PA_labels : numpy.ndarray
        label image of Pseudomonas aeruginosa colonies
    SA1_labels : numpy.ndarray
        label image of Staphylococcus aureus 1 colonies
    SA2_labels : numpy.ndarray
        label image of Staphylococcus aureus 2 colonies
    
    Returns
    -------
    pandas.DataFrame
        table of region properties with colony ids and distance to closest edge of PA colony
    '''
    
    #initialize dataframe
    df = df.copy()
    df['min_dist_PA_edge2edge'] = np.nan
    
    for frm in df['frame'].unique():

        #get pixels of PA colony
        x_PA, y_PA = np.nonzero(PA_labels[frm,:])

        for i in df.index[(df['frame']==frm)]: #loop through colonies in frame
            if df.loc[i, 'strain'] == 'PA': #skip Pseudomonas aeruginosa
                continue
            if df.loc[i, 'colony_id'] == -1: #skip untracked colonies
                continue
            
            #get pixels of target colony
            target_im = SA1_labels[frm,:] if df.loc[i, 'strain'] == 'SA1' else SA2_labels[frm,:] 
            x_SA, y_SA = np.nonzero(target_im == df.loc[i, 'label'])
            
            #calculate distance to closest PA colony
            dist = calc_min_dist((x_SA, y_SA), (x_PA, y_PA))            
            df.loc[i, 'min_dist_PA_edge2edge'] = dist.min()
        
    return df
