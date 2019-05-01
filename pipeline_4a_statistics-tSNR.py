#!/usr/bin/env python
# coding: utf-8

# In[92]:


import nilearn
import nibabel as nib

import os
import pandas as pd
import numpy as np
import nibabel as nib
import glob
import re

from nilearn.plotting import find_xyz_cut_coords
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.input_data import NiftiMasker, NiftiMapsMasker
from nilearn.image import concat_imgs, index_img
from nilearn._utils import check_niimg
#%matplotlib inline


# In[94]:


def get_fn(sub, ses, run=1, echo_n=None, combmode=None, 
             root_dir='/home/stevenm/MultiEchoEPISeq/data/deriv/fmriprep'):
    if not isinstance(sub, str):
        sub = str(sub).zfill(2)

    if ses == 'se':
        hdr = os.path.join(root_dir, 'sub-{}/ses-se/func/sub-{}_ses-se_task-stop_run-{}_space-MNI152NLin2009cAsym_desc-preproc-hp_bold.nii.gz'.format(sub, sub, run))
        mask = os.path.join(root_dir, 'sub-{}/ses-se/func/sub-{}_ses-se_task-stop_run-{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub, run))
    elif ses == 'me':
        mask = os.path.join(root_dir, 'sub-{}/ses-me/func/sub-{}_ses-me_task-stop_run-{}_echo-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(sub, sub, run))
        if echo_n is None and combmode is not None:
            hdr = os.path.join(root_dir, 'sub-{}/ses-me/func/sub-{}_ses-me_task-stop_run-{}_space-MNI152NLin2009cAsym_desc-preproc-hp-{}_bold.nii.gz'.format(sub, sub, run, combmode))
        elif echo_n is not None and combmode is None:
            hdr = os.path.join(root_dir, 'sub-{}/ses-me/func/sub-{}_ses-me_task-stop_run-{}_echo-{}_space-MNI152NLin2009cAsym_desc-preproc-hp_bold.nii.gz'.format(sub, sub, run, echo_n))
        else:
            raise(IOError('Requires either combmode or echo_n'))
    
    return hdr, mask    

def make_tSNR_map(sub, ses, run=1, echo_n=None, combmode=None, save=True):
    
    hdr_fn, mask_fn = get_fn(sub, ses, run, echo_n, combmode)
    hdr = nib.load(hdr_fn)
    mask = nib.load(mask_fn)
    
    masker = NiftiMasker(mask)
    data = masker.fit_transform(hdr)

    sig = data.mean(0)
    noise = data.std(0)
    tsnr = sig/noise

    sig_img = masker.inverse_transform(sig)
    noise_img = masker.inverse_transform(noise)
    tsnr_img = masker.inverse_transform(tsnr)
    tsnr_img = concat_imgs([tsnr_img, sig_img, noise_img])
    
    nib.save(tsnr_img, hdr_fn.replace('_bold', '_tsnr'))
    return hdr_fn.replace('_bold', '_tsnr')


# In[ ]:


subs = np.arange(1, 19)

for sub in subs:
    print(sub, end=': ')
    if not os.path.exists(get_fn(sub, 'se')[0].replace('_bold', '_tsnr')):
        make_tSNR_map(sub, 'se')
    if sub == 12:
        continue

    print('oc', end='... ')
    if not os.path.exists(get_fn(sub, 'me', combmode='optcomb')[0].replace('_bold', '_tsnr')):
        make_tSNR_map(sub, 'me', combmode='optcomb')

    print('paid', end='... ')
    if not os.path.exists(get_fn(sub, 'me', combmode='PAID')[0].replace('_bold', '_tsnr')):
        make_tSNR_map(sub, 'me', combmode='PAID')

    print('echo 1', end='... ')
    if not os.path.exists(get_fn(sub, 'me', echo_n='1')[0].replace('_bold', '_tsnr')):
        make_tSNR_map(sub, 'me', echo_n=1)

    print('2', end='... ')
    if not os.path.exists(get_fn(sub, 'me', echo_n='2')[0].replace('_bold', '_tsnr')):
        make_tSNR_map(sub, 'me', echo_n=2)

    print('3', end='... ')
    if not os.path.exists(get_fn(sub, 'me', echo_n='3')[0].replace('_bold', '_tsnr')):
        make_tSNR_map(sub, 'me', echo_n=3)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script pipeline_4a_statistics-tSNR.ipynb')


# In[48]:


# atlas with attributes
fns = glob.glob('./masks/final_masks_mni09c_1p6mm/*')
fns.sort()
names = [re.match('.*space-(?P<space>[a-zA-Z0-9]+)_label-(?P<label>[a-zA-Z0-9]+)_probseg.nii.gz', fn).groupdict()['label'] for fn in fns]
roi_dict = dict(zip(names, fns))

# make nice plot
from nilearn import image
atlas = image.concat_imgs(roi_dict.values())
# labels = dict(zip(np.arange(len(roi_dict)), roi_dict.keys()))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
atlas = AttrDict({'maps': atlas, 'labels': roi_dict.keys()})


# In[ ]:


# get all niftis


# In[86]:


def get_volume(img, volume=0):
    
    if isinstance(img, str):
        img = nib.load(img)
        
    data = img.get_data()[:,:,:,volume]
    
    return nib.Nifti1Image(data, img.affine)
    
all_imgs = []
for sub in np.arange(1, 12):
    bold, mask = get_fn(sub, 'se', run=1)
    print(bold)
    tsnr = bold.replace('_bold', '_tsnr')
    
    all_imgs.append(get_first_volume(tsnr))


# In[5]:


data = image.concat_imgs(all_imgs)
data.shape


# In[ ]:





# In[30]:


def make_4D(img):
    data = img.get_data()
    return nib.Nifti1Image(data[:,:,:,np.newaxis], img.affine)


# In[34]:


test = make_4D(index_img(data, 0))


# In[33]:





# In[50]:


## NiftiLabelMasker
masker = NiftiMapsMasker(atlas.maps,
                         mask_img=mask,
                         standardize=False,
                         detrend=False,
                         low_pass=None,
                         high_pass=None)

# get tsnr from multi-volume
results = masker.fit_transform(test)

# For weird atlases that have a label for the background
if len(atlas.labels) == results.shape[1] + 1:
    atlas.labels = atlas.labels[1:]

index = pd.Index(np.arange(0, test.shape[-1], 1), name='subject_id')
columns = pd.Index(atlas.labels, name='roi')

pd.DataFrame(results, index=index, columns=columns)


# In[39]:


results


# In[37]:


index = pd.Index(np.arange(0, test.shape[-1], 1), name='subject_id')
columns = pd.Index(atlas.labels, name='roi')

pd.DataFrame(results, index=index, columns=columns)


# In[36]:


results


# In[ ]:


results


# In[62]:


index = pd.Index(np.arange(0, data.shape[-1], 1),
                 name='subject_id')

columns = pd.Index(atlas.labels, name='roi')


# In[7]:


pd.DataFrame(results, index=index,columns=columns)


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotting.plot_stat_map(get_volume(data, 0))


# In[13]:


plotting.plot_roi(get_volume(atlas.maps, 4))


# In[14]:


find_xyz_cut_coords(get_volume(atlas.maps, 4))


# In[79]:


plotting.plot_stat_map(get_volume(data, 1), cut_coords=[-50, -4, 42], vmax=150)


# In[55]:


get_volume(data, 0).get_data().min()


# In[38]:


# manually mask


# In[77]:


map_id = 7


# In[78]:


np.sum(test.get_data()[:,:,:,0] * get_volume(atlas.maps, map_id).get_data())/get_volume(atlas.maps, map_id).get_data().sum()


# In[81]:


sub = '01'
run = 1
root_dir = './data/deriv/fmriprep'
hdr = os.path.join(root_dir, 'sub-{}/ses-se/func/sub-{}_ses-se_task-stop_run-{}_space-MNI152NLin2009cAsym_desc-preproc-hp_bold.nii.gz'.format(sub, sub, run))
hdr = nib.load(hdr)

data = hdr.get_data()
tsnr = data.mean(-1)/data.std(-1)


# In[82]:


tsnr_img = nib.Nifti1Image(tsnr, hdr.affine)


# In[85]:


plotting.plot_stat_map(tsnr_img, vmax=150)


# In[91]:


sub = '01'
run = 1
root_dir = './data/deriv/fmriprep'
hdr = os.path.join(root_dir, 'sub-{}/ses-se/func/sub-{}_ses-se_task-stop_run-{}_space-MNI152NLin2009cAsym_desc-preproc-hp_tsnr.nii.gz'.format(sub, sub, run))
plotting.plot_stat_map(index_img(hdr, 2))

