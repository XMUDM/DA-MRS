# coding: utf-8

import numpy as np
import pandas as pd

names = ['baby', 'sports', 'clothing']

for name in names:
    image = np.load('./data/' + name + '/image_feat.npy') 
    text = np.load('./data/' + name + '/text_feat.npy') 

    kk = [0.05, 0.1, 0.15, 0.2]

    for k in kk:
        
        num_samples = image.shape[0] 
        num_samples_to_noise = int(num_samples * k)  
        
        
        indices_to_noise = np.random.choice(num_samples, num_samples_to_noise, replace=False)
        for index in indices_to_noise:
            
            replacement_index = np.random.choice(num_samples)
            while index == replacement_index:
                replacement_index = np.random.choice(num_samples)
            image[index] = image[replacement_index]
        
        
        indices_to_noise = np.random.choice(num_samples, num_samples_to_noise, replace=False)
        for index in indices_to_noise:
            replacement_index = np.random.choice(num_samples)
            while index == replacement_index:
                replacement_index = np.random.choice(num_samples)
            text[index] = text[replacement_index]
        
        np.save('./data/' + name + '/replace_image_feat_' + str(k) + '.npy', image)
        np.save('./data/' + name + '/replace_text_feat_' + str(k) + '.npy', text)   
