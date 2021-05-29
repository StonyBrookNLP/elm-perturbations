import os
import itertools
from collections import namedtuple
import random
import numpy as np
import copy
import math



def intersection(lst1, lst2): 
    temp = set([item for item in lst2])
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 

 
def load_dataset(filename,mode,perturb): 
    shuffle_data = False
    counter = 0     
    all_samples = []  
    with open(filename, 'r', encoding='utf-8') as fi:
        for sample in fi: #each line is a document
            if sample != '\n':
                sample = sample.replace('\n','')
                all_samples.append(sample)

    ## To use for LM model with no perturbations
    if perturb == 'lm':
        shuffled_all_samples = []
        new_sample = sample + " <TUP>"
        new_sample = ' '.join(new_sample.split())
        shuffled_all_samples.append((new_sample,new_sample))
        random.shuffle(shuffled_all_samples)
        return shuffled_all_samples

    
    ## to create permuted instances for the LM model for sequences with event length >= 4
    if perturn == 'permute':
        if mode == 'train':
            shuffled_all_samples = []
            for sample in all_samples:
                sample_split = sample.split('<TUP>')
                if len(sample_split) >= 4:
                    input_text = ' <TUP> '.join(sample_split)
                    input_text = input_text + " <TUP>"
                    text_original = ' '.join(input_text.split())
                    shuffled_all_samples.append((text_original,text_original))

                    ## store different permutations of data
                    # add the reverse of the sequence
                    input_seq_reversed = sample_split[::-1]
                    input_seq_reversed_text = ' <TUP> '.join(input_seq_reversed)
                    input_seq_reversed_text = input_seq_reversed_text + " <TUP>"
                    text_reversed = ' '.join(input_seq_reversed_text.split())
                    shuffled_all_samples.append((text_original,text_reversed))

                    # add the sequence of even elements accompanied by odd elements
                    even_elements = sample_split[::2]
                    odd_elements = sample_split[1::2]

                    text = ' <TUP> '.join(odd_elements+even_elements)
                    text = text + " <TUP>"
                    text = ' '.join(text.split())
                    shuffled_all_samples.append((text_original,text))

                    # add the sequence of even elements accompanied by odd elements for the reverse of input
                    even_elements_reversed = input_seq_reversed[::2]
                    odd_elements_reversed = input_seq_reversed[1::2]

                    text = ' <TUP> '.join(odd_elements_reversed+even_elements_reversed)
                    text = text + " <TUP>"
                    text = ' '.join(text.split())
                    shuffled_all_samples.append((text_original,text))

            #random.shuffle(shuffled_all_samples)
            return shuffled_all_samples

        elif mode == 'eval':
            shuffled_all_samples = []
            for sample in all_samples:
                output_text = sample + " <TUP>"
                output_text = ' '.join(output_text.split())
                shuffled_all_samples.append((output_text,output_text))

            random.shuffle(shuffled_all_samples)
            return shuffled_all_samples
    
    
    ## create input sequences with dropped tuples
    if perturb == 'drop':
        if mode == 'train':
            shuffled_all_samples = []
            for sample in all_samples:
                sample_split = sample.split('<TUP>')
                output_text = ' '.join(sample.split())
                output_text = output_text + " <TUP>"

                prev_idx_list =[]
                repetition = True
                if len(sample_split) > 2:
                    for k in range(len(sample_split)//3):
                        repetition = True
                        sample_split_new = copy.deepcopy(sample_split)

                        while repetition is True:
                            idx_list = np.random.choice(range(len(sample_split)),len(sample_split)//3, replace=False)
                            if intersection(prev_idx_list, idx_list) != []:
                                idx_list = np.random.choice(range(len(sample_split)),len(sample_split)//3, replace=False)

                            else:
                                repetition = False
                                prev_idx_list = idx_list
                        if not repetition:
                            for idx in idx_list:
                                sample_split_new.remove(sample_split[idx])

                            input_text = ' <TUP> '.join(sample_split_new)
                            input_text = ' '.join(input_text.split())
                            shuffled_all_samples.append((input_text,input_text))

                else:
                    shuffled_all_samples.append((output_text,output_text))

            random.shuffle(shuffled_all_samples)
            return shuffled_all_samples

        elif mode == 'eval':
            shuffled_all_samples = []
            for sample in all_samples:
                output_text = ' '.join(sample.split())
                shuffled_all_samples.append((output_text,output_text))

            random.shuffle(shuffled_all_samples)
            return shuffled_all_samples
    
    
    ## create input sequences with masked tuples
    if perturb == 'mask':
        if mode == 'train':
            shuffled_all_samples = []
            for sample in all_samples:
                sample_split = sample.split('<TUP>')
                #output_text = '<TUP>'.join(sample_split)
                output_text = ' '.join(sample.split())
                output_text = output_text + " <TUP>"

                prev_idx_list =[]
                repetition = True
                if len(sample_split) > 2:
                    for k in range(len(sample_split)//3):
                        repetition = True
                        sample_split_new = copy.deepcopy(sample_split)

                        while repetition is True:
                            idx_list = np.random.choice(range(len(sample_split)),len(sample_split)//3, replace=False)
                            if intersection(prev_idx_list, idx_list) != []:
                                idx_list = np.random.choice(range(len(sample_split)),len(sample_split)//3, replace=False)

                            else:
                                repetition = False
                                prev_idx_list = idx_list
                        if not repetition:
                            for idx in idx_list:
                                ## for masked model, comment the next three lines if drop-out model
                                tuple_tobe_masked = sample_split[idx]
                                masked = ["<mask>" for i in tuple_tobe_masked.split()]
                                sample_split_new[idx] = ' ' + ' '.join(masked) + ' '

                                ## for drop-out model, comment the next line if masked model
                                #sample_split_new.remove(sample_split[idx])

                            input_text = ' <TUP> '.join(sample_split_new)
                            input_text = ' '.join(input_text.split())
                            shuffled_all_samples.append((input_text,output_text))

                else:
                    shuffled_all_samples.append((output_text,output_text))

            random.shuffle(shuffled_all_samples)
            return shuffled_all_samples

        elif mode == 'eval':
            shuffled_all_samples = []
            for sample in all_samples:
                output_text = ' '.join(sample.split())
                shuffled_all_samples.append((output_text,output_text))

            random.shuffle(shuffled_all_samples)
            return shuffled_all_samples

