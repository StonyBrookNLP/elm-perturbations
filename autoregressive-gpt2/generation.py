'''
Generate schemas using the seed events
To use: python generation.py --seeds_file="seeds.txt" --generation_file="generation.txt" --output_dir="baseline" --generation_type=beam

'''
import random
import numpy as np
import argparse
import torch
from transformers import *

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",type=str, default="/Users/Mahnaz/Documents/KAIROS/schema_generation")
parser.add_argument("--seeds_file",type=str, default="/Users/Mahnaz/Documents/KAIROS/robust_generation_idea/preproc_data/seeds.txt")
parser.add_argument("--generation_file",type=str, default="/Users/Mahnaz/Documents/KAIROS/robust_generation_idea/preproc_data/generated_schemas.txt")
parser.add_argument("--generation_type", type=str, default="beam",help="it can be greedy, beam, sampling")
parser.add_argument("--seed", type=int, default=12)
parser.add_argument("--output_length", type=int, default=40)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))


## load a pre-trained GPT2LM model
model_class = GPT2LMHeadModel
tokenizer_class = GPT2Tokenizer
pretrained_weights = 'gpt2'

# Load the model 
tokenizer = tokenizer_class.from_pretrained(args.output_dir)
model = model_class.from_pretrained(args.output_dir)
model.to(device)


model.eval()

## read seeds from the seeds file to generate event sequences
samples =[]
with open(args.seeds_file, 'r', encoding='utf-8') as fi:
    for seed in fi:
        samples.append(seed)

## go over all the samples and generate the output with respect to the seed event.
with open(args.generation_file, 'w') as g: 
    for sample in samples:
        if sample == '\n':
            g.write('\n')
        else:
            sample = sample.strip()
            sample_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample)) 
            input_ids = torch.tensor(sample_tokenized).unsqueeze(0)  
            input_ids = input_ids.to(device)
            
            ## for beam search
            if args.generation_type == "beam":
                beam_output = model.generate(
                input_ids, 
                max_length=100, 
                num_beams=1,
                repetition_penalty=1.4,
                #no_repeat_ngram_size=4, ##to remove the repetitive n-grams
                early_stopping=True
                )
                out = tokenizer.decode(beam_output[0], skip_special_tokens=True)
                generation = ' '.join(out.split())
                generation = generation.replace('!', '')
                g.write(generation)
                g.write('\n')

            
            ## for greedy search 
            if args.generation_type == "greedy":
                greedy_output = model.generate(input_ids, max_length=args.output_length)
                out = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
                generation = ' '.join(out.split())
                generation = generation.replace('!', '')
                g.write(generation)
                g.write('\n')
            
            
            ## activate sampling and deactivate top_k by setting top_k sampling to 0
            if args.generation_type == "sampling":
                sample_output = model.generate(
                    input_ids, 
                    do_sample=True, 
                    max_length=args.output_length, 
                    top_k=0,
                    temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                    #top_p=0.92
                )
                out = tokenizer.decode(sample_output[0] skip_special_tokens=True)
                generation = ' '.join(out.split())
                generation = generation.replace('!', '')
                g.write(generation)
                g.write('\n')
                
            
