'''
Creating a new GPT model for training the dataset

To use:  python train.py --train_dataset="train.txt" --eval_dataset="dev.txt" --output_dir="baseline" --num_train_epochs=1 
'''

import argparse
import torch
from transformers import *
import logging
import random
import numpy as np
from tqdm import tqdm, trange
from data_preprocess import load_dataset
import os
from torch.nn import CrossEntropyLoss


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai-gpt", help="pretrained model_old name")
parser.add_argument("--perturb",type=str,default="lm", help="perturbation to be applied to data, lm:no perturbation, permute, mask, drop for permuted, masked and drop-out model respectively.")
parser.add_argument("--do_train", action="store_true", default="True", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", default="True", help="Whether to run eval on the dev set.")
parser.add_argument("--output_dir",type=str, default="/Users/Mahnaz/Documents/KAIROS/schema_generation",
    help="The output directory where the model_old predictions and checkpoints will be written.",
)
parser.add_argument("--train_dataset", type=str, default="/Users/Mahnaz/Documents/KAIROS/robust_generation_idea/preproc_data/ollie_extraction_data_newform_rand_dev.txt")
parser.add_argument("--eval_dataset", type=str, default="/Users/Mahnaz/Documents/KAIROS/robust_generation_idea/preproc_data/ollie_extraction_data_newform_rand_dev.txt")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training \
                    steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before\
                    performing a backward/update pass.",
)
parser.add_argument("--learning_rate", type=float, default=6.25e-5)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lm_coef", type=float, default=0.9)
parser.add_argument("--n_valid", type=int, default=374)

parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
## load a pre-trained GPT2LM model_old
model_class = GPT2LMHeadModel
tokenizer_class = GPT2Tokenizer
pretrained_weights = 'gpt2'

## loading a pretrained model
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
tokenizer.add_tokens(['<TUP>','_NULL_','<mask>'])
model = model_class.from_pretrained(pretrained_weights)
model.resize_token_embeddings(len(tokenizer))

## loading an already trained model from directory
#tokenizer = tokenizer_class.from_pretrained(args.output_dir)
#model = model_class.from_pretrained(args.output_dir)

model.to(device)
configuration = model.config

def prepare_data(filename,input_len,mode):
    all_samples = load_dataset(filename,mode,args.perturb)
    n_batch = len(all_samples)
    input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
    label_ids = np.zeros((n_batch, input_len), dtype=np.int64)
    for i in range(len(all_samples)):
        sample_input = all_samples[i][0]
        sample_output = all_samples[i][1]
        
        sample_inp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample_input))
        sample_inp = sample_inp[:input_len]
        input_ids[i, :len(sample_inp)] = sample_inp
        
        sample_out = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample_output))
        sample_out = sample_out[:input_len]
        label_ids[i, :len(sample_out)] = sample_out
        
    return input_ids , label_ids

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

logger.info("Encoding dataset...")
train_inputs, train_outputs = prepare_data(args.train_dataset,50,mode='train')
eval_inputs, eval_outputs = prepare_data(args.eval_dataset,50,mode='eval')

        
# Prepare optimizer
if args.do_train:
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_inputs) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_inputs) // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # Load optimizer and scheduler state
    #optimizer.load_state_dict(torch.load(os.path.join(args.output_dir,'optimizer.pt')))
    #scheduler.load_state_dict(torch.load(os.path.join(args.output_dir,'scheduler.pt')))

if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for t in trange(int(args.num_train_epochs), desc="Epoch"):            
            tr_loss = 0
            nb_tr_steps = 0
            batch_idxs = np.random.permutation(len(train_inputs)//args.train_batch_size)
            line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
            
            for batch_idx in line_tqdm:
                batch_inputs = train_inputs[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_inputs))]
                batch_inputs = torch.tensor(batch_inputs)
                
                batch_outputs = train_outputs[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_outputs))]
                batch_outputs = torch.tensor(batch_outputs)
                
                input_ids = batch_inputs.to(device)
                output_ids = batch_outputs.to(device)
                
                losses = model(input_ids,labels=output_ids)
                loss = losses[0]

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = (
                    loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                )
                nb_tr_steps += 1
                if nb_tr_steps % 50000 == 0:
                    model_to_save = model.module if hasattr(model, "module") else model  # Only save the model itself
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_pretrained(args.output_dir)
                    torch.save(optimizer.state_dict(),os.path.join(args.output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(),os.path.join(args.output_dir, 'scheduler.pt'))
                        
                line_tqdm.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])
            
            model_to_save = model.module if hasattr(model, "module") else model  # Only save the model itself                
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(optimizer.state_dict(),os.path.join(args.output_dir, 'optimizer.pt'))
            torch.save(scheduler.state_dict(),os.path.join(args.output_dir, 'scheduler.pt'))

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch_idx in range(len(eval_inputs)//args.eval_batch_size):
                batch_inputs = eval_inputs[batch_idx*args.eval_batch_size:min((batch_idx+1)*args.eval_batch_size, len(eval_inputs))]
                batch_inputs = torch.tensor(batch_inputs)
                input_ids = batch_inputs.to(device)
                
                batch_outputs = eval_outputs[batch_idx*args.eval_batch_size:min((batch_idx+1)*args.eval_batch_size, len(eval_outputs))]
                batch_outputs = torch.tensor(batch_outputs)
                output_ids = batch_outputs.to(device)
        
                with torch.no_grad():
                    mc_loss, mc_logits = model(input_ids,labels=output_ids)[:2]
                
                mc_logits = mc_logits.detach().cpu().numpy()
                mc_labels = input_ids.to("cpu").numpy()
                tmp_eval_accuracy = accuracy(mc_logits, mc_labels)
        
                eval_loss += mc_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
        
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
        
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            train_loss = tr_loss / nb_tr_steps if args.do_train else None
            result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy, "train_loss": train_loss}
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
        
            
