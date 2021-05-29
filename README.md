# Sequence Perturbations

Code for the ACL 2021 paper "Don't Let Discourse Confine Your Model: Sequence Perturbations for Improved Event Language Models"

The code consists of two parts:
The first part is the implementation of the sequence perturbations for autoregressive models, gpt-2 in our case. The gpt-2 implementaion is based on Huggingface library and the run_language_modeling.py script. 

The second part is the implementation of the HAQAE model. We have used the implementation in (HAQAE paper) and added the modifications required for perturbations. You can find the modified scripts here but please refer to (Noah's work) for the complete implementation of HAQAE.

To run the gpt-2 model, you can use a script similar to:
python train.py --train_dataset="train.txt" --eval_dataset="dev.txt" --output_dir="baseline" --num_train_epochs=1 

Once the model is trained, you can use generate event sequences given the seed events as follows:
python generation.py --seeds_file="seeds.txt" --generation_file="generation.txt" --output_dir="baseline" --generation_type=beam


For more details on different arguments and their usage, you can have a look into the files. 

To run the HAQAE model you can use the following:
python train.py --train_data='train.txt' --valid_data='val.txt' --vocab='vocab.pkl' --epochs=2 --cuda 

To enable perturbations for the input sequences, you need to set the corresponding variables in data_utils.py.

For detaile regarding different parameters of the HAQAE model, please refer to (Noah's work)



Please email <mkoupaee@cs.stonybrook.edu> for questions and support. 
 