# Final Project
## Background
This GitHub Repository houses the coding pipeline to re-implement the "Explainable automated coding of clinical notes using hierarchical label-wise attention networks and label embedding initialisation", and to extend research with incorporating BERT embeddings into this model, which we title "BHLAN: BERT Integration Into Hierarchial Label-Wise Attention Networks for Automatic Clinical Coding". This was developed as part of UW-Madison's CS 769 Advanced Natural Language Processing for Fall 2024. The underlying code-base to perform data pre-processing, embedding generation and model training are present in the repository, however due to the DUA for MIMIC-III no data is available for reproducing the results of this re-implementation. To run the pipeline for each of the models, utilize the files as follows:
## Data Pre-Processing
### Data Folder
#### notebooks subfolder
- Run dataproc_mimic_III.ipynb according to the CAML github [CAML](https://github.com/jamesmullenbach/caml-mimic/tree/master) to process the MIMIC-III text. The CAML code was utilized in the original HLAN model paper for pre-processing the data, so we replicated this process using the available CAML code. You will run this with the MIMIC-III Data to get the splits based on the MIMIC-III-50 train, dev and test.
## HLAN Model
### Embeddings Folder
Run the following python files, replacing the folder paths with the data splits for the MIMIC-III data to pre-train the models and generate the word2vec embeddings
- pre-train_embeddings.py: Code to pre-train the word and label CBoW models
- code-emb.mode: Saved pre-trained label model
- word-emb.model: Saved pre-trained word model
- word-emb.model.syn1neg.npy: Additional file for word model
- word-emb.model.wv.vectors.npy: Additional file for the word model
- generate_code_embeddings.py: Code to generate embeddings from the pre-trained label model for the notes corresponding labels
- generate_word_embeddings.py: Code to generate embeddings from the pre-trained word model for the notes
- embed_data_helper.py: Helper functions to generate embeddings
### HLAN_Model Folder
Run main_run.py with HLAN.py  to train the model for 250 epochs, with additional shell scripts under Train subfolder. The results can be displayed, and shown with the Plots.ipynb.
- HLAN.py: Main pytorch model for HLAN
- Plots.ipynb: Notebook for training performance monitoring plots and determining best performing epochs
- main_run.py: Main model training and evaluation code
- model_helper.py: Helper function for model main_run (unused)
- Train/run_python.sh: Shell script to run HLAN model
- Train/run_python.sub: Script to submit job to Condor
- Train/scenario_list.txt: List of scenarios to run

## BHLAN Model
### Data Folder
#### notebooks subfolder
- Run dataproc_mimic_III.ipynb to process the text for the 25 word, 100 sentence split
- Run updated_caml_dataproc_mimic_III_for_sentence_split.ipynb to process the medical documents to ensure they can be split by their sentence boundaries for the BERT_Embeddings tokenization process
## HLAN Model
### BERT_Embeddings Folder
Run the generate_bert_token_embeddings.py file to get the BERT embeddings. Change the helper function to bert_embed_helper_sent_split.py to get the sentence split embeddings.
- generate_bert_token_embeddings: Script to get the tokenized embeddings for all splits of the data for all BERT models
- bert_embed_helper.py: Helper python script with functions to tokenize the documents
- bert_embed_helper_sent_split.py: Helper python script to tokenize and split the text based on the actual sentence boundaries

### BERT_HLAN Folder
Run main_run_bert.py with BERT_HLAN.py to train the BERT models with different scenarios. Use the shell scripts under Train subfolder, as well as the Plots.ipynb to get the results.
- BERT_HLAN.py: Main pytorch model for BHLAN
- main_run_bert.py: Main model training and evaluation code
- Train/Plots.ipynb: Notebook for training performance monitoring plots and determining best performing epochs
- Train/Condor/instance_list.txt: List of scenarios to run
- Train/Condor/run_python.sh: shell script to run main_run_bert with scenarios from instance_list
- Train/Condor/run_python.sub: Condor submit file
- Train/CS_GPU/bert_hlan_sent_split_dim_red.sh: shell script to run main_run_bert for sentence split/dim reduction
- Train/CS_GPU/bert_hlan_sent_split_script.sh: shell script to run main_run_bert for sentence_split

### xAI Folder
To analyze explainability of the models, run the HLAN_xai_notebook.ipynb with the BERT_HLAN_xai.py, HLAN_xai.py and HLAN_xai_helper.py. Thee top_ten.pkl file is used as reference for the index for the labels that are most prevalent in the corpus
- BERT_HLAN_xai.py: BERT_HLAN pytorch model for BHLAN, with added functionality for xAI
- HLAN_xai.py: HLAN pytorch model, with added functionality for xAI
- HLAN_xai_helper.py: helper functions for xAI notebook
- HLAN_xai_notebook.ipynb: Notebook to run to generate examples of explainabililty and seaborn plots
- top_ten.pkl: Top Ten labels within dataset, with indices

## Additional Notes:
Outside of this process, there are additional items in the github in the Data folder.
- Data/Training_results: folder housing the results from the HLAN model. The models for BERT are too large to store on Github, so it was decided to scrap replicating these stored results for BERT.
- Data/Data Analysis/HLAN_Data_Analysis.ipynb: Notebook to perform data analysis on the label space for HLAN, as well as the tokenization process for using BERT within HLAN.


