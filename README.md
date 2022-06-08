# 575 Project
Group 1 for "Analyzing Neural Language Models" research seminar  
Group members: Qingxia Guo, Lindsay Skinner, Gladys Wang, Saiya Karamali  

## Environment Setup
The code is able to run in `Python 3.9` with appropiate requirement in `environment.yml`, one can also run the following command if they have annaconda. <br>
`conda env create -f environment.yml`

## Evaluate
User could choose to evaluate the model's performance on either ccg-tagging task or semantic role labeling task on the 
word embeddings that have either the ccg information or srl information removed using INLP. <br>
Example command for finetuning the prob for srl-tagging task using embedding without ccg information: 
````shell
$ ./src/eval.sh train -c -r
````
Example command for evaluation of ccg-tagging task performance using embedding without srl information: 
```` shell
$ ./src/eval.sh evaluate -s -g
````
Use the `-h` flag to get help.
