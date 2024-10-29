This experiment is developed for the purpose of participating in the FinCausal 2025 English sub-track contest.


The dataset consists of a text, a causal question, and an answer. The problem formulation is to predict the answer given the question and the text.


In the training set, 1988 out of 2000 data entries have the answer completely contained in the text. In the practice set, 100% of the data entries have the answer completely contained in the text.


Given the problem formulation, two approaches are devised:


Firstly, leveraging Large Language Models (LLMs) and prompt engineering in a zero-shot or few-shot learning approach to retrieve the part of the text that answers the question.


Secondly, utilizing Language Models (LMs) such as BERT to perform conditional sequence tagging, identifying the part of the text that answers the question. Specifically, this model is trained to predict whether a given token embedding, along with the question embedding, is part of the answer or not.




to run the model follow run  

python src/Model_vanilla_seq_tagger/training_loop.py 

to run the evelauation 

python src/Model_vanilla_seq_tagger/predict_example.py 
