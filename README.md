# owkin_challenge

The repo contains one python file. The python file contains three functions. The first one is used to preprocess the datas. Images are not used for predcitions but still loaded and sorted in the same ID patient order as clinical datas. The second function is a multi-heads/multi-outputs which can be used for to analyse heterogenous datas at the same time and to do both regression and classification operations as needed for this challenge. We have considered other methods about which i will be pleased to discuss during a future interview. The third function reprocess the results of the softmax just in case some results doesn't exactly equals to either 0 or 1 so that the reviewer can easily analyze the results stored in the csv. 

The csv file contains in index the patient ID and in columns the 'Survival Time' and the 'Event'. 
