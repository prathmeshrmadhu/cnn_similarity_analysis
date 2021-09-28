# cnn_similarity_analysis
This is a working repository for understanding the similarity in features space for SOTA CNNs


### The Replica Project [[Link](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46604-0_52.pdf)]

#### Training Details
50 epochs to converge  
SGD with momentum (learning rate: 10−5, momentum term: 0.9)   
minimal batch size of 5   
maximum batch size of 10  
Note - (Batches are slightly tricky to make as we need each part of the triplet to have
similar sized images (i.e. all the Qi of the batch to have size s1, all the Ti,j to
have size s2 etc.). Because of this, we had to discard a small portion of the data
to make batches)  
25k training triplets   
5k validation triplets

We divided our dataset into separate sub-graphs.  
50-25-25 % --> train, valid, test splits.  
The testing set was made of 199 images.