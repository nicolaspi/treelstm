Forked from 
https://github.com/sapruash/RecursiveNN

Tensorflow implementation of N-ary Recursive Neural Networks using LSTM units as
described in "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks" by Kai Sheng Tai, Richard Socher, and Christopher D. Manning.

This implementation builds a meta-tree per minibatch that aggregates all its samples. 
A meta-tree is then processed heightwise. Each nodes with a given height h depends 
only from nodes of height h-1, this allows to aggregate all matrix mutliplications within h. 
Thus the number of matrix multiplication per mini batch can be reduced from O(MxN) to O(log(N)) where M is the mini batch size and N the number of nodes.
The result is a training time 70x faster with the reference model : https://github.com/sapruash/RecursiveNN.

Usage :

- From your shell run : 

    ./fetch_and_preprocess.sh 
    
  to download and preprocess data. (This may take a while)
- Then : 
    
    python tf_sentimentmain.py -optimized

  to train the model on the SST dataset with optimized model version, without argument the sapruash's version is launched.




