# d-gex-reproduction
Mitchell de Boer - m.m.deboer-1@student.tudelft.nl\
Jan-Willem van Rhenen - j.w.vanrhenen@student.tudelft.nl\
Delft University of Technology

## Introduction
This repository contains our reproduction of the 2016 paper: _Gene expression inference with deep learning_ by _Y. Chen et al_ [1]. This reproduction is part of the 2021 course: Deep Learning, given at the TU Delft. We have reproduced the results of this paper on one of their datasets (the GEO dataset). Using a more modern deep learning implementation than used in the original paper (pytorch vs. pylearn2) we found a much better higher accuracy. In this blog, we will first give some biological background, as well as explain the method used in data preprocessing and the model. We will then present our results and discuss the difference between our results and the original paper. 

### Background
The 2016 paper is about gene expression profiling. The expression of certain genes has been shown to correlate strongly with the expression of certain other genes. Earlier research [2] has found 943 genes (from now on referred to as landmark genes) that allow one to predict expression levels of ~80% of all genes in the human genome (consisting of ~22000 genes). By measuring the expression levels of these landmark genes, one can infer the expression levels of most other genes in the human genome. However, the problem is that the relations between these landmark and target genes are not known. The original paper tries to find these relations by means of a deep learning neural network. Other methods have been tried, like linear regression [2]. However, linear regression fails to capture non-linear relations between the landmark and target genes. Another proposed method by the paper is support vector machines (SVM), which has been successfully applied to other bioinformatics problems (for example [3]). However, SVM’s suffer from poor scalability.

## Method
### Data preparation
#### Data Acquisition
The raw data used for this project can be found on the  D-GEX data repository [4], for this project we only used the GEO dataset which is called bdgev2 in this repository. The data has a structure as illustrated in table 1. In addition to this dataset we also used the text files used in the original project containing all landmark genes and selected target genes and combined target genes. These textfiles can be found on the github page of D-GEX [5].
 
|        | Sample 1               | Sample 2            | Sample 3            |
|--------|------------------------|---------------------|---------------------|
| Gene A | Expression value A1    | Expression value A2 | Expression value A3 |
| Gene B | Expression value B1    | Expression value B2 | Expression value B3 |
| Gene C | Expression value C1    | Expression value C2 | Expression value C3 |

_Table 1: This figure shows the structure of the dataset._

#### Data Processing
The data is processed in similar fashion as the article, for which the exact implementation can be found in the article's supplementary files [1]. The size of the dataset, as expected, was huge. We chose to instead do it on a local device on Matlab, as preprocessing the data in google colab proved impossible.
 
To start off with the data processing we used parse_gctx from the cmapm library to import the gctx data file to Matlab. This imports a struct file with multiple matrices. The matrices used from this struct file are the data matrix (data.mat), which includes multiple samples of gene expression per gene, and the gene name matrix (data.rid). The data is then clustered into 100 clusters in order to make finding duplicate genes quicker. 
Duplicates are identified by taking the pairwise euclidean distance in the clusters and finding genes that have a distance of less than 1 with other genes. Genes are removed so that there are no more genes left with a distance less than 1. The text files are then used in order to find the landmark and target genes. In some target genes multiple genes are combined, so in that case the means needs to be taken for the combined genes. This results in a combination of 943 landmark genes and 9520 target genes. This data is then quantile normalized.

Quantile normalization is a technique where two distributions are made to have the same statistical properties. It involves ordering off all the data within each sample(column). Followed by the averaging per row and substituting these cells with the average. Finally a re-ordering step is done in order to get all the values back in their original position Fig. 1.

![Quantile normalization](<Images/Quantile normalization.png>)

_Fig. 1: This figure shows the principle of quantile normalization. Image source: [6]._

During the initial stages of our project we tried uploading the full dataset to Google Colab in order to train our network. However due to the size of the preprocessed data (4Gb) we ran into RAM issues on Google Colab. In order to alleviate this problem we decided to only work with 20% off the data, which was randomly selected. This also meant that we did not have to split the target genes in two, as was done in the paper.
The uploaded data was then again normalized in order to make the network have less trouble training, this time using the z-score normalization method. This normalized data was separated into training and test data of which the training data was 80% of the dataset and the test data 20%.

### Network

![Fully connected neural network](<Images/FCNN.png>)

_Fig. 2: Neural network with a single hidden layer. The input layer has 943 neurons, one for each landmark gene. The network had 1, 2 or 3 fully connected hidden layers, each layer consisting of 3000, 6000 or 9000 hidden nodes. The output layer has 9520 neurons, corresponding to the amount of target genes. image adapted from [7]._

The network is a fully connected neural network with an input layer equal to the number of landmark genes (N = 943) and an output layer equal to the number of preprocessed target genes (N = 9520). The different configurations of the network contain between one and three hidden layers, with each hidden layer containing 3000, 6000 or 9000 neurons. The paper used a Mean Squared Error (MSE) loss function in order to train the model. In order to evaluate the model the paper used a Mean Absolute Error (MAE) loss function. Using these loss functions, the paper created an adaptive learning rate function. The learning rate started at 5e-4 for a single hidden layer, or 3e-4 for multiple hidden layers. If the evaluation loss (MAE) function didn’t show any improvement, they multiplied the learning rate with a factor of 0.9, until they reached a minimum learning rate of 1e-5. Weights of all hidden layers were initialized with an uniform distribution between -W and W, with: W =  \frac{ \sqrt{6}}{ \sqrt{N_{Nodes In} +  N_{Nodes Out}}}.

#### Optimizations
The authors use a momentum based approach that updates the parameters using accumulated velocity in the direction of the gradient. They also used dropout layers after each hidden layer with a chance of 0%, 10% or 25% in order to test the optimal rate of dropout. They found 10% to be optimal, so that was the dropout rate we used in our paper.

#### Implementation
The authors used python and a library called pylearn2 in order to train and test their model. We used the more modern pytorch library, which has greater functionality and is better optimized than pylearn2. We wrote our code in google colab.

## Results
Training on the original learning rate of 5e-4 meant our weights went to infinity. Therefore we first tested lower learning rates, as can be seen in the figures below. The original learning rate had a curve similar to figure 3a.

![Test loss for a single epoch](<Images/single epoch.png>)

_Fig. 3: This figure shows the MAE during 1 epoch over multiple training iteration. For each of the subplots a different starting learning rate is chosen. These learning rates(LR) are as follows: a) LR= 5e-5, this learning rate is too high and causes weights to go to infinity. b) LR= 5e-6, this is the highest LR for which the weights do not go to infinity. c) LR = 5e-7, d)Lr = 5e-8, e) LR = 5e-9, this learning rate is the highest learning rate where the error does not converge to 0.25 within 1 epoch, f) LR = 5e-10._

We decided to use a learning rate of 5e-9, as that learning rate seemed like a good compromise between fast training times and convergence on the optimum.

![Training and test loss for a 20 epochs](<Images/20 epochs.png>)
 
_Fig.4: This figure shows the MAE during 20 epochs for a learning rate of 5e-9. it shows that even with a lower learning rate and more epochs the accuracy does not become better then 0.25. This is an example of the learning curve that is created using 1 hidden layer and 3000 hidden units. Other configuration curves can be found in the appendix._

| LR=5e-9 | 1 layer       | 2 layers      | 3 layers         |
|---------|---------------|---------------|------------------|
| 3000 HU | 0.2501/0.3421 | 0.2500/0.3337 | 0.2500/0.3300    |
| 6000 HU | 0.2501/0.3377 | 0.2500/0.3280 | 0.2500/0.3224    |
| 9000 HU | 0.2501/0.3362 | 0.2500/0.3252 | 0.2500/0.3204    |

_Table 2: This table shows the MAE of the network using several network structures. The left result in each cell is our own result, the right result in each cell is the result in the original paper._

## Discussion
For our implemented network the errors of the network converge to 0.25 for all settings (1-3 layers, 3000-9000HU). This is different from the paper where they got an error ranging from 0.32 to 0.35. The difference in results from our method could be as a result of a few factors in which we implemented the network differently than what the paper suggested. One of these factors is that we used Pytorch in order to create our neural network while the paper used Pylearn2. Pytorch is a more recent library, which could mean that it is more optimized and that it therefore creates a better result. 
Another factor in which we differ from the paper is that we needed to use a lower learning rate. When using the learning rate as suggested in the paper our weights go to infinity, which forced us to use a lower learning rate. This difference in learning rate however does not explain the difference in results because with every learning rate (that did not cause our weights to go to infinity) the error converged to 0.25. 
A third factor that could have influenced the results has to do with a restriction that we encountered while using Google Colab. Google Colab restricts the amount of RAM that can be used by a program, making us unable to run the network on the entire dataset. In order to still get an accurate representation of the data we randomly selected 20% of the data. It is possible that with this 20% we randomly got data which was more correlated which made our error lower. 
Overall with these factors our error was a lot lower than that of the paper.

The fact that all of our network configurations result with an error of 0.25 we think is odd. However this could have something to do with that the 943 landmark genes only capture about 80% of the information in our data. This would cause the setup to have an inherent error of 20% with an added error of 5% as a result of our network or the partial data.

### Comments on paper
As stated above, the original error was quite high. A few oddities were noted in the original paper, which might explain this high error. First and foremost, the adaptive learning rate. An adaptive learning rate is a common tool used in order to optimize a neural network. However, the implementation used in this paper seems rather unrefined. The authors of the paper multiply the learning rate by 0.9 every time the test loss function (MAE) becomes higher (up to a minimum learning rate). This means that every iteration the test loss funcion stochastically becomes higher, the learning rate decreases. Modern optimizers like Adam solve implement this adaptive learning rate more cleanly.

## References
1.	https://academic.oup.com/bioinformatics/article/32/12/1832/1743989?login=true#84798257
2.	http://www.lincsproject.org/
3.	Ye G. et al. . (2013) Low-rank regularization for learning gene expression programs. PloS One, 8, e82146
4.	https://cbcl.ics.uci.edu/public_data/D-GEX/
5.	https://github.com/uci-cbcl/D-GEX/
6.	https://www.nature.com/articles/s41598-020-72664-6
7.	https://www.researchgate.net/figure/Left-Fully-connected-network-with-one-hidden-layer-Each-arrow-represents-multiplication_fig8_320075165_

## Appendix

![Training and test loss for a 20 epochs for a single hidden layer](<Images/single hidden layer.png>)

_Fig. 5: This figure shows the learning curve for different configurations of the neural network for1 hidden layers._

![Training and test loss for a 20 epochs for multiple hidden layers](<Images/multiple hidden layers.png>)
 
_Fig. 6: This figure shows the learning curve for different configurations of the neural network for 2 and 3 hidden layers._






