### Overview
This NLP project use a 2-layer neural network to classify 17 possible **framing** used in Congress tweets. Inputs for the model take the following format:
![](https://github.com/BonanKou/-PyTorch-Framing/blob/master/example.png?raw=tru)

- Framing comprises a set of concepts and theoretical perspectives on how individuals, groups, and societies organize, perceive, and communicate about reality. In reality, politicians carefully choose framing to place their dicussion of social affairs so public opinion can be guided in desired direction.

Back to the project, there are two working versions of the project in this repository: **a baseline model for comparison (in `base` folder)** and **an improved version (in `project` folder) of the model**. Both models complete the same task:
- Train the neural network with the first 900 labeled tweets in `train.csv` and test its performance by labeling the 900th~1200th tweets in `train.csv`. Tweets are stored in `train.csv` in the following format:
![](https://github.com/BonanKou/-PyTorch-Framing/blob/master/example2.png?raw=tru)
- After training is done, predict framings for tweets in `test_proj.csv` and print the results in `test_proj.csv`. Tweets in `test_proj.csv` are stored in the same format as `train.csv` with only the `label` column left blank.

Different from the** baseline model**, before the **improved model** starts to play with the labeled tweets in `train.csv`, it pretrains the model with tweets retrieved from `congresstweet` git repository for which **not the exact label but the possibilities for all 17 categories are known**. Intuitively, this pretraining stage would provides the improved model nothing but a head start so it converges faster. In practice, however, the improved model shows a 10% increase in accuracy than the baseline model. 

Here is an example tweet retrieved from `congresstweet` repository.
![](https://github.com/BonanKou/-PyTorch-Framing/blob/master/example3.png?raw=tru)

Both models are written with Pytorch library.
### Run the code
- To run the base model:
```
cd base
python main.py
```
After execution, **a graph showing test and train accuracies versus epoch number** will emerge. The train accuracy peaks at **95%**, meaning the model correctly labeled 95% of the training tweets. The test accuracy peaks at **27%** in baseline model, meaning our baseline model correctly labeled 27% of the test tweets.
![](https://github.com/BonanKou/-PyTorch-Framing/blob/master/base/base.png?raw=tru)

- To run the improved model:
```
cd project
python main.py
```
After execution, **a graph showing test/train accuracies in both pretrain and training stages  versus epoch number will emerge**. The train accuracy peaks at **95%** while test accuracy in stage 2 peaks at **42%** in the improved model, a **15%** improvement over baseline model.
![](https://github.com/BonanKou/-PyTorch-Framing/blob/master/project/final.png?raw=tru)

- To save the model to test_proj.csv
```
cd project
python save.py
```

### Archetecture, explained

#### Model input

##### Vocabulary dimensions
This project uses **Bag of words** approach to represent tweet texts. The model read through all the tweets from training set and create a dictionary of unique words before training starts. Each tweet is represented by a long vector of frequencies of each entry in the dictionary.  With Pytorch, a vector like this is called a **tensor** and bits in a tensor are also known as **dimensions**. For instance, the baseline model create 2717 dimensions in representation vector because it has seen 2717 unique words (excluding special characters like ~!@#$) in tweets from training set. Because the improved version uses more tweets to train, this number increases to 8112. 

##### Other dimensions
2 more dimensions of the tweet are concatenated at the end of tweet representation. 
- A bit set to 1 if the tweet's author is a democrat, 0 otherwise.
- A constant bit set to 1 that represents possible bias in the model.

In the improved model, a vector of 6 dimensions  indicating `issue` of the tweet is concatenated at the end of tweet representation. This is meant to grasp the possible relationship between tweet `issue` and the tweet's framing.

| Issue | Corresponding bits |
| ------------- | ------------- |
| guns | 0th bit  set to 1, 0 for other bits|
| isis  | 1th bit  set to 1, 0 for other bits |
| aca  | 2th bit  set to 1, 0 for other bits |
| immigration  | 3th bit  set to 1, 0 for other bits |
| abortion | 4th bit  set to 1, 0 for other bits  |
| lgbtq  | 5th bit  set to 1, 0 for other bits  |

#### Neural Network layers and Model Output
There are two **linear layers** inside this neural network model. 
- The first layer takes tweet representation as input and outputs an 100-bit tensor.
- The 100-bit tensor from the first layer is **activated** before passing to the second layer as the new input. The activation function used in this project is `F.relu()`.
- Since there are 17 possible framings for model to pick from, the second layer outputs a 17-bit tensor with value at each bit representing the possiblity of the tweet belonging one particular category.

#### Training
##### Expected output
An ideal model predicts the correct result without any hesitation. So we train the model with an expected output for a tweet with a known framing would be a 17-bit one-hot vector with only 1 bit set to 1 and all others to 0. This means our model is 100% sure a particular framing is used instead of the others.
In the improved model, we cannot build such an one-hot vector in the first training stage because we don't know what framing is used in tweets retrieved from `congresstweet` repository. Luckily, a paper from *https://www.aclweb.org/anthology/P17-1069.pdf*  shows distributions over 17 framing for tweets concerning one of the 6 issues. Instead of an one-hot vector, the 17-bit target tensor is now comprised of possibilities for each framing. For example, expected output for a tweet concerning `issue` **guns** will be:
- *[7.094596448545265e-23, 1.42498778922372e-21, ..., 3.873518413222867e-21]*
, where value at the ith index is the **softmaxed possibility** of a tweet concerning **guns** belonging to the ith category of framing.

##### One step in training
All tweet inputs and their expected outputs are stored in a self-defined `Dataset()` object that inherits `Dataset()` interface in `torch.utils.data` library. This library also provides a `DataLoader()` object that feeds the model with a batch of input/output pairs in each step of the training. The neural network udpates its internal state to minimize difference from the output to the expected output, or **loss**. When the target output is an one-hot vector, a `CrossEntropyLoss()` function is used to calculate loss. When the target output is a vector of possibilities, I used `MSELoss()` instead.

Training parameters:

| Batch size  | Epochs|Learning rate|Weight decay|
| ------------- | ------------- | ------------- | ------------- |
|50 | 46 for improved model, 25 for baseline model|0.001 |0.0001|

### Referrence
- `congresstweet` repository: https://github.com/alexlitel/congresstweets/tree/master/data

