A typical **NLP classification practice** in **Sentiment Analysis**

Reference by Yoon Kim "Convolutional Neural Networks for Sentence Classification" in 2014


Data Set: Sentiment_Analysis/data/
  - "neg.txt": the negative corpus.
  - "pos.txt": the French corpus.

DNN model: Sentiment_Analysis/sentiment_analysis.ipynb

RNN(LSTM) model: Sentiment_Analysis/sentiment_analysis.ipynb

CNN model: Sentiment_Analysis/sentiment_analysis.ipynb

What is the differences among these 3 models applied in Sentiment Analysis?
  - DNN and RNN almost have the same accuracy rate. But CNN model's accuracy on test is 1%~2% higher than DNN and RNN.
  - Overall, these models have achieved good accuracy, which is largely due to pre-trained word embedding.
  - Conclusion: DNN cannot capture sequence relationships, RNN (LSTM) can capture long-dependent sequence relationships, and CNN can capture local sequence relationships.
