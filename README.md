# Helper News Generator by Transformer


### Summary

Generating news by computers is not always bad for people. There are some news that are basic and do not have an impact on people's lives. Like weather, sports, product description which can be automated and also be helpful. Doing this there are some important topics that needs to be in consideration. The output should not be offending people, It should explain simple news in short paragraphs. Journals can use this to fill in some news in their website's or magazines. The models initial knowledge can be used to be specialize in domain specific like sports news or weathers news.

I used Multihead-Attention Transformer model and trained it on [All the news](https://www.kaggle.com/snapcrack/all-the-news) with 143,000 articles from 15 American publishers mostly between the years 2016-2017.

### Outline

1. Data Preprocessing
2. Data Exploration
3. Model Building
4. Training
5. Tuning
6. Enjoying the model
7. Conclusion

### Data Preprocessing
For Data Preprocessing and data loading PyTorch provides iterators to load data as its expected. Text data easily fills RAM space and crashes the training procedure. So we used Iterator to yield text in batches and [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) to tokenize data. BPE is used for GPT-2 Model.

### The GPT-2 Model

Firstly we tried training the baseline  [model](https://github.com/kaanbursa/NewsGenerator/blob/master/Model%20try.ipynb) with self created none-pre-trained model with Encoder - Decoder structure where the Decoder is a Fully Connected Layer for the Encoder we used Multihead-Attention Attention. We only tried it on a small dataset to see if any

One of the most important feature of Transformer model is you can you pre-trained model and fine-tune it. We use [Hugging Face's](https://huggingface.co/) pretrained GPT-2 model which uses only **Decoder** Layer of transformer models.

GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1] of 8 million web pages [1](https://huggingface.co/transformers/model_doc/gpt2.html). We are going to fine-tune it on our newspaper dataset.

![GPT-2](http://jalammar.github.io/images/gpt2/gpt2-simple-output-2.gif)
[Courtesy of Jammar](http://jalammar.github.io/illustrated-gpt2/)

**Byte Pair Encoding**
Byte Pair Encoding is a word tokenization algorithm used by GPT-2 model to overcome some of the limitations of Word2Vec on capturing similarities on words with suffixes like smart - smart-er.
[Detailed Explanation](https://leimao.github.io/blog/Byte-Pair-Encoding/)

**Positional Encoding**
Positional Encoding is an add on to Embedding vector which encodes the distance between words.[1](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model)

**Decoder**
