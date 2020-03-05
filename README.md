# Helper News Generator by Transformer


## Summary

Generating news by computers is not always bad for people. There are some news that are basic and do not have an impact on people's lives. Like weather, sports, product description which can be automated and also be helpful. Doing this there are some important topics that needs to be in consideration. The output should not be offending people, It should explain simple news in short paragraphs. Journals can use this to fill in some news in their website's or magazines. The models initial knowledge can be used to be specialize in domain specific like sports news or weathers news.

I used Multihead-Attention Transformer model and trained it on [All the news](https://www.kaggle.com/snapcrack/all-the-news) with 143,000 articles from 15 American publishers mostly between the years 2016-2017.

## Outline

1. Data Preprocessing
2. Model Building
3. Training
5. Enjoying the model
6. Limitations
7. Conclusion

## Data Preprocessing
For Data Preprocessing and data loading PyTorch provides iterators to load data as its expected. Text data easily fills RAM space and crashes the training procedure. So we used Iterator to yield text in batches and [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) to tokenize data. BPE is used for GPT-2 Model.

## The GPT-2 Model

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

GPT-2 Architecture only contains decoder model which includes Masked Attention Mechanism with Layer Normalization and 12 of these layers. Which is than passed to FC with out token size.


## Training

Training on text data is hard since it occupies lots of memory on RAM. Also GPT-2 large model is huge model which occupies a lot of space in the memory. We use PyTorch's DatasetIterator library to load text in batches from the text file on the fly. We tried several Sequence Length to train not drain memory and 512 seems to be working fine where the GPT-2 tokenizer cannot tokenize more than 1024 words.

## Model outputs

The dataset that the model is trained on is old so it works better on older topics. Some of the new topics do not produce coherent results.

**Input**: Kanye West has said on a public comment -  **Continued**: Tuesday (Feb. 23) that Kanye must step down as the president of this whole system. As reported by David Sirota, West signed N.W.A' s first two and a half albums with Michael "Puff Daddy" Puff Jr. While he regrets the actions of N.W.A. when he was younger, West insists that he played no role in N.W.A. forming after he left Knappensack.

**Input**:Obama Administration is considering - **Continued**: a military strike against Syria as a direct result of alleged chemical weapon use, Barack Obama says the using of such weapons "elses imminent danger, and in fact it's part of a grave international effort to cover up and whitewash war crimes."

Speaking in Berlin, Obama told the German audience that the attempted use of vials of toxic chemical weapons after a gassing of people in the Syrian city of Khan Sheikhun in the last four weeks was unacceptable and that the red line that Obama.

## Limitations

#### Size
The model is to big and there are some tries on Distilling the model and reducing the model size and produce the same performance.

#### Bias
Unfortunetly  there is Bias in data which the model captures lets see this example.
**Input**: Kim Kardashian has won the nobel prize
**Output**: His mind has been investing in philanthropy since A Billion Lives came out ...

The model has seen Kim Kardashian on the news most probabily however it decided that Nobel Prize winners are men so it continued as He/His


## Conclusion

GPT-2 is an amazing model for multi-task and can be fine tuned for a lot of text. The goal is to reduce the size of the model and find a way to fine-tune the model the new information.
