# X-Transformer

Models based on Transformers are wildly successful for a wide variety of Natural Language Processing (NLP) tasks and consequently are a mainstay of modern NLP research. Transformer is constituted of a self-attention and a feed-forward module. The self-attention mechanism allows each token in the input sequence to attend independently to every other token in the sequence. From the view of graph representation, the generalized attention mechanism can be described by a Undirected Complete Graph whose vertex is the token. So, the attention module can be implemented by a graph library, especially recently the efficient attention implementation, e.g.  [BigBird](https://arxiv.org/abs/2007.14062) \ [LongFormer](https://arxiv.org/abs/2004.05150) \ [Sparse Transformer](https://arxiv.org/abs/1904.10509). 

We have showcased the [BigBird](https://arxiv.org/abs/2007.14062) implementation and tested the performence as show below, and the [LongFormer](https://arxiv.org/abs/2004.05150) \ [Sparse Transformer](https://arxiv.org/abs/1904.10509) can be easily implemented by revised the correspoding code. 



## Dependencies

- [paddlepaddle >= 1.7](https://github.com/PaddlePaddle/paddle)
- [pgl 1.1](https://github.com/PaddlePaddle/PGL)


## Performance

We have evaluate the implemented method on a summarization dataset CNN/DM. The experiment was conducted on two P40 GPU cards. 

| CNN/DM             | BatchSize | R1                | R2                | R3                | speed(steps/s)  |
| ------------------ | --------- | ----------------- | ----------------- | ----------------- | ------ |
| LEAD               | - | 40.42             | 17.62             | 36.67             | - |
| Oracle             | - | 52.59             | 31.24             | 48.87             | - |
| non-sparse,  L=512 | 32        | 42.175            | 19.392            | 38.613            | 0.6359 |
| L=2048             | 10        | 41.334            | 18.369            | 37.752            | 0.8246 |
| L=1024             | 20        | 41.453            | 18.529            | 37.872            | 0.6432 |
| L=768              | 26        | 41.611            | 18.735            | 38.051            | 0.6517 |
| L=512              | 40        | 41.742            | 18.733            | 38.127            | 0.6213 |

**\**** For this task, we warm up from ERNIE 2.0 en directly rather than pretrain the model for the additional position embedding, so the embedding for the position which is larger than 512 is used repeatedly from ERNIE 2.0.
This may cause score degradation. But in the future, we will test the pre-trained model.

