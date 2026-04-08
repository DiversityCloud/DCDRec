>📋 Collaborative-Guided Diffusion for Sequential
Recommendation 

This repository is the official implementation of [Collaborative-Guided Diffusion for Sequential
Recommendation]. 

###  Framework:
This is the framework of proposed DCDRec model:
![performance](figs/Framework.png)

## Abstract
Sequential recommendation aims to predict the next item that aligns with user preferences based on their historical interaction sequence. Traditional sequence recommendation methods can be seen as understanding-based approaches, where the next relevant item is determined by analyzing user past interactions and rating patterns. Recently, diffusion-based models have emerged as a promising paradigm, focusing on learning data distributions rather than explicitly mining sequential patterns. However, existing diffusion-based methods still face two limitations. Firstly, they often map discrete target items into continuous spaces through transformations, failing to accurately model target items in a way that reflects future user preferences. Secondly, existing conditionally guided diffusion models rely heavily on explicit conditions derived from intra-sequence patterns while neglecting inter-sequence collaborative signals, which hinders the robustness of user preference modeling. To bridge the gap between collaborative signals and diffusion models, we propose DCDRec, a Dual Collaborative Signal-Guided Diffusion  Recommendation model. Specifically, for target item representation, we employ a cross-attention based encoder to obtain context-aware target item embeddings. For conditional guidance modeling, we incorporate social homophily theory and item-item affinity into the conditional generation process, introducing a dual collaborative signal-guided denoising mechanism to generate new items. Extensive experiments demonstrate the effectiveness of DCDRec and its superiority over state-of-the-art methods.

## Code Structures

```text
datasets/
├── Toys/
│   ├── train_data_date.pkl
│   ├── val_data_date.pkl
│   ├── test_data_date.pkl
│   ├── user_vocab_size_date.pkl
│   └── movie_vocab_size_date.pkl
├── Music/
├── Video/
├── ML1M/
├── ML10M/
└── Yelp/


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python main.py --dataset Toys
```

## Datasets
Our experiments are conducted on six benchmark datasets collected from Amazon (Toys, Music, and Video), Movielens (1M and 10M), and Yelp. The detailed statistical properties of these datasets are summarized below.

| Dataset | # Users | # Items | # Interactions | # Ave. length | # Data Sparsity |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Toys | 16362 | 22431 | 376278 | 7.39 | 0.9990 |
| Music | 9906 | 12381 | 346525 | 14.23 | 0.9971 |
| Video | 10194 | 32349 | 140928 | 24.68 | 0.9989 |
| Movielens 1M | 6040 | 3633 | 1000209 | 165.6 | 0.95442 |
| Movielens 10M | 69878 | 10583 | 10000054 | 143.11 | 0.9865 |
| Yelp | 21097 | 20205 | 354764 | 16.82 | 0.9991 |


## Contributing

>📋  DCDRec is released under MIT License. The dataset is available for research purposes only.
