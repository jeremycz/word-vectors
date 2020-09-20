# Word Vectors

## SVD

### Methodology

1. Loop over corpus and accumulate word co-occurrence counts in a matrix $X$
2. Apply SVD to get $X = USV^T$
3. Use the rows of $U$ as the word embeddings for all words in the dictionary (use the first $k$ columns to get $k$-dimensional word vectors)

Variance captured:

$$
\frac{\sum_{i=1}^k\sigma_i}{\sum_{j=1}^{|V|}\sigma_j}
$$

where $\sigma$ are the singular values of $X$ contained in the diagonal of $S$.

Issues:

- $X$ is very sparse since most words do not co-occur
- $X$ is very large - expensive to perform SVD (computational cost for $m\times n$ matrix is $\mathcal{O}(mn^2)$)
- Need to make adjustments

## LSA

## LDA

## word2vec

### References

1. [2013, Mikolov et al. Efficient Estimation of Word Representations in Vector Space. arxiv:1307.3781v3](https://arxiv.org/pdf/1301.3781.pdf)
2. [2013, Mikolov et al. Distributed Representations of Words and Phrases and their Compositionality. NIPS 2013.](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### Theory

Two model architectures:

1. Continuous Bag-of-Words (CBOW) - uses context words to predict target word
2. Continuous Skip-gram - uses target word to predict context word

### Skip-gram Model

#### Objective function

$$
\textnormal{Likelihood} = L(\theta) = \prod_{t=1}^{T}\prod_{-c\leq j\leq c, j\neq 0}P(w_{t+j}|w_t;\theta)
$$

Use negative log-likelihood for better scaling

$$
J(\theta) = -\frac{1}{T}\log L(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\leq j\leq c, j\neq 0}\log P(w_{t+j}|w_t;\theta)
$$

- $c$ is the size of the training context (which can be a function of the center word $w$)
- Larger $c$ - more training examples, higher accuracy, increased training time

The probability $P(w_{t+j}|w_t)$ is calculated using the softmax function:

$$
P(w_O|w_I) = \frac{\exp(u_O^Tv_I)}{\sum_{w=1}^{W}\exp(u_w^Tv_I)}
$$

- $u_w$ and $v_w$ are the 'input' and 'output' vector representations of $w$, and $W$ is the size of the vocabulary
- This formulation is impractical computationally because it requires computing the softmax over all the representations in the vocabulary

### CBOW Model

The probability $P(w_t|w_c)$ is calculated using the softmax function:

$$
\textnormal{Likelihood} = L(\theta) = \prod_{t=1}^{T}P(w_t|\{w_j\}_{-c\leq j\leq c, j\neq 0};\theta)
$$

$$
P(w_I|w_C) = \frac{\exp(u_I^Tv_C)}{\sum_{w=1}^{W}\exp(u_w^Tv_C)}
$$

where $v_C$ is the sum of 'output' representations of all words in the context window:

$$
v_C = \sum_{-c\leq j\leq c, j\neq 0}v_j
$$

#### Objective function




## GloVe