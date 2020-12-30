# Word Vectors

A repository to explore dense representations of words.

## Useful Links

- [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)
- [http://mccormickml.com/](http://mccormickml.com/)
- [https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#examples-word2vec-on-game-of-thrones](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#examples-word2vec-on-game-of-thrones)

## Models

- [SVD-based](notebooks/svd_word_vectors.ipynb)
- LSA
- LDA
- word2vec
- GloVe

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

#### Gradient

$$
\begin{aligned}
    \frac{\partial P(w_O|w_I)}{\partial v_I} &= u_O - \sum_{w\in V}P(w_w|w_I)\cdot u_w \\
    \frac{\partial P(w_O|w_I)}{\partial u_O} &= v_I - v_I\cdot P(w_O|w_I) \\
    \frac{\partial P(w_O|w_I)}{\partial u_{w, w\in V, w\neq O}} &= -v_I\cdot P(w_w|w_I)
\end{aligned}
$$

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