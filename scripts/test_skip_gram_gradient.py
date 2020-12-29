"""Test gradient computation for skip-gram word vector model
"""

import numpy as np


def softmax(target: np.array, candidates: np.array, candidate_ind: int) -> np.float64:
    return np.exp(np.dot(candidates[candidate_ind], target)) / np.sum(np.exp(np.dot(candidates, target)))


def cost(tokens: np.array, embeddings: np.array) -> np.float64:
    """Compute skip-gram cost function

    embeddings = [[-u-], [-v-]]
    embeddings.shape = (2 * vocab_size, embedding_dim)
    """
    #
    if len(tokens.shape) == 1:
        tokens = np.expand_dims(tokens, axis=0)
    elif len(tokens.shape) > 2:
        return
    
    batch_size = tokens.shape[0]
    print(f"Batch size: {batch_size}")

    #
    n_tokens_per_sample = tokens.shape[1]
    print(f"Tokens per sample: {n_tokens_per_sample}")
    if n_tokens_per_sample % 2 == 0:
        return

    center_token_ind = n_tokens_per_sample // 2
    print(f"Center token ind: {center_token_ind}")

    #
    if embeddings.shape[0] % 2 != 0:
        return
    
    vocab_size = embeddings.shape[0] // 2
    print(f"Vocab size: {vocab_size}")

    #
    cost = 0.0
    for sample in tokens:
        for ind, token in enumerate(sample):
            if ind == center_token_ind:
                continue

            cost += np.log(softmax(
                embeddings[vocab_size + sample[center_token_ind]],
                embeddings[:vocab_size],
                token))

    cost *= -1. / batch_size
    return cost


def test_skip_gram_cost_function() -> None:
    print("=== Test skip-gram cost function === ")

    embeddings = np.array([
        [1,4,5,8],
        [2,3,6,7],
        [1,2,3,4],
        [1,3,5,7]],
        dtype=np.float64    
    )

    batch0 = np.array([1,0,0], dtype=np.int64)
    print(f"Cost: {cost(batch0, embeddings):.6f}")
    print(f"Expected cost: {2.253856}")

    print()

    batch1 = np.array([[1,0,0], [0,1,0]], dtype=np.int64)
    print(f"Cost: {cost(batch1, embeddings):.6f}")
    print(f"Expected cost: {1.145078}")


def test_skip_gram_gradient() -> None:
    print("=== Test skip-gram gradient function === ")
    pass


if __name__ == "__main__":
    test_skip_gram_cost_function()
    # test_skip_gram_gradient()
