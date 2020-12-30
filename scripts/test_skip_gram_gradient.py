"""Test cost and gradient computation for skip-gram word vector model
"""

import numpy as np


def softmax(target: np.array, candidates: np.array, candidate_ind: int) -> np.float64:
    return np.exp(np.dot(candidates[candidate_ind], target)) / np.sum(np.exp(np.dot(candidates, target)))


def cost(batch: np.array, embeddings: np.array, debug: bool = False) -> np.float64:
    """Compute skip-gram cost function

    embeddings = [[-u-], [-v-]]
    embeddings.shape = (2 * vocab_size, embedding_dim)
    """
    #
    if len(batch.shape) == 1:
        batch = np.expand_dims(batch, axis=0)
    elif len(batch.shape) > 2:
        raise ValueError("Tokens array should not have more than 2 dimensions")
    
    batch_size = batch.shape[0]
    if debug:
        print(f"Batch size: {batch_size}")

    #
    n_tokens_per_sample = batch.shape[1]
    if n_tokens_per_sample % 2 == 0:
        raise ValueError("Sample size should be an odd number")

    if debug:
        print(f"Tokens per sample: {n_tokens_per_sample}")

    center_token_ind = n_tokens_per_sample // 2
    if debug:
        print(f"Center token ind: {center_token_ind}")

    #
    if embeddings.shape[0] % 2 != 0:
        raise ValueError("The first dimension of the embeddings array should be an even number")
    
    vocab_size = embeddings.shape[0] // 2
    if debug:
        print(f"Vocab size: {vocab_size}")

    #
    cost = 0.0
    for sample in batch:
        for ind, token in enumerate(sample):
            if ind == center_token_ind:
                continue

            cost += np.log(softmax(
                embeddings[vocab_size + sample[center_token_ind]],
                embeddings[:vocab_size],
                token))

    cost *= -1. / batch_size
    return cost


def gradient_fd(batch: np.array, embeddings: np.array, step_size: float = 1e-6) -> np.array:
    """"""
    cost_init = cost(batch, embeddings)

    gradient = np.zeros(embeddings.shape, dtype=np.float64)
    for ind, _ in np.ndenumerate(embeddings):
        print(ind)
        embeddings_copy = np.copy(embeddings)
        embeddings_copy[ind] += step_size
        cost_update = cost(batch, embeddings_copy)
        gradient[ind] = (cost_update - cost_init) / step_size

    return gradient


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
    print(f"Cost: {cost(batch0, embeddings, True):.6f}")
    print(f"Expected cost: {2.253856}")

    print()

    batch1 = np.array([[1,0,0], [0,1,0]], dtype=np.int64)
    print(f"Cost: {cost(batch1, embeddings, True):.6f}")
    print(f"Expected cost: {1.145078}")


def test_skip_gram_gradient() -> None:
    print("=== Test skip-gram gradient function === ")

    embeddings = np.array([
        [1,4,5,8],
        [2,3,6,7],
        [1,2,3,4],
        [1,3,5,7]],
        dtype=np.float64    
    )

    batch0 = np.array([1,0,0], dtype=np.int64)

    # Compute gradient using finite difference
    print(gradient_fd(batch0, embeddings, step_size=1e-6))
    

if __name__ == "__main__":
    test_skip_gram_cost_function()
    test_skip_gram_gradient()
