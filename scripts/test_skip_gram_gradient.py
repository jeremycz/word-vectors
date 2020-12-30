"""Test cost and gradient computation for skip-gram word vector model
"""
from typing import List, Union, Tuple

import numpy as np


def softmax(target: np.array, candidates: np.array, candidate_ind: Union[np.int64, int, List[int]]) -> Union[np.float64, np.array]:
    if not isinstance(candidate_ind, (np.int64, int, list)):
        raise TypeError(f"Type of candidate_ind ({type(candidate_ind)}) needs to be one of (int, list, np.int64)")
        
    return np.exp(np.dot(candidates[candidate_ind], target)) / np.sum(np.exp(np.dot(candidates, target)))


def cost(batch: np.array, embeddings: np.array, compute_gradients: bool, debug: bool = False) -> Union[np.float64, Tuple[np.float64, np.array]]:
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
    gradient = np.zeros(embeddings.shape, dtype=np.float64)

    for sample in batch:
        p_w_c = None
        if compute_gradients:
            # Probabilty distribution over all candidate words
            p_w_c = softmax(embeddings[vocab_size + sample[center_token_ind]], embeddings[:vocab_size], list(range(vocab_size)))
            p_w_c = np.expand_dims(p_w_c, axis=1)

        for ind, token in enumerate(sample):
            if ind != center_token_ind:
                cost += np.log(softmax(
                    embeddings[vocab_size + sample[center_token_ind]],
                    embeddings[:vocab_size],
                    token))

                if compute_gradients and p_w_c is not None:
                    # d/dvc
                    gradient[vocab_size + sample[center_token_ind]] += embeddings[token] - np.sum(p_w_c * embeddings[:vocab_size], axis=0)

                    # d/duw
                    gradient[:vocab_size] -= p_w_c * np.expand_dims(embeddings[vocab_size + sample[center_token_ind]], axis=0)

                    # d/duo
                    gradient[token] += embeddings[vocab_size + sample[center_token_ind]]

    cost *= -1. / batch_size

    if compute_gradients:
        gradient *= -1. / batch_size
        return cost, gradient

    return cost


def gradient_fd(batch: np.array, embeddings: np.array, step_size: float = 1e-6) -> np.array:
    """Computes gradient of the skip-gram cost function using finite differencing"""
    cost_init = cost(batch, embeddings, compute_gradients=False)

    gradient = np.zeros(embeddings.shape, dtype=np.float64)
    for ind, _ in np.ndenumerate(embeddings):
        embeddings_copy = np.copy(embeddings)
        embeddings_copy[ind] += step_size
        cost_update = cost(batch, embeddings_copy, compute_gradients=False)
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
    cost_val, gradient = cost(batch0, embeddings, compute_gradients=True, debug=True)
    print(f"Cost: {cost_val:.6f}")
    print(f"Expected cost: {2.253856}")
    print("Finite difference gradient:")
    print(gradient_fd(batch0, embeddings, step_size=1e-6))
    print("Analytical gradient:")
    print(gradient)

    print()

    batch1 = np.array([[1,0,0], [0,1,0]], dtype=np.int64)
    cost_val, gradient = cost(batch1, embeddings, compute_gradients=True, debug=True)
    print(f"Cost: {cost_val:.6f}")
    print(f"Expected cost: {1.145078}")
    print("Finite difference gradient:")
    print(gradient_fd(batch1, embeddings, step_size=1e-6))
    print("Analytical gradient:")
    print(gradient)


if __name__ == "__main__":
    test_skip_gram_cost_function()
