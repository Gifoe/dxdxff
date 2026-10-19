"""Fixed parameter-matched fusion; no fitted values, labels, or patient context."""
import math


def fuse_hidden(head, raw_projection, feature_projection, split):
    bias = head[0].bias
    if not split:
        return head[1](raw_projection + feature_projection + bias)
    return (head[1](raw_projection + bias / 2)
            + head[1](feature_projection + bias / 2)) / math.sqrt(2)
