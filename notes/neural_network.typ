The loss for one sample:

$ L = -log(p_t) $

where p_t is the softmax probability of the correct class t, and softmax is:
$ p_j = frac(e^(z_j), sum_k e^(z_k)) $

$ frac(partial L, partial z_j) $


Case 1: $j!=t$ (wrong class)

$ frac(partial L, partial z_j) = -frac(partial, partial z_j) log(p_t) $
$ = -frac(1, p_t) dot frac(partial p_t, partial z_j) $
The softmax cross-derivative (when j!=tj != t
j!=t) is:
$ frac(partial p_t, partial z_j) = -p_t dot p_j $
So:
$ frac(partial L, partial z_j) = -frac(1, p_t) dot (-p_t dot p_j) = p_j $
Case 2: j==tj == t
j==t (correct class)
The softmax self-derivative is:
$ frac(partial p_t, partial z_t) = p_t (1 - p_t) $
So:
$ frac(partial L, partial z_t) = -frac(1, p_t) dot p_t(1 - p_t) = -(1 - p_t) = p_t - 1 $