import torch
torch.manual_seed(1337)

# toy example: batch of 4, sequence length 8, 2 channels per token
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)

# We want: for each position t, xbow[b,t] = average of x[b, 0..t]
# ("bow" = bag of words -- just an average, no attention weights yet)
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]      # (t+1, C) -- all tokens up to and including t
        xbow[b, t] = torch.mean(xprev, dim=0)

print(x[0])
print(xbow[0])

# --- Version 2: same result via a single matrix multiply ---
# Key insight: "average of the first t+1 rows" is just a weighted sum where
# each of the first t+1 rows gets weight 1/(t+1) and the rest get weight 0.
# Stack those weight-rows for t=0..T-1 and you get a (T,T) matrix.
wei = torch.tril(torch.ones(T, T))     # lower-triangular: wei[t,j] = 1 if j<=t else 0
wei = wei / wei.sum(1, keepdim=True)   # normalize each row to sum to 1
print(wei)

# (T,T) @ (B,T,C) -- torch broadcasts the batch dim automatically
xbow2 = wei @ x
print(xbow2[0])
print(torch.allclose(xbow, xbow2, atol=1e-6))

# --- Version 3: the same thing again, via softmax ---
# This is the version that generalizes to *real* attention: instead of
# hardcoding "equal weight to every past token", we start from a matrix of
# raw affinity scores (all zero for now = "no preference yet"), mask out the
# future with -inf, and let softmax turn each row into a probability
# distribution. exp(-inf) = 0, so masked positions still get exactly zero
# weight -- but now the *unmasked* weights are learnable, not fixed to 1/(t+1).
import torch.nn.functional as F

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))                          # raw affinities, all zero (no data-dependence yet)
wei = wei.masked_fill(tril == 0, float('-inf'))     # future positions -> -inf
wei = F.softmax(wei, dim=-1)                        # -inf -> 0 after softmax, rest -> uniform
print(wei)

xbow3 = wei @ x
print(torch.allclose(xbow, xbow3, atol=1e-6))
