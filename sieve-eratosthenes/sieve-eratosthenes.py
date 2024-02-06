# Sieve of Eratosthenes
# 
# Algorithm:
# 
# 1. Create a list $l$ of integers from $2$ through $M$.
# 2. Select $p = 2$, the smallest prime number.
# 3. Compute multiples of prime $p$ up until $M$.
# 4. Mark those multiples in the original list $l$.
# 5. Repeat steps 3 & 4 until $p \gt \sqrt{M}$.
# 6. Now, all unmarked integers in $l$ are primes.

import numpy as np

# define M here
M = 1001

# create list of consecutive integers
l = np.arange(M+1)
# create boolean mask
isprime = np.ones(len(l), dtype=bool)
# set 0 & 1 to False
isprime[[0, 1]] = False

# If p^2 is greater than M, the sieve can be stopped as
# all remaining non-prime integers will already be marked!
stop = np.floor(np.sqrt(M)).astype(int)

# start loop with smallest prime (p=2)
for p in range(2, stop+1):
    # continue, if p is unmarked (otherwise, p is no prime)
    if isprime[p]:
        # compute multiples of prime p
        multis = np.arange(2*p, M+1, p) # exclude p itself
        # mark multiples of p
        isprime[multis] = False

# print unmarked integers
print(f"prime numbers for M={M}:")
print(l[isprime])
