Q: For the Gibbs (and Metropolis-Hastings) sampling, the acceptance criterion seems way too loose. It is easy to get stuck in a sampling "hole" for 10 iterations and end up with a really crummy estimation for 10 iterations in a row. A comparison threshold a few orders of magnitude smaller (say, 0.0001) seems more interesting, especially to compare the two.

- Yes, the convergence criteria has been kept loose. You can change the threshold (delta) to anything <= 0.1, if you want. Doing so might also improve the converged posterior.

Q: What about "burn in" time. That is, early iterations of the sampling algorithms may not be capturing realistic states at a proportional level. Are we allowed to run the initial state through the algorithm for a few hundred iterations to give it a better starting point?
- Yes, you can and should run the initial state a few hundred/thousand times, but do include it in your final count of iterations returned.

Q: Going back to the acceptance conditions - most writeups use a minimum (or fixed) number of iterations to run the sampling algorithms. Should we be setting a minimum? Technically, we can start testing after the first iteration, but there is a good possibility that after 11 iterations the state space has not been adequately explored and the acceptance criteria are met (too early).
- Yes, you can set a minimum.. go ahead and experiment with different values.

Q: For MH sampling, one iteration means one sample that may or may not be rejected, right? I'm checking for rejection in my MH_sampling function, and return None if it is rejected. Is this ok? You can't reject and return the previous sample, because that will misrepresent the previous sample as getting over-represented.
- You mustn't return None.

Q: Is an iteration of Gibbs ?
(a) - one sample of one variable resampled at random
(b) - a complete iteration over all variables not held constant.
- The correct answer is (a)
 Example: Given the [0..0] vector in the starter code a single call to gibb_sampling should pick one of the nodes at random and update its results. As a result on the 10-vector of zeros, the returned results would be nine zeros and one value that is potentially changed. If the index for T3 was chosen, only that variable would be updated (lets say to '2' and thus the returned value would be (0,0,2,0...0)).

Q: How will 2c and 2d be graded?
- 2c: Correctness of a new sample based on previous sample (differing in atmost 1 position)
2d: Completeness of the method to be able to converge to any posterior, preferably close to correct (but not necessary!)

Q: Bonnie says return types not matching even though I'm returning a list (or float)?
- Please check that you're not returning a numpy array (or numpy float) instead of a list (float). 

