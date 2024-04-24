Two networks A and B. wlog,
1. ✅ Do warm-up training for A (give image, get caption)
2. ✅ Start "real" training and record loss $L$ for how well each caption matches the image
3. ✅ Fit a two-component GMM to $L$
4. ✅ Retrieve probability of clean caption—w = p(g|l), g is Gaussian component with smaller mean/loss—and spilt captions into clean set and noisy set based on a threshold τ on w. This is for B's use
5. ✅ Some magical fun on B
    a. ✅ Take minibatches, C and N, of both clean set and noisy set in the form of (image, caption)
    b. ✅ Apply augmentations to images, get C' and N'
    c. ✅ Clean: combine captions generated for C', use as refined label. Augmented set now C_hat
    d. ✅ Noisy: combine captions generated for N' from BOTH A and B, used as guessed label. Augmented set now N_hat
6. ✅ Do MixMatch on C_hat and N_hat
7. Update B's parameters with mixmatch losses (and its losses from step 5,, somehow)
8. Loop
9. Profit