Two networks A and B. wlog,
1. Do warm-up training for A (give image, get caption) with entropy penalization
2. Start “real” training and record loss $L$ for how well each caption matches the image
3. Fit a two-component GMM to $L$ using EM algorithm
4. Retrieve probability of clean caption—w = p(g|l), g is Gaussian component with smaller mean/loss—and spilt captions into clean set and noisy set based on a threshold τ on w. This is for B's use
5. Some magical fun on B
    1. Take minibatches, C and N, of both clean set and noisy set in the form of (image, caption)
    2. Apply augmentations to images, get C' and N'
    3. Clean: ask LLM to generate best combination of captions from captions generated for C', use as refined label. (Also apply temperature sharpening???) Augmented set now C_hat
    4. Noisy: ask LLM to generate best combination of captions from captions generated for N' from BOTH A and B, used as guessed label. (Apply temperature sharpening???) Augmented set now N_hat
7. Do MixMatch/SeqMix on C_hat and N_hat
8. Update B's parameters with mixmatch losses (and its losses from step 5?)
9. Loop
10. Profit
