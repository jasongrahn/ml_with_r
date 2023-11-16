chapter_3, Nearest Neighbors
================
jason grahn
2023-11-15

> Traditionally, the k-NN algorithm uses Euclidean distance, which is
> the distance one would measure if it were possible to use a ruler to
> connect two points. Euclidean distance is measured “as the crow
> flies,” which implies the shortest direct route.

> The decision of how many neighbors to use for k-NN determines how well
> the model will generalize to future data.

> The balance between overfitting and underfitting underfitting the
> training data is a problem known as the **bias-variance tradeoff**.

Begin with a *k* equal to the square root of the number of training
examples. (Not a hard rule, but a fast rule.)

the larger the training dataset, the larger the pool of neighbor
examples, the easier to classify.

# Preparing data for use with k-NN

> Features are typically transformed to a standard range prior to
> applying the k-NN algorithm.

aka: Normalize or standardize your variables that go into training. If
some variables have a huge range while others do not, then *k*-NN is
more heavily influenced by the larger-ranged variables.

There are lots of ways to do this:

- min-max normalization
  - transforms items to a scale from 0-1 (0 - 100%)
  - $X_{new} = \frac{X-min(X)}{max(X)-min(X)}$
  - problem’s arise if test data has values beyond the min/max
    evaluated, so need to take theoreticals into account.
- z-score standardization
  - how many standard deviations above or below the mean
  - assumes normally distributed data.
  - no predefined min and max values, so could float *anywhere*
- dummy coding
  - converting categories into *binary* variables
  - case when X then 1 else 0 (I like this one quite a bit, very useful)
- “one-hot encoding”
  - more conversionof categories into binary variables
  - case when hot then 1 = hot else 0, when medium then 1 else 0, else
    cold
  - this creates multiple binaries
  - one-hot makes problems with linear-regression.
  - however, DOES make models easier to understand.

# what is “lazy”

systems that merely store training verbatim. You tell it what to do, it
goes and does it. Heavy reliance on training rather than building an
abstracted model is called “instance-based learning”. And because it
doesn’t build a model it’s called “non-parametric” because it doesn’t
learn about any of the parameters.

# Example – diagnosing breast cancer with the k-NN algorithm
