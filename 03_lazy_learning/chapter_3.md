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

What we’re doing to do is the whole chapter normally, then if we feel
like doing it in tidy style, a redo section..

## base R

### step 1 collection

``` r
wbcd_tidy <- read_csv(here::here("03_lazy_learning", "data", "wisc_bc_data.csv")) %>% 
    janitor::clean_names() %>% 
    mutate(diagnosis = factor(diagnosis, levels = c("B", "M"),
                              labels = c("Benign", "Malignant"))
    )
```

    ## Rows: 569 Columns: 32
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (1): diagnosis
    ## dbl (31): id, radius_mean, texture_mean, perimeter_mean, area_mean, smoothne...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
wbcd <- read.csv("data/wisc_bc_data.csv")[-1]

wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))
```

### step 2 exploration and data prep

``` r
table(wbcd$diagnosis)
```

    ## 
    ##    Benign Malignant 
    ##       357       212

``` r
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)
```

    ## 
    ##    Benign Malignant 
    ##      62.7      37.3

``` r
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])
```

    ##   radius_mean       area_mean      smoothness_mean  
    ##  Min.   : 6.981   Min.   : 143.5   Min.   :0.05263  
    ##  1st Qu.:11.700   1st Qu.: 420.3   1st Qu.:0.08637  
    ##  Median :13.370   Median : 551.1   Median :0.09587  
    ##  Mean   :14.127   Mean   : 654.9   Mean   :0.09636  
    ##  3rd Qu.:15.780   3rd Qu.: 782.7   3rd Qu.:0.10530  
    ##  Max.   :28.110   Max.   :2501.0   Max.   :0.16340

the data ranges for these values are massively different. it’s going to
be hard to do any kNN on them. need to normalize them somehow.

Oh, the book gives us a function for min-max normalization.

``` r
normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
}

normalize(c(1,2,3,4,5))
```

    ## [1] 0.00 0.25 0.50 0.75 1.00

``` r
normalize(c(10,20,30,40,50))
```

    ## [1] 0.00 0.25 0.50 0.75 1.00

we’re going to use `lapply` to apply the `normalize` function we just
made to each numeric feature in the dataset. `lapply` and it’s tidyverse
counterpart `map` are a bit mysterious to me, so i’ll use the code from
the book.

``` r
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
summary(wbcd_n$area_mean)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.0000  0.1174  0.1729  0.2169  0.2711  1.0000

Now we’re going to divide the data into training and testing data. I
know there’s a tidy version of doing this too.

``` r
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]
```

``` r
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]
```

### step 3 training

``` r
library(class)
wbcd_test_pred <- knn(train = wbcd_train,  #469 rows
                      test = wbcd_test,
                      cl = wbcd_train_labels, #469 rows
                      k = 21)
```

getting error: \> Error in knn(train = wbcd_train, test = wbcd_test, cl
= wbcd_train_labels\[, : ‘train’ and ‘class’ have different lengths

error fixed by adding `$diagnosis` to the `cl` parameter.

### step 4 model performance

``` r
library(gmodels) #for evaluating model performance
```

``` r
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred,
             prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        61 |         0 |        61 | 
    ##                  |     1.000 |     0.000 |     0.610 | 
    ##                  |     0.968 |     0.000 |           | 
    ##                  |     0.610 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        37 |        39 | 
    ##                  |     0.051 |     0.949 |     0.390 | 
    ##                  |     0.032 |     1.000 |           | 
    ##                  |     0.020 |     0.370 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        63 |        37 |       100 | 
    ##                  |     0.630 |     0.370 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

(I really dont like this format for results. Partly because I’m not used
to seeing them this way.)

Top left is true negative. 61% of values sit here. Bottom right = true
positive. 37%. lower left = false negative. 2% predicted benign but
actually malignant. top right, false positive. 0%

67+37+2 = 100%

### step 6 improvments

let’s z-score standardize and see if that helps.

``` r
#standarization
wbcd_z <- as.data.frame(scale(wbcd[-1]))

#example summary stats.
summary(wbcd_z$area_mean)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -1.4532 -0.6666 -0.2949  0.0000  0.3632  5.2459

The mean of z-score standardization should always be zero, and here it
is. on the low side, nothing smaller than -3; on the high side we
definitely have outliers.

regardless, we follow the same process as before, just now with the
z-score normalized data.

``` r
wbcd_train <- wbcd_z[1:469, ]
wbcd_test <- wbcd_z[470:569, ]
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]
wbcd_test_pred <- 
    knn(train = wbcd_train, 
        test = wbcd_test,
        cl = wbcd_train_labels, k = 21)

CrossTable(x = wbcd_test_labels, y = wbcd_test_pred,
             prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        61 |         0 |        61 | 
    ##                  |     1.000 |     0.000 |     0.610 | 
    ##                  |     0.924 |     0.000 |           | 
    ##                  |     0.610 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         5 |        34 |        39 | 
    ##                  |     0.128 |     0.872 |     0.390 | 
    ##                  |     0.076 |     1.000 |           | 
    ##                  |     0.050 |     0.340 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        66 |        34 |       100 | 
    ##                  |     0.660 |     0.340 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

Now we have 61 + 34 = 95% classified correctly; so the z-scoring
normalization made things worse. having 5% misclassified as *false
negatives* is a big problem when it comes to tumors!

OK, but what if we change how many neighbors we’re using for sampling?
(PS: part of the reason i dont like this method *so far* is because we
dont have an visualizations showing the classification objects, just an
output table!)

they give us a for-loop code to run through different k-values.

``` r
k_values <- c(1, 5, 11, 15, 21, 27)

for (k_val in k_values) {
    wbcd_test_pred <- knn(train = wbcd_train,
                          test = wbcd_test,
                          cl = wbcd_train_labels,
                          k = k_val)
    CrossTable(x = wbcd_test_labels,
               y = wbcd_test_pred,
               prop.chisq = FALSE)
  }
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        59 |         2 |        61 | 
    ##                  |     0.967 |     0.033 |     0.610 | 
    ##                  |     0.952 |     0.053 |           | 
    ##                  |     0.590 |     0.020 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         3 |        36 |        39 | 
    ##                  |     0.077 |     0.923 |     0.390 | 
    ##                  |     0.048 |     0.947 |           | 
    ##                  |     0.030 |     0.360 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        62 |        38 |       100 | 
    ##                  |     0.620 |     0.380 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ##  
    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        60 |         1 |        61 | 
    ##                  |     0.984 |     0.016 |     0.610 | 
    ##                  |     0.968 |     0.026 |           | 
    ##                  |     0.600 |     0.010 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        37 |        39 | 
    ##                  |     0.051 |     0.949 |     0.390 | 
    ##                  |     0.032 |     0.974 |           | 
    ##                  |     0.020 |     0.370 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        62 |        38 |       100 | 
    ##                  |     0.620 |     0.380 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ##  
    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        60 |         1 |        61 | 
    ##                  |     0.984 |     0.016 |     0.610 | 
    ##                  |     0.952 |     0.027 |           | 
    ##                  |     0.600 |     0.010 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         3 |        36 |        39 | 
    ##                  |     0.077 |     0.923 |     0.390 | 
    ##                  |     0.048 |     0.973 |           | 
    ##                  |     0.030 |     0.360 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        63 |        37 |       100 | 
    ##                  |     0.630 |     0.370 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ##  
    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        61 |         0 |        61 | 
    ##                  |     1.000 |     0.000 |     0.610 | 
    ##                  |     0.953 |     0.000 |           | 
    ##                  |     0.610 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         3 |        36 |        39 | 
    ##                  |     0.077 |     0.923 |     0.390 | 
    ##                  |     0.047 |     1.000 |           | 
    ##                  |     0.030 |     0.360 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        64 |        36 |       100 | 
    ##                  |     0.640 |     0.360 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ##  
    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        61 |         0 |        61 | 
    ##                  |     1.000 |     0.000 |     0.610 | 
    ##                  |     0.924 |     0.000 |           | 
    ##                  |     0.610 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         5 |        34 |        39 | 
    ##                  |     0.128 |     0.872 |     0.390 | 
    ##                  |     0.076 |     1.000 |           | 
    ##                  |     0.050 |     0.340 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        66 |        34 |       100 | 
    ##                  |     0.660 |     0.340 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ##  
    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        61 |         0 |        61 | 
    ##                  |     1.000 |     0.000 |     0.610 | 
    ##                  |     0.924 |     0.000 |           | 
    ##                  |     0.610 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         5 |        34 |        39 | 
    ##                  |     0.128 |     0.872 |     0.390 | 
    ##                  |     0.076 |     1.000 |           | 
    ##                  |     0.050 |     0.340 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        66 |        34 |       100 | 
    ##                  |     0.660 |     0.340 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

## Tidy styling

Ohh, right, `mutate_if` let’s me use the normalize function against the
data. But ideally these would be *new* columns just in case we want to
look at the original data.

the tidy way to do this would be to retain the ID column, then normalize
as a mutate, then `slice_head` to get the same rows.

``` r
cancer_tidy <- 
    wbcd_tidy %>% 
    mutate_if(is.double, normalize) 

cancer_tidy_train <- 
    cancer_tidy %>% 
    slice_head(n = 469)

cancer_tidy_test <- 
    cancer_tidy %>% 
    anti_join(cancer_tidy_train, by = 'id')
```

I like the tidy way better because it maintains the other variables and
we dont need to rejoin to get the diagnosis info. But I digress, let’s
make data frames of the diagnosis column data…

``` r
cancer_predict <- knn(train = cancer_tidy_train %>% select(-c(1:2)),  #469 rows
    test = cancer_tidy_test %>% select(-c(1:2)),
    cl = cancer_tidy_train$diagnosis, #469 rows
    k = 21)
```

At the end of the chapter, we have implication that we’re going to be
doing things a bit differently later on, so I’ll save my tidy
translations until I see otherwise. Maybe I’ll just get another book of
the same topic that does it all Tidy.
