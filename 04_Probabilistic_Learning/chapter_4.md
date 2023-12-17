chapter_4\_probabilistic_classifiers_with_bayes
================
jason grahn
2023-11-21

“Bayes” = principles for describing the probability of events and how
those probabilities are revised with additional information.

Refresher:

- probability measured as a number between 0 and 1 (0 to 100%)
- low = not likely to occur, high = likely to occur.

Bayes Classifiers use training data to find probability of outcomes
based on featurd-based evidence. Classifier then applied to unlabeled
data to predict most likley class from the new example.

Use these when info from numorous attributes should be considered at the
same time, to extimate overall probability of an outcome.

Most ML Algos ignore weak effect features, Bayes uses them all and
adjusted subtly. Lots of small features could have a combined impact!

Bayes in a nutshell: likelihood of an event (outcome) should be based on
evidence across many opportunities for the events to occur.

Examples: \* heads / tails \* rainy weather \* winning the lottery \*
spam text

**Probability**

$\frac{successes}{attempts}$

denoted as

The Probability that “A” event occurred: $P(A)$ Probability that rain
occured? $P(rain)$

The inverse is probably that rain did not occur: $1 - P(rain)$ aka the
complement $P(A^c)$ (Probability of the compliment of event A)

**Joint Probability**

How the probabilty of one event is related to the probability of the
other. (and we use the upside-down “U” symbol to denote these
intersections.)

When “Spam” and “Viagra” occur, then the *intersection* between
$P(spam) \cap P(viagra)$ is a joint probability.

Quote from the text: **dependent events** are the basis of predictive
modeling. Just as the presence of clouds is predictive of a rainy day,
the appearance of the word Viagra is predictive of a spam email.

$P(A \cap B) = P(A)*P(B)$

# Computing conditional probability with Bayes’ theorem

\$P(A\|B) = \$

The probability of *Event A* happened given *Event B* also happened.

Bayes says: The best estimate of $P(A|B)$ is the proportion of trials
where A occurred with B out of *all* events where B occurred.

which we can flip to
$P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B|A)*P(A)}{P(B)}$ using the
transitive property of multiplication!

“Prior probability” is the known. In the case of the spam filter and
viagra supplements, its easier to measure the $P(spam)$.

Then we can do *posterior probability* to measure how likely a message
is to be spam. Greater than 50%? more likely spam!

$P(spam|viagra) = \frac{P(viagra|spam) * P(spam)} {P(viagra)}$

P(spam\|viagra) = posterior probability P(viagra\|spam) = likelihood
P(spam) = prior probability P(viagra) = marginal likelihood

So… $posterior = \frac{likelihood * prior}{marginal}$

# the Naive Bayes algo

strengths \* simple, fast, effective \* works well with noise, missing
data, & lots of features \* few examples needed to train \* easy to get
estimated probability

weaknesses \* often-faulty assumptions of equal importance and
independence (rarely true) \* not great with numeric features \*
estimates not as reliable as predicated classes

## Classification with Naive Bayes

**class-conditional independence** = events are independent so long as
they are conditioned on the same class value. Allows us to multiply
individual conditional probabilities instead of looking at joint
probability.

“The probability of spam is equal to the likelihood that the message is
spam divided by the likelihood that the message is *either* spam or
ham.” 0.857

The probability of ham is equal to the likelihood that the message is
ham divided by the likelihood that the message is eitehr. 0.143

Build a frequency table. build a likelihood table. multiply the
conditional probabilities with the naive assumption of independence.
divide by the total likelihood – transforms each class likelihood into a
probability.

## The LaPlace estimator

when we’ve never seen something before, it’s prior is zero. So when we
do a ‘normal’ Bayesian classifier multiplying the conditional
probabilities, zero times zero = zero. The **Laplace estimator** adds a
small number to each of the counts, which ensures that each feature has
a non-zero probability of occuring. Typically it’s set to 1 so all
classes-features have at least *1*. (later notes: 1 is almost always
used.)

You add the laplace estimator in both the numerator and denominator of
each likelihood.

## Using numeric features

Because naive bayes users frequency tables, each feature *must* be
categorical. Numeric features don’t have categories!

- make the numeric features discrete; that is, put them into bins (aka:
  **Binning**). ++ Bin these according to logical conclusions about the
  data. ++ a clock has 24 discrete hours, or maybe 4 bins of time per
  day “early morning” “late morning” “afternoon” evening”. ++ use a
  histogram!
- if you can’t figure our how to bin, just pop ’em into quantiles.

# example, filtering spam texts

## 1, collecting data

``` r
sms_raw <- read_csv("https://raw.githubusercontent.com/PacktPublishing/Machine-Learning-with-R-Fourth-Edition/main/Chapter%2004/sms_spam.csv")
```

    ## Rows: 5559 Columns: 2
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (2): type, text
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# the file is also in the /data folder
```

## 2 exploration & prep

``` r
str(sms_raw)
```

    ## spc_tbl_ [5,559 × 2] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
    ##  $ type: chr [1:5559] "ham" "ham" "ham" "spam" ...
    ##  $ text: chr [1:5559] "Hope you are having a good week. Just checking in" "K..give back my thanks." "Am also doing in cbe only. But have to pay." "complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. 09066364349 NOW from Landline "| __truncated__ ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   type = col_character(),
    ##   ..   text = col_character()
    ##   .. )
    ##  - attr(*, "problems")=<externalptr>

``` r
dplyr::glimpse(sms_raw)
```

    ## Rows: 5,559
    ## Columns: 2
    ## $ type <chr> "ham", "ham", "ham", "spam", "spam", "ham", "ham", "ham", "spam",…
    ## $ text <chr> "Hope you are having a good week. Just checking in", "K..give bac…

``` r
sms_raw$type <- factor(sms_raw$type)

str(sms_raw)
```

    ## spc_tbl_ [5,559 × 2] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
    ##  $ type: Factor w/ 2 levels "ham","spam": 1 1 1 2 2 1 1 1 2 1 ...
    ##  $ text: chr [1:5559] "Hope you are having a good week. Just checking in" "K..give back my thanks." "Am also doing in cbe only. But have to pay." "complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. 09066364349 NOW from Landline "| __truncated__ ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   type = col_character(),
    ##   ..   text = col_character()
    ##   .. )
    ##  - attr(*, "problems")=<externalptr>

``` r
table(sms_raw$type)
```

    ## 
    ##  ham spam 
    ## 4812  747

We use the `tm` package to text mining.

``` r
library(tm)
```

    ## Loading required package: NLP

We’re going to create a corpus. A corpus is a collection of text
documents. The text “documents” can be of any length. Use `VCorpus()`
for this, the “v” meaning “volatile” (we’d used `PCorpus()` to access a
permanent corpus on a database).

``` r
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

sms_corpus
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 5559

A corpus is a list, so we can use list tools to pick documents out of
it.

``` r
inspect(sms_corpus[1:2])
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 2
    ## 
    ## [[1]]
    ## <<PlainTextDocument>>
    ## Metadata:  7
    ## Content:  chars: 49
    ## 
    ## [[2]]
    ## <<PlainTextDocument>>
    ## Metadata:  7
    ## Content:  chars: 23

And to see what’s *in* those..

``` r
as.character(sms_corpus[[1]])
```

    ## [1] "Hope you are having a good week. Just checking in"

ok so let’s look at multiple documents…

``` r
lapply(sms_corpus[1:2], as.character)
```

    ## $`1`
    ## [1] "Hope you are having a good week. Just checking in"
    ## 
    ## $`2`
    ## [1] "K..give back my thanks."

Lots of problems with text strings… Punctuation and character cases are
two main ones. So let’s try to clean this up a bit.

``` r
sms_corpus_clean <- 
    tm_map(sms_corpus,
           content_transformer(tolower))

# did it work? 
as.character(sms_corpus[[1]]) == as.character(sms_corpus_clean[[1]])
```

    ## [1] FALSE

``` r
as.character(sms_corpus_clean[[1]])
```

    ## [1] "hope you are having a good week. just checking in"

Notice that everything is lower case now. Next we remove numbers (and
these iterative steps are things that seem better wrapped in `mutate()`)

``` r
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

lapply(sms_corpus[1:5], as.character)
```

    ## $`1`
    ## [1] "Hope you are having a good week. Just checking in"
    ## 
    ## $`2`
    ## [1] "K..give back my thanks."
    ## 
    ## $`3`
    ## [1] "Am also doing in cbe only. But have to pay."
    ## 
    ## $`4`
    ## [1] "complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. 09066364349 NOW from Landline not to lose out! Box434SK38WP150PPM18+"
    ## 
    ## $`5`
    ## [1] "okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award! Call 09061743806 from landline. TCs SAE Box326 CW25WX 150ppm"

``` r
lapply(sms_corpus_clean[1:5], as.character)
```

    ## $`1`
    ## [1] "hope you are having a good week. just checking in"
    ## 
    ## $`2`
    ## [1] "k..give back my thanks."
    ## 
    ## $`3`
    ## [1] "am also doing in cbe only. but have to pay."
    ## 
    ## $`4`
    ## [1] "complimentary  star ibiza holiday or £, cash needs your urgent collection.  now from landline not to lose out! boxskwpppm+"
    ## 
    ## $`5`
    ## [1] "okmail: dear dave this is your final notice to collect your * tenerife holiday or # cash award! call  from landline. tcs sae box cwwx ppm"

Then we remove stop words, and may as well include puntuation next..

``` r
sms_corpus_clean <-
    tm_map(sms_corpus_clean,
           removeWords, 
           stopwords())

sms_corpus_clean <-
    tm_map(sms_corpus_clean,
           removePunctuation)
```

… Note that removing punctuation can change a `hello...world` into
`helloworld`

We need to install the `SnowballC` package to do stemming sooo that’s
being done on the side.. Stemming strips `ed`, `ing`, and plurals from
words so we can count the root words.

``` r
sms_corpus_clean <-
    tm_map(sms_corpus_clean, stemDocument)

# and lets strip whitespaces too..

sms_corpus_clean <- 
    tm_map(sms_corpus_clean, stripWhitespace)
```

What does this end up looking like?

``` r
lapply(sms_corpus[1:5], as.character)
```

    ## $`1`
    ## [1] "Hope you are having a good week. Just checking in"
    ## 
    ## $`2`
    ## [1] "K..give back my thanks."
    ## 
    ## $`3`
    ## [1] "Am also doing in cbe only. But have to pay."
    ## 
    ## $`4`
    ## [1] "complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. 09066364349 NOW from Landline not to lose out! Box434SK38WP150PPM18+"
    ## 
    ## $`5`
    ## [1] "okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award! Call 09061743806 from landline. TCs SAE Box326 CW25WX 150ppm"

``` r
lapply(sms_corpus_clean[1:5], as.character)
```

    ## $`1`
    ## [1] "hope good week just check"
    ## 
    ## $`2`
    ## [1] "kgive back thank"
    ## 
    ## $`3`
    ## [1] "also cbe pay"
    ## 
    ## $`4`
    ## [1] "complimentari star ibiza holiday £ cash need urgent collect now landlin lose boxskwpppm"
    ## 
    ## $`5`
    ## [1] "okmail dear dave final notic collect tenerif holiday cash award call landlin tcs sae box cwwx ppm"

### data prep - text docs to words

**tokenization**: a single element of a text string, words.

so we split the ‘document’ (text strings) into their individual words,
then count how many times those words occur in each document. This
produces with a massively wide data, as we attempt to count *each* word
of the entire corpus for each document. We end with a **sparce matrix**
where most of the counts are zero.

We use the default settings of `DocumentTermMatrix()`.

``` r
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
```

At least some of that prep we did earlier could’ve been done here too.
When we do it this way, we notice the mix of cases in the `tm` package.
(there’s probably a better way out there to do this that has more
attention to detail…).

``` r
sms_dtm2 <- 
    DocumentTermMatrix(
        sms_corpus,
        control = list(
            tolower = TRUE,
            removeNumbers = TRUE,
            stopwords = TRUE,
            removePunctuation = TRUE,
            stemming = TRUE))
```

Though, doing it this way provides slightly different results.

``` r
sms_dtm
```

    ## <<DocumentTermMatrix (documents: 5559, terms: 6542)>>
    ## Non-/sparse entries: 42113/36324865
    ## Sparsity           : 100%
    ## Maximal term length: 40
    ## Weighting          : term frequency (tf)

``` r
sms_dtm2
```

    ## <<DocumentTermMatrix (documents: 5559, terms: 6940)>>
    ## Non-/sparse entries: 43186/38536274
    ## Sparsity           : 100%
    ## Maximal term length: 40
    ## Weighting          : term frequency (tf)

``` r
# lets get rid of this because we aren't going to use it.
rm(sms_dtm2)
```

(It would be nice if the results output was a table so we could compare
the output values against each other.)

## 3. training & test sets

we’ll split 75% training, 25% testing. data is already randomized, so
we’re just going to take a lop off the top for training.

``` r
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

# and save a vector of their labels. 
# this is not how i like to do this kind of thing, because it relies on manual input, 
# which is prone to mistypes and errors! 

sms_train_labels <- sms_raw[1:4169,]$type
sms_test_labels <- sms_raw[4170:5559,]$type
```

``` r
prop.table(table(sms_train_labels))
```

    ## sms_train_labels
    ##       ham      spam 
    ## 0.8647158 0.1352842

``` r
# tibble(sms_train_labels) %>%
#     count(sms_train_labels) %>% 
#     mutate(freq = n / sum(n))
```

``` r
prop.table(table(sms_test_labels))
```

    ## sms_test_labels
    ##       ham      spam 
    ## 0.8683453 0.1316547

## 4. Visualizing with word clouds

and of course we need a new `wordcloud` package. i’m installing that
through the console.

The idea here is that different words should **POP** between the spam &
ham data.

``` r
library(wordcloud)
```

    ## Loading required package: RColorBrewer

``` r
wordcloud(sms_corpus_clean, 
          min.freq = 50, 
          random.order = FALSE)
```

![](chapter_4_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

Since this shows *all* the messages, we need to force a subset then
cloud again (again… this method doesn’t scale!)

``` r
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")
```

``` r
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
```

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation): transformation
    ## drops documents

    ## Warning in tm_map.SimpleCorpus(corpus, function(x) tm::removeWords(x,
    ## tm::stopwords())): transformation drops documents

![](chapter_4_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

``` r
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
```

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation): transformation
    ## drops documents

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation): transformation
    ## drops documents

![](chapter_4_files/figure-gfm/unnamed-chunk-24-2.png)<!-- -->

## 5. creating indicator features for frequent words

we need to reduce the number of “features” (columns) because that sparse
matrix sucks. we’ll get ride of any word that appears in less than 5
messages. We save that as it’s own vector.

``` r
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

str(sms_freq_words) 
```

    ##  chr [1:1137] "£wk" "abiola" "abl" "abt" "accept" "access" "account" ...

``` r
## 1137 different terms make the cut!
```

Let’s apply this vector as a filter to our training data!

``` r
sms_dtm_freq_train <- sms_dtm_train[ ,sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ ,sms_freq_words]
```

*now* we convert the numeric features (the word counts) into categorical
features (`Yes` or `No` strings, which would probably be better as
TRUE/FALSE strings instead). Over in Tidyland, we’d probably get away
with a mutate_all that does this through a case statement?

``` r
convert_counts <- function(x) {
    x <- ifelse(x > 0, "Yes", "No")
}
```

``` r
sms_train <- apply(sms_dtm_freq_train,
                   MARGIN = 2, # applies the function to the columns, (MARGIN = 1 applies the function to rows)
                   convert_counts)

sms_test <- apply(sms_dtm_freq_test,
                   MARGIN = 2,
                   convert_counts)
```

## 6. training the model

we use the `naivebayes` package which I also installed through the
console.

``` r
library(naivebayes)
```

    ## naivebayes 0.9.7 loaded

``` r
sms_classifier <- naive_bayes(sms_train, sms_train_labels)

# muting errors because the book says I dont need to worry about this for now. 
# the error come from using the defalt leplace estimator
```

``` r
sms_test_pred <- predict(sms_classifier, sms_test)
```

``` r
gmodels::CrossTable(sms_test_pred, 
                    sms_test_labels,
                    prop.chisq = FALSE,
                    prop.c = FALSE,
                    prop.r = FALSE,
                    dnn = c('predicted', 'actual'))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  1390 
    ## 
    ##  
    ##              | actual 
    ##    predicted |       ham |      spam | Row Total | 
    ## -------------|-----------|-----------|-----------|
    ##          ham |      1201 |        30 |      1231 | 
    ##              |     0.864 |     0.022 |           | 
    ## -------------|-----------|-----------|-----------|
    ##         spam |         6 |       153 |       159 | 
    ##              |     0.004 |     0.110 |           | 
    ## -------------|-----------|-----------|-----------|
    ## Column Total |      1207 |       183 |      1390 | 
    ## -------------|-----------|-----------|-----------|
    ## 
    ## 

## 7. improving the model

``` r
sms_classifier2 <- naive_bayes(sms_train,
                               sms_train_labels,
                               laplace = 1)

# see, no errors when we make laplace = 1
```

``` r
sms_test_pred2 <- predict(sms_classifier2, sms_test)
```

``` r
gmodels::CrossTable(sms_test_pred2, 
                    sms_test_labels,
                    prop.chisq = FALSE,
                    prop.c = FALSE,
                    prop.r = FALSE,
                    dnn = c('predicted', 'actual'))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  1390 
    ## 
    ##  
    ##              | actual 
    ##    predicted |       ham |      spam | Row Total | 
    ## -------------|-----------|-----------|-----------|
    ##          ham |      1202 |        28 |      1230 | 
    ##              |     0.865 |     0.020 |           | 
    ## -------------|-----------|-----------|-----------|
    ##         spam |         5 |       155 |       160 | 
    ##              |     0.004 |     0.112 |           | 
    ## -------------|-----------|-----------|-----------|
    ## Column Total |      1207 |       183 |      1390 | 
    ## -------------|-----------|-----------|-----------|
    ## 
    ## 

Notice that this reduced the number of false positives (ham messages
classified as spam) as well as false negatives (spam messages classfied
as ham). This model is 3 pp better than 0.
