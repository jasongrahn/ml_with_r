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

    ## Rows: 5559 Columns: 2
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (2): type, text
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

## 2 exploration & prep

    ## spc_tbl_ [5,559 × 2] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
    ##  $ type: chr [1:5559] "ham" "ham" "ham" "spam" ...
    ##  $ text: chr [1:5559] "Hope you are having a good week. Just checking in" "K..give back my thanks." "Am also doing in cbe only. But have to pay." "complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. 09066364349 NOW from Landline "| __truncated__ ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   type = col_character(),
    ##   ..   text = col_character()
    ##   .. )
    ##  - attr(*, "problems")=<externalptr>

    ## Rows: 5,559
    ## Columns: 2
    ## $ type <chr> "ham", "ham", "ham", "spam", "spam", "ham", "ham", "ham", "spam",…
    ## $ text <chr> "Hope you are having a good week. Just checking in", "K..give bac…

    ## spc_tbl_ [5,559 × 2] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
    ##  $ type: Factor w/ 2 levels "ham","spam": 1 1 1 2 2 1 1 1 2 1 ...
    ##  $ text: chr [1:5559] "Hope you are having a good week. Just checking in" "K..give back my thanks." "Am also doing in cbe only. But have to pay." "complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. 09066364349 NOW from Landline "| __truncated__ ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   type = col_character(),
    ##   ..   text = col_character()
    ##   .. )
    ##  - attr(*, "problems")=<externalptr>

    ## 
    ##  ham spam 
    ## 4812  747

We use the `tm` package to text mining.

    ## Loading required package: NLP

We’re going to create a corpus. A corpus is a collection of text
documents. The text “documents” can be of any length. Use `VCorpus()`
for this, the “v” meaning “volatile” (we’d used `PCorpus()` to access a
permanent corpus on a database).

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 5559

A corpus is a list, so we can use list tools to pick documents out of
it.

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

    ## [1] "Hope you are having a good week. Just checking in"

ok so let’s look at multiple documents…

    ## $`1`
    ## [1] "Hope you are having a good week. Just checking in"
    ## 
    ## $`2`
    ## [1] "K..give back my thanks."

Lots of problems with text strings… Punctuation and character cases are
two main ones. So let’s try to clean this up a bit.

    ## [1] FALSE

    ## [1] "hope you are having a good week. just checking in"

Notice that everything is lower case now. Next we remove numbers (and
these iterative steps are things that seem better wrapped in `mutate()`)

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

… Note that removing punctuation can change a `hello...world` into
`helloworld`

We need to install the `SnowballC` package to do stemming sooo that’s
being done on the side.. Stemming strips `ed`, `ing`, and plurals from
words so we can count the root words.

What does this end up looking like?

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

At least some of that prep we did earlier could’ve been done here too.
When we do it this way, we notice the mix of cases in the `tm` package.
(there’s probably a better way out there to do this that has more
attention to detail…).

Though, doing it this way provides slightly different results.

    ## <<DocumentTermMatrix (documents: 5559, terms: 6542)>>
    ## Non-/sparse entries: 42113/36324865
    ## Sparsity           : 100%
    ## Maximal term length: 40
    ## Weighting          : term frequency (tf)

    ## <<DocumentTermMatrix (documents: 5559, terms: 6940)>>
    ## Non-/sparse entries: 43186/38536274
    ## Sparsity           : 100%
    ## Maximal term length: 40
    ## Weighting          : term frequency (tf)

(It would be nice if the results output was a table so we could compare
the output values against each other.)

## 3. training & test sets

we’ll split 75% training, 25% testing. data is already randomized, so
we’re just going to take a lop off the top for training.

    ## sms_train_labels
    ##       ham      spam 
    ## 0.8647158 0.1352842

    ## sms_test_labels
    ##       ham      spam 
    ## 0.8683453 0.1316547

## 4. Visualizing with word clouds

and of course we need a new `wordcloud` package. i’m installing that
through the console.

The idea here is that different words should **POP** between the spam &
ham data.

    ## Loading required package: RColorBrewer

![](chapter_4_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

Since this shows *all* the messages, we need to force a subset then
cloud again (again… this method doesn’t scale!)

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation): transformation
    ## drops documents

    ## Warning in tm_map.SimpleCorpus(corpus, function(x) tm::removeWords(x,
    ## tm::stopwords())): transformation drops documents

![](chapter_4_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation): transformation
    ## drops documents

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation): transformation
    ## drops documents

![](chapter_4_files/figure-gfm/unnamed-chunk-24-2.png)<!-- -->

## 5. creating indicator features for frequent words

we need to reduce the number of “features” (columns) because that sparse
matrix sucks. we’ll get ride of any word that appears in less than 5
messages. We save that as it’s own vector.

    ##  chr [1:1137] "£wk" "abiola" "abl" "abt" "accept" "access" "account" ...

Let’s apply this vector as a filter to our training data!

*now* we convert the numeric features (the word counts) into categorical
features (`Yes` or `No` strings, which would probably be better as
TRUE/FALSE strings instead). Over in Tidyland, we’d probably get away
with a mutate_all that does this through a case statement?

## 6. training the model

we use the `naivebayes` package which I also installed through the
console.

    ## naivebayes 0.9.7 loaded

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
