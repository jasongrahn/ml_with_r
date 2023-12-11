chapter_5\_decision_tree_classification
================
jason grahn
2023-12-07

We’re gunna make a few decision trees.

- C5.0, 1R, and RIPPER (huh?) ((that’s why I’m doing this!))

decision tree terms:

- **root node**: where the tree starts
- **decision node**: choices to be made based on attributes of the job
- **branch**: indicators of potential outcomes of a decision
- **leaf node / terminal node**: terminators that denote the action
  taken after the decision.

These work like flow-charts and should output a structure that humans
can consume. If someone hands you a decision tree model and doesn’t give
you the structure, they’re lying to you. Important for legal
transparency, or if the results might inform practices.

very widely used.

more terms:

- **recursive partitioning** (aka **divide and conquer**): a heuristic
  that splits data into subsets repeatedly until the algo decides
  homogeneity.

This would’ve been great to use on the renewals project. Probably
would’ve lifted it higher than I did and taken less time to do it.
Coulda used Frank’s dataset for price sensitivity.

“overly specific decisions do not always generalize more broadly.” -
Brett Lantz

- **axis-parallel splits**: each data split or decision considers one
  feature at a time. The split occurs parallel to one of the axis
  (visualize a scatterplot).

side-note, I should start playing with Quarto. I’ll start that in the
next chapter.

# C5.0

the industry standard for decision trees. Does well out-of-box. easy to
understand. easy to deploy.

## choosing a split

- **purity**: degree to which a subset of examples contains a single
  class
- **pure**: any subset of data composed of only a single class
- **entropy**: quantified randomness within a set of class values

A decision tree hopes to find splits that reduce entropy.

- **bits**: measurement of entropy. closer to 0 means closer to
  homogeneity, higher values indicate diversity.

an entropy curve:

``` r
curve(-x * log2(x) - (1 - x) * log2(1 - x), # the entropy algo for two class outcomes 
      col = "red", # just giving a spash of color
      xlab = "x", # titling an axis
      ylab = "Entropy", # titling an axis
      lwd = 1) # line width
```

![](chapter_5_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

Note that as one class dominates the other, entropy scales to zero.

- **information gain**: when the algorithm calculates changes in
  homogeneity that results from splits on features. Or: “the reduction
  in entropy or surprise by transforming a dataset and is often used in
  training decision trees. Information gain is calculated by comparing
  the entropy of the dataset before and after a transformation” (per
  <https://machinelearningmastery.com/information-gain-and-mutual-information/>)
  ++ $InfoGain(F) = Entropy(S_1) - Entryopy(S_2)$ ++ The higher the
  information gain, the better a feature is at creating groups.

After a split, the data divides into multiple partitions, so $(S_2)$
must consider entropy across all partitions resulting from the split.

Trees can split on numeric variables too. Gotta test split results that
divide into groups though (binning that makes sense for the given
challenge) - effectively turning the numeric variables into categorical
data.

## Pruning

- **pruning**: reducing the size of the decision tree to better
  generalize unseen data.
- **early stopping**: stopping tree growth once it has reached a set
  number of decisions. AKA *pre-pruning*
- **post-pruning**: building a tree “too large” then cutting leaf nodes
  to reduce to more appropriate level. often more effective because it
  helps illustrate how large or deep the tree will go. ++C5.0 overfits,
  then prunes nodes and branches that have little effect or moves them.
- **subtree raising** or **subtree replacement**: the act of moving
  branches to earlier in the tree

## Example: finding risky bank loans

### data ingestion from online csv

    ## 'data.frame':    1000 obs. of  17 variables:
    ##  $ checking_balance    : Factor w/ 4 levels "< 0 DM","> 200 DM",..: 1 3 4 1 1 4 4 3 4 3 ...
    ##  $ months_loan_duration: int  6 48 12 42 24 36 24 36 12 30 ...
    ##  $ credit_history      : Factor w/ 5 levels "critical","good",..: 1 2 1 2 4 2 2 2 2 1 ...
    ##  $ purpose             : Factor w/ 6 levels "business","car",..: 5 5 4 5 2 4 5 2 5 2 ...
    ##  $ amount              : int  1169 5951 2096 7882 4870 9055 2835 6948 3059 5234 ...
    ##  $ savings_balance     : Factor w/ 5 levels "< 100 DM","> 1000 DM",..: 5 1 1 1 1 5 4 1 2 1 ...
    ##  $ employment_duration : Factor w/ 5 levels "< 1 year","> 7 years",..: 2 3 4 4 3 3 2 3 4 5 ...
    ##  $ percent_of_income   : int  4 2 2 2 3 2 3 2 2 4 ...
    ##  $ years_at_residence  : int  4 2 3 4 4 4 4 2 4 2 ...
    ##  $ age                 : int  67 22 49 45 53 35 53 35 61 28 ...
    ##  $ other_credit        : Factor w/ 3 levels "bank","none",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ housing             : Factor w/ 3 levels "other","own",..: 2 2 2 1 1 1 2 3 2 2 ...
    ##  $ existing_loans_count: int  2 1 1 1 2 1 1 1 1 2 ...
    ##  $ job                 : Factor w/ 4 levels "management","skilled",..: 2 2 4 2 2 4 2 1 4 1 ...
    ##  $ dependents          : int  1 1 2 2 2 2 1 1 1 1 ...
    ##  $ phone               : Factor w/ 2 levels "no","yes": 2 1 1 1 1 2 1 2 1 1 ...
    ##  $ default             : Factor w/ 2 levels "no","yes": 1 2 1 1 2 1 1 1 1 2 ...

    ## 
    ##     < 0 DM   > 200 DM 1 - 200 DM    unknown 
    ##        274         63        269        394

    ## 
    ##      < 100 DM     > 1000 DM  100 - 500 DM 500 - 1000 DM       unknown 
    ##           603            48           103            63           183

These values use Deutschemark as the currency. Would probably be a good
idea to order those factors!

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     4.0    12.0    18.0    20.9    24.0    72.0

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     250    1366    2320    3271    3972   18424

    ## 
    ##  no yes 
    ## 700 300

### data prep

creating random training & test data

using 90% for training, 10% for test because the small size of the data
(1000 observations) using the `random()` function. So we need to set a
seed value using `set.seed()`. The book provides a seed of **9829** so
we’ll use that too.

ah, this is interesting. They’re making a dataset of the rows we’re
going to use for the training set; then use *that* as the filter for the
credit data.

yep.

    ## 
    ##        no       yes 
    ## 0.7055556 0.2944444

    ## 
    ##   no  yes 
    ## 0.65 0.35

### training

We’ll build a model to predict the `default` variable using *formula
syntax*, then use the model to predict.

    ## 
    ## Call:
    ## C5.0.formula(formula = default ~ ., data = credit_train)
    ## 
    ## Classification Tree
    ## Number of samples: 900 
    ## Number of predictors: 16 
    ## 
    ## Tree size: 67 
    ## 
    ## Non-standard options: attempt to group attributes

We used 16 predictors to determine default, and the tree is 67 decisions
deep.

I bet there’s a different package out there that develops better
visualizations for this.

    ## 
    ## Call:
    ## C5.0.formula(formula = default ~ ., data = credit_train)
    ## 
    ## 
    ## C5.0 [Release 2.07 GPL Edition]      Mon Dec 11 15:07:50 2023
    ## -------------------------------
    ## 
    ## Class specified by attribute `outcome'
    ## 
    ## Read 900 cases (17 attributes) from undefined.data
    ## 
    ## Decision tree:
    ## 
    ## checking_balance in {> 200 DM,unknown}: no (415/55)
    ## checking_balance in {< 0 DM,1 - 200 DM}:
    ## :...credit_history in {perfect,very good}: yes (59/16)
    ##     credit_history in {critical,good,poor}:
    ##     :...months_loan_duration > 27:
    ##         :...dependents > 1:
    ##         :   :...age <= 45: no (12/2)
    ##         :   :   age > 45: yes (2)
    ##         :   dependents <= 1:
    ##         :   :...savings_balance = > 1000 DM: no (2/1)
    ##         :       savings_balance = 500 - 1000 DM: yes (1)
    ##         :       savings_balance = 100 - 500 DM:
    ##         :       :...credit_history = critical: no (1)
    ##         :       :   credit_history = good: yes (7)
    ##         :       :   credit_history = poor:
    ##         :       :   :...existing_loans_count <= 1: no (3)
    ##         :       :       existing_loans_count > 1: yes (3/1)
    ##         :       savings_balance = unknown:
    ##         :       :...checking_balance = 1 - 200 DM: no (8/1)
    ##         :       :   checking_balance = < 0 DM:
    ##         :       :   :...credit_history = critical: no (1)
    ##         :       :       credit_history in {good,poor}: yes (4)
    ##         :       savings_balance = < 100 DM:
    ##         :       :...job in {skilled,unskilled}: yes (43/9)
    ##         :           job = unemployed: no (1)
    ##         :           job = management:
    ##         :           :...existing_loans_count > 1: yes (4)
    ##         :               existing_loans_count <= 1:
    ##         :               :...amount <= 7582: no (5)
    ##         :                   amount > 7582:
    ##         :                   :...purpose in {business,car,education,
    ##         :                       :           furniture/appliances,
    ##         :                       :           renovations}: yes (4)
    ##         :                       purpose = car0: no (1)
    ##         months_loan_duration <= 27:
    ##         :...months_loan_duration <= 11:
    ##             :...job in {management,unemployed}:
    ##             :   :...percent_of_income <= 1: yes (3)
    ##             :   :   percent_of_income > 1:
    ##             :   :   :...age <= 34: yes (2)
    ##             :   :       age > 34: no (7/1)
    ##             :   job in {skilled,unskilled}:
    ##             :   :...age > 24: no (52/2)
    ##             :       age <= 24:
    ##             :       :...years_at_residence <= 1: no (3)
    ##             :           years_at_residence > 1:
    ##             :           :...job = skilled: yes (4)
    ##             :               job = unskilled: no (1)
    ##             months_loan_duration > 11:
    ##             :...credit_history = poor:
    ##                 :...housing = other: yes (2)
    ##                 :   housing in {own,rent}: no (20/4)
    ##                 credit_history = critical:
    ##                 :...purpose in {business,education}: no (10/1)
    ##                 :   purpose in {car0,renovations}: yes (2)
    ##                 :   purpose = car:
    ##                 :   :...other_credit in {none,store}: no (18/3)
    ##                 :   :   other_credit = bank:
    ##                 :   :   :...job in {management,skilled,unemployed}: yes (5)
    ##                 :   :       job = unskilled: no (2)
    ##                 :   purpose = furniture/appliances:
    ##                 :   :...phone = yes: no (11)
    ##                 :       phone = no:
    ##                 :       :...savings_balance in {> 1000 DM,
    ##                 :           :                   unknown}: no (0)
    ##                 :           savings_balance in {100 - 500 DM,
    ##                 :           :                   500 - 1000 DM}: yes (2)
    ##                 :           savings_balance = < 100 DM:
    ##                 :           :...age <= 29: no (8)
    ##                 :               age > 29: yes (4/1)
    ##                 credit_history = good:
    ##                 :...purpose in {car0,renovations}: no (7/2)
    ##                     purpose = business:
    ##                     :...dependents <= 1: no (8/1)
    ##                     :   dependents > 1: yes (3/1)
    ##                     purpose = education:
    ##                     :...savings_balance = < 100 DM: yes (3)
    ##                     :   savings_balance in {> 1000 DM,100 - 500 DM,
    ##                     :                       500 - 1000 DM,unknown}: no (3)
    ##                     purpose = car:
    ##                     :...amount <= 1391:
    ##                     :   :...savings_balance in {< 100 DM,100 - 500 DM,
    ##                     :   :   :                   500 - 1000 DM,
    ##                     :   :   :                   unknown}: yes (20/2)
    ##                     :   :   savings_balance = > 1000 DM: no (2)
    ##                     :   amount > 1391:
    ##                     :   :...amount <= 9629: no (30/8)
    ##                     :       amount > 9629: yes (3)
    ##                     purpose = furniture/appliances:
    ##                     :...savings_balance in {> 1000 DM,
    ##                         :                   500 - 1000 DM}: no (7/1)
    ##                         savings_balance = 100 - 500 DM:
    ##                         :...checking_balance = < 0 DM: yes (4)
    ##                         :   checking_balance = 1 - 200 DM:
    ##                         :   :...age <= 28: yes (2)
    ##                         :       age > 28: no (2)
    ##                         savings_balance = unknown:
    ##                         :...job = management: yes (1)
    ##                         :   job in {unemployed,unskilled}: no (3)
    ##                         :   job = skilled:
    ##                         :   :...age <= 28: yes (6/1)
    ##                         :       age > 28: no (4)
    ##                         savings_balance = < 100 DM:
    ##                         :...employment_duration = 4 - 7 years: no (5)
    ##                             employment_duration = > 7 years:
    ##                             :...job = management: yes (2)
    ##                             :   job in {skilled,unemployed,
    ##                             :           unskilled}: no (7/1)
    ##                             employment_duration = unemployed:
    ##                             :...housing = other: no (1)
    ##                             :   housing in {own,rent}: yes (3)
    ##                             employment_duration = < 1 year:
    ##                             :...checking_balance = < 0 DM: no (9/1)
    ##                             :   checking_balance = 1 - 200 DM:
    ##                             :   :...job in {management,skilled,
    ##                             :       :       unemployed}: yes (3)
    ##                             :       job = unskilled: no (1)
    ##                             employment_duration = 1 - 4 years:
    ##                             :...months_loan_duration <= 15: no (13/2)
    ##                                 months_loan_duration > 15:
    ##                                 :...checking_balance = 1 - 200 DM: no (2)
    ##                                     checking_balance = < 0 DM:
    ##                                     :...months_loan_duration <= 22: yes (8)
    ##                                         months_loan_duration > 22: no (6/1)
    ## 
    ## 
    ## Evaluation on training data (900 cases):
    ## 
    ##      Decision Tree   
    ##    ----------------  
    ##    Size      Errors  
    ## 
    ##      66  118(13.1%)   <<
    ## 
    ## 
    ##     (a)   (b)    <-classified as
    ##    ----  ----
    ##     604    31    (a): class no
    ##      87   178    (b): class yes
    ## 
    ## 
    ##  Attribute usage:
    ## 
    ##  100.00% checking_balance
    ##   53.89% credit_history
    ##   47.33% months_loan_duration
    ##   26.11% purpose
    ##   24.33% savings_balance
    ##   18.22% job
    ##   12.56% dependents
    ##   12.11% age
    ##    7.22% amount
    ##    6.67% employment_duration
    ##    2.89% housing
    ##    2.78% other_credit
    ##    2.78% phone
    ##    2.22% existing_loans_count
    ##    1.33% percent_of_income
    ##    0.89% years_at_residence
    ## 
    ## 
    ## Time: 0.0 secs

Now we predict.

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                | predicted default 
    ## actual default |        no |       yes | Row Total | 
    ## ---------------|-----------|-----------|-----------|
    ##             no |        56 |         9 |        65 | 
    ##                |     0.560 |     0.090 |           | 
    ## ---------------|-----------|-----------|-----------|
    ##            yes |        24 |        11 |        35 | 
    ##                |     0.240 |     0.110 |           | 
    ## ---------------|-----------|-----------|-----------|
    ##   Column Total |        80 |        20 |       100 | 
    ## ---------------|-----------|-----------|-----------|
    ## 
    ## 

24% predicted `not default` when in actuality, yes, they defaulted.
That’s not great and needs to be improved.

### improving results

We’re gunna use *boosting* to improve the model. Boosting seems to do a
bit of component combination.

    ## 
    ## Call:
    ## C5.0.formula(formula = default ~ ., data = credit_train, trials = 10)
    ## 
    ## Classification Tree
    ## Number of samples: 900 
    ## Number of predictors: 16 
    ## 
    ## Number of boosting iterations: 10 
    ## Average tree size: 57.3 
    ## 
    ## Non-standard options: attempt to group attributes

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                | predicted default 
    ## actual default |        no |       yes | Row Total | 
    ## ---------------|-----------|-----------|-----------|
    ##             no |        58 |         7 |        65 | 
    ##                |     0.580 |     0.070 |           | 
    ## ---------------|-----------|-----------|-----------|
    ##            yes |        19 |        16 |        35 | 
    ##                |     0.190 |     0.160 |           | 
    ## ---------------|-----------|-----------|-----------|
    ##   Column Total |        77 |        23 |       100 | 
    ## ---------------|-----------|-----------|-----------|
    ## 
    ## 

#### cost matrix

We build a cost matrix in order to penalize the decision tree for
errors, preventing mistakes.

    ## $predicted
    ## [1] "no"  "yes"
    ## 
    ## $actual
    ## [1] "no"  "yes"

Matrix built, now the costs of each. R goes top to bottom.

    ##          actual
    ## predicted no yes
    ##       no   0   4
    ##       yes  1   0

No penalties for guessing correctly, but a penalty of 1 for predicting
*Yes* when the default is *no*; and a penality of 4 when the prediction
is *No* when the actual is *yes*.

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                | predicted default 
    ## actual default |        no |       yes | Row Total | 
    ## ---------------|-----------|-----------|-----------|
    ##             no |        34 |        31 |        65 | 
    ##                |     0.340 |     0.310 |           | 
    ## ---------------|-----------|-----------|-----------|
    ##            yes |         5 |        30 |        35 | 
    ##                |     0.050 |     0.300 |           | 
    ## ---------------|-----------|-----------|-----------|
    ##   Column Total |        39 |        61 |       100 | 
    ## ---------------|-----------|-----------|-----------|
    ## 
    ## 

# Rule Learners

(Different than decision tree learners)

— you are on page 315 —-
