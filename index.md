
- <a href="#setup" id="toc-setup">Setup</a>
- <a href="#forward-propagation" id="toc-forward-propagation">Forward
  propagation</a>

# Setup

### Goal

The goal is to create a working example of a basic deep neural network
in R. This example uses the tidyverse style of R. The main goal is
clarity and explaining the concept, not speed or practical application.

### Packages

``` r
library(tidyverse)
```

### Training data

The training data used in this example is fake data. And this example
starts of with an output that is lineairly related to two input
variables.

``` r
data <- tibble(
  input_value_1 = c(0:10) / 10,
  input_value_2 = c(10:0) / 50,
  output_value = c(0:10) / -5 + 1
)

data |> head(7)
```

    ## # A tibble: 7 × 3
    ##   input_value_1 input_value_2 output_value
    ##           <dbl>         <dbl>        <dbl>
    ## 1           0            0.2           1  
    ## 2           0.1          0.18          0.8
    ## 3           0.2          0.16          0.6
    ## 4           0.3          0.14          0.4
    ## 5           0.4          0.12          0.2
    ## 6           0.5          0.1           0  
    ## 7           0.6          0.08         -0.2

### Setup neural network

``` r
input_neurons  <- 2
output_neurons <- 1
layers         <- 2
layer_neurons  <- 3
```

Every layer, except for the input layer, has a weigth matrix and bias
vector associated with it. These are also related to the computational
steps when passing the signal forward. So the structure to store these
parameters will look like this:

    ## $first_step
    ## $first_step$weigths
    ##      [,1] [,2]
    ## [1,]    1    1
    ## [2,]    1    1
    ## 
    ## $first_step$biases
    ## [1] 1 1
    ## 
    ## 
    ## $second_step
    ## $second_step$weigths
    ##      [,1] [,2]
    ## [1,]    1    1
    ## [2,]    1    1
    ## 
    ## $second_step$biases
    ## [1] 1 1

First a matrix of all weights (w) from the input neurons to the first
layer of hidden neurons is created. This is an input_neurons x
layer_neurons matrix, where input_neurons is the amount of neurons in
the input layer, and layer_neurons is the amount of neurons per hidden
layer. And this is repeated until the last layer to the output neurons
is innitialised.

``` r
neurons_per_layer <- c(input_neurons, rep(layer_neurons, layers), output_neurons)

create_weights_and_biases <- function(neurons_from, neurons_to) {
  list(
    weights = matrix(
      runif(neurons_from * neurons_to, -1, 1),
      nrow = neurons_from,
      dimnames = list(
        paste0("from_", 1:neurons_from),
        paste0("to_", 1:neurons_to)
      )
    ),
    biases = runif(neurons_to, -1, 1)
  )
}

weights_and_biases <-
  map2(neurons_per_layer[-length(neurons_per_layer)],
       neurons_per_layer[-1],
       create_weights_and_biases)

names(weights_and_biases) <-
  paste0("step_", 1:length(weights_and_biases))

weights_and_biases
```

    ## $step_1
    ## $step_1$weights
    ##               to_1       to_2       to_3
    ## from_1  0.01526446 -0.1457231 0.66190010
    ## from_2 -0.08460486 -0.5003968 0.02126486
    ## 
    ## $step_1$biases
    ## [1]  0.9831834 -0.5801799  0.5784724
    ## 
    ## 
    ## $step_2
    ## $step_2$weights
    ##              to_1       to_2       to_3
    ## from_1 -0.1133457 -0.8707159  0.9119872
    ## from_2  0.8050565 -0.2220603 -0.4712405
    ## from_3 -0.3201872  0.7425106 -0.2090557
    ## 
    ## $step_2$biases
    ## [1] -0.08921057  0.35723003 -0.33216029
    ## 
    ## 
    ## $step_3
    ## $step_3$weights
    ##              to_1
    ## from_1 -0.6074537
    ## from_2 -0.4153567
    ## from_3 -0.1390953
    ## 
    ## $step_3$biases
    ## [1] -0.5970573

# Forward propagation
