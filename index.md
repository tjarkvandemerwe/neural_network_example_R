
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
    weights = matrix(runif(neurons_from * neurons_to, -1, 1), nrow = neurons_from),
    biases = runif(neurons_to, -1, 1)
  )
}

weights_and_biases <-
  map2(neurons_per_layer[-length(neurons_per_layer)],
       neurons_per_layer[-1],
       create_weights_and_biases)

weights_and_biases
```

    ## [[1]]
    ## [[1]]$weights
    ##            [,1]        [,2]       [,3]
    ## [1,] -0.5447291 -0.05338535 -0.3846959
    ## [2,] -0.8776336  0.77318856  0.4785767
    ## 
    ## [[1]]$biases
    ## [1] 0.1557528 0.7701158 0.4271647
    ## 
    ## 
    ## [[2]]
    ## [[2]]$weights
    ##            [,1]       [,2]       [,3]
    ## [1,]  0.9059758  0.5886812 0.58645821
    ## [2,] -0.3910393  0.6240854 0.07163238
    ## [3,]  0.0614815 -0.4041383 0.42474352
    ## 
    ## [[2]]$biases
    ## [1]  0.1348914  0.9244048 -0.9783119
    ## 
    ## 
    ## [[3]]
    ## [[3]]$weights
    ##            [,1]
    ## [1,] -0.5119770
    ## [2,] -0.9987706
    ## [3,] -0.8815305
    ## 
    ## [[3]]$biases
    ## [1] 0.3786865

# Forward propagation
