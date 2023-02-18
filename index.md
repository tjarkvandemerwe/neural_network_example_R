
- <a href="#setup" id="toc-setup">Setup</a>
- <a href="#forward-pass" id="toc-forward-pass">Forward pass</a>
- <a href="#back-propagation" id="toc-back-propagation">Back
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
      nrow = neurons_to,
      dimnames = list(
        paste0("to_", 1:neurons_to),
        paste0("from_", 1:neurons_from)
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
    ##            from_1     from_2
    ## to_1  0.709380241  0.2175265
    ## to_2 -0.009977894  0.8085342
    ## to_3 -0.722198173 -0.4571307
    ## 
    ## $step_1$biases
    ## [1] -0.2760984  0.4944348 -0.2455274
    ## 
    ## 
    ## $step_2
    ## $step_2$weights
    ##          from_1      from_2     from_3
    ## to_1  0.4240072 -0.56112708  0.7498339
    ## to_2  0.8195974 -0.09180253 -0.1685863
    ## to_3 -0.5462511  0.76670882 -0.3449100
    ## 
    ## $step_2$biases
    ## [1]  0.4533401  0.6423786 -0.2082427
    ## 
    ## 
    ## $step_3
    ## $step_3$weights
    ##           from_1      from_2     from_3
    ## to_1 -0.09265419 -0.03291666 -0.3298822
    ## 
    ## $step_3$biases
    ## [1] 0.7473469

# Forward pass

To pass the signal through the network we need to iterate over the
layers in the network, following the steps: - Calculate weigthed sum of
inputs to each neuron - Minus bias of neuron - Activation function

To do this three functions are build.

### Weighted sum activation

This function computes the weighted sum of all activations leading to
this layer.

``` r
weighted_activation <- function(input_activation, weights) {
  weights %*% input_activation
}

# Example:
weights_and_biases$step_1$weights
```

    ##            from_1     from_2
    ## to_1  0.709380241  0.2175265
    ## to_2 -0.009977894  0.8085342
    ## to_3 -0.722198173 -0.4571307

``` r
weighted_activation(c(1, 0.5), weights_and_biases$step_1$weights)
```

    ##            [,1]
    ## to_1  0.8181435
    ## to_2  0.3942892
    ## to_3 -0.9507635

### Reduce activation with bias

The bias is removed form the activation for each neuron in the layer.
This acts as a sort of treshhold.

``` r
remove_bias <- function(basic_activations, biases) {
  basic_activations - biases
}

# Example:
c(1, 0.5) |>
  weighted_activation(weights_and_biases$step_1$weights) |>
  remove_bias(c(-1, 0, 1))
```

    ##            [,1]
    ## to_1  1.8181435
    ## to_2  0.3942892
    ## to_3 -1.9507635

### Apply activation function

For each neuron the value after the bias is removed, is put into an
activation function. This is the final value for the activation of this
layer, and provides the input for the next layer.

``` r
sigmoid <- function(basic_activations) {
  1 / (1 + exp(-basic_activations))
}

# Example:
c(1, 0.5) |>
  weighted_activation(weights_and_biases$step_1$weights) |>
  remove_bias(c(-1, 0, 1)) |> 
  sigmoid()
```

    ##           [,1]
    ## to_1 0.8603432
    ## to_2 0.5973148
    ## to_3 0.1244701

# Back propagation
