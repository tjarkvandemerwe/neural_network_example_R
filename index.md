
- <a href="#setup" id="toc-setup">Setup</a>
- <a href="#forward-pass" id="toc-forward-pass">Forward pass</a>
- <a href="#loss-function" id="toc-loss-function">Loss function</a>
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
layer. This way, the n-th, and the n+1-th layers connections are made.
And this is repeated until the last layer to the output neurons is
innitialised.

``` r
neurons_per_layer <- c(input_neurons, rep(layer_neurons, layers), output_neurons)

neurons_per_layer
```

    ## [1] 2 3 3 1

``` r
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
    ##          from_1     from_2
    ## to_1 -0.4997575  0.6837431
    ## to_2 -0.7703853 -0.2668109
    ## to_3 -0.1781081  0.5893086
    ## 
    ## $step_1$biases
    ## [1] -0.4900420 -0.5066407  0.4384855
    ## 
    ## 
    ## $step_2
    ## $step_2$weights
    ##           from_1    from_2     from_3
    ## to_1  0.04392015 0.3071597  0.2845464
    ## to_2 -0.60557903 0.6634979 -0.8743305
    ## to_3  0.97904834 0.3358192  0.2247424
    ## 
    ## $step_2$biases
    ## [1]  0.45497936 -0.54720477  0.03120902
    ## 
    ## 
    ## $step_3
    ## $step_3$weights
    ##          from_1     from_2    from_3
    ## to_1 0.04403913 -0.2195956 0.5941846
    ## 
    ## $step_3$biases
    ## [1] -0.9083015

# Forward pass

To pass the signal through the network we need to iterate over the
layers in the network, following the steps:

- Calculate weigthed sum of inputs to each neuron
- Minus bias of neuron
- Activation function

To do this three functions are build.

### Weighted sum activation

This function computes the weighted sum of all activations leading to
this layer.

``` r
weighted_activation <- function(input_activation, weights) {
  weights %*% input_activation
}

# Example (using example input vector [1, 0.5]):
weights_and_biases$step_1$weights
```

    ##          from_1     from_2
    ## to_1 -0.4997575  0.6837431
    ## to_2 -0.7703853 -0.2668109
    ## to_3 -0.1781081  0.5893086

``` r
c(1, 0.5) |>
  weighted_activation(weights_and_biases$step_1$weights)
```

    ##            [,1]
    ## to_1 -0.1578860
    ## to_2 -0.9037908
    ## to_3  0.1165462

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
    ## to_1  0.8421140
    ## to_2 -0.9037908
    ## to_3 -0.8834538

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
    ## to_1 0.6989103
    ## to_2 0.2882721
    ## to_3 0.2924626

### Total forward pass

The total feed forward pass apply’s the above three functions in a
sequence. To make it easy a new function is build applying the above
three steps.

``` r
activate_layer <- function(input_activations, weights_and_biases) {
  input_activations |>
    weighted_activation(weights_and_biases$weights) |>
    remove_bias(weights_and_biases$biases) |>
    sigmoid()
}

# Now we want to iteratively apply this function, carrying the activation forward, 
#   this can be done like this, however this doesn't scale with the network:
c(1, 0.5) |>
  activate_layer(weights_and_biases$step_1) |>
  activate_layer(weights_and_biases$step_2) |>
  activate_layer(weights_and_biases$step_3)
```

    ##           [,1]
    ## to_1 0.7719308

``` r
# This can be made more simple and scaling using the reduce function:
reduce(weights_and_biases, activate_layer, .init = c(1, 0.5))
```

    ##           [,1]
    ## to_1 0.7719308

This concludes the example of the forward pass.

### Shortform

We can rewrite the big example in the following short code:

``` r
activate_layer_short <- function(a, wb) {
  1 / (1 + exp(-((wb$weights %*% a) - wb$biases)))
}

reduce(weights_and_biases, activate_layer_short, .init = c(1, 0.5))
```

    ##           [,1]
    ## to_1 0.7719308

# Loss function

# Back propagation
