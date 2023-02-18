
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

First a matrix of all weights (w) from the input neurons to the first
layer of hidden neurons is created. This is an input_neurons x
layer_neurons matrix, where input_neurons is the amount of neurons in
the input layer, and layer_neurons is the amount of neurons per hidden
layer.

``` r
nn_input <- list(
  w_input_hidden_layer_1 = matrix(sample(0:100 / 100, layer_neurons * input_neurons), nrow = input_neurons)
)

nn_input
```

    ## $w_input_hidden_layer_1
    ##      [,1] [,2] [,3]
    ## [1,] 0.89 1.00 0.55
    ## [2,] 0.52 0.28 0.24

If there are multiple hidden layers, the weights between these layers
are created, this is an layer_neurons x layer_neurons matrix. Each
hidden layer also has a bias value for every neuron in the layer. This
is a single row matrix (vector) of length layer_neurons.

``` r
w_hidden_layers <- list()
b_hidden_layers <- list()

for(n in 1:layers){
  
  b_hidden <- list(matrix(sample(0:100 / 100, layer_neurons), nrow = 1))
  names(b_hidden) <- paste("b_hidden_layer", n , sep = "_")
  b_hidden_layers <- c(b_hidden_layers, b_hidden)
  
  if(n > 1){
    w_hidden <- list(matrix(sample(0:100 / 100, layer_neurons * layer_neurons), nrow = layer_neurons))
    names(w_hidden) <- paste("w_hidden_layer", n - 1, n, sep = "_")
    w_hidden_layers <- c(w_hidden_layers, w_hidden)
  }
}

nn_hidden <- c(w_hidden_layers, b_hidden_layers)
nn_hidden
```

    ## $w_hidden_layer_1_2
    ##      [,1] [,2] [,3]
    ## [1,] 0.14 0.48 0.31
    ## [2,] 0.95 0.45 0.84
    ## [3,] 0.41 0.23 0.07
    ## 
    ## $b_hidden_layer_1
    ##      [,1] [,2] [,3]
    ## [1,] 0.42 0.07 0.05
    ## 
    ## $b_hidden_layer_2
    ##      [,1] [,2] [,3]
    ## [1,] 0.03 0.42 0.21

``` r
nn_output <- list(
  w_output_layer_n = matrix(sample(0:100 / 100, layer_neurons * output_neurons), nrow = output_neurons),
  b_output_layer = matrix(sample(0:100 / 100, output_neurons), nrow = 1)
  )

nn_output
```

    ## $w_output_layer_n
    ##      [,1] [,2] [,3]
    ## [1,]  0.5 0.13 0.51
    ## 
    ## $b_output_layer
    ##      [,1]
    ## [1,] 0.18

``` r
nn <- c(nn_input, nn_hidden, nn_output)
nn
```

    ## $w_input_hidden_layer_1
    ##      [,1] [,2] [,3]
    ## [1,] 0.89 1.00 0.55
    ## [2,] 0.52 0.28 0.24
    ## 
    ## $w_hidden_layer_1_2
    ##      [,1] [,2] [,3]
    ## [1,] 0.14 0.48 0.31
    ## [2,] 0.95 0.45 0.84
    ## [3,] 0.41 0.23 0.07
    ## 
    ## $b_hidden_layer_1
    ##      [,1] [,2] [,3]
    ## [1,] 0.42 0.07 0.05
    ## 
    ## $b_hidden_layer_2
    ##      [,1] [,2] [,3]
    ## [1,] 0.03 0.42 0.21
    ## 
    ## $w_output_layer_n
    ##      [,1] [,2] [,3]
    ## [1,]  0.5 0.13 0.51
    ## 
    ## $b_output_layer
    ##      [,1]
    ## [1,] 0.18

# Forward propagation
