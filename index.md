
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
starts of with an output that is lineairly related to an input variable.

``` r
data <- tibble(
  input_value = c(0:10) / 10,
  output_value = c(0:10) / -5 + 1
)

data |> head(7)
```

    ## # A tibble: 7 × 2
    ##   input_value output_value
    ##         <dbl>        <dbl>
    ## 1         0            1  
    ## 2         0.1          0.8
    ## 3         0.2          0.6
    ## 4         0.3          0.4
    ## 5         0.4          0.2
    ## 6         0.5          0  
    ## 7         0.6         -0.2

### Setup neural network

``` r
input_neurons  <- 1
output_neurons <- 1
layers         <- 1
layer_neurons  <- 2


nn <- list(
  w_input = matrix(sample(0:100 / 100, layer_neurons * input_neurons), nrow = 1)
)

if(layers > 1){
  for(n in 1:(layers - 1)){
    a <- list(matrix(sample(0:100 / 100, layer_neurons * layer_neurons), nrow = layer_neurons))
    names(a) <- paste0("w_hidden_layer_", n)
    nn <- c(nn, a)
  }
}

nn <- c(nn,
        list(w_output = matrix(
          sample(0:100 / 100, layer_neurons * output_neurons), ncol = 1
        )))


nn
```

    ## $w_input
    ##      [,1] [,2]
    ## [1,] 0.48  0.3
    ## 
    ## $w_output
    ##      [,1]
    ## [1,] 0.44
    ## [2,] 0.97

# Forward propagation
