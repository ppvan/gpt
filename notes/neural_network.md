
Neural network: A small model that will predict something. For example, house pricing
size -> NN -> price

For more complex more, need complex NN. Like

position, size, bedroom -> NN -> price



Lost function (errror) function: Error of a single training example
Cost function: error of whole train set.

Many type of good math properties of fucntions:
- binary cross-entropy loss (BCE): `result += -y*math.Log(y1) - (1-y)*math.Log(1-y1)`
- Mean Squared Error: `result = (y−y^​)2`