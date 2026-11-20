package main

import (
	"fmt"
	"log"

	"github.com/inahym196/mnist-cnn/internal/datasets/mnist"
)

func main() {
	train, err := mnist.Load(
		"data/mnist/train-images-idx3-ubyte",
		"data/mnist/train-labels-idx1-ubyte",
	)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(train.Len())
	fmt.Println(len(train.Images[0]))
	fmt.Println(train.Labels[0])
}
