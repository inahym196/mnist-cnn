package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/inahym196/mnist-cnn/internal/datasets/mnist"
	"github.com/inahym196/mnist-cnn/internal/optim"
	"github.com/inahym196/mnist-cnn/internal/train"
)

// MNISTのDatasetをtrain.Datasetとして扱う。
// 読み込み済みの画像データはコピーしない。
type mnistDataset struct {
	*mnist.Dataset
}

func (d mnistDataset) Get(i int) ([]float64, int) {
	return d.Images[i], int(d.Labels[i])
}

func main() {
	// 1. データセットの読み込み
	trainData, err := mnist.Load(
		"data/mnist/train-images-idx3-ubyte",
		"data/mnist/train-labels-idx1-ubyte",
	)
	if err != nil {
		log.Fatal(err)
	}

	testData, err := mnist.Load(
		"data/mnist/t10k-images-idx3-ubyte",
		"data/mnist/t10k-labels-idx1-ubyte",
	)
	if err != nil {
		log.Fatal(err)
	}

	// 2. モデルの構築・初期化
	model := NewMLP()

	initRNG := rand.New(rand.NewSource(42))
	model.L1.InitHe(initRNG)
	model.L2.InitHe(initRNG)

	// 3. Optimizer
	optimizer := optim.NewSGD(0.01)

	// 4. 学習条件
	const epochs = 10

	trainRNG := rand.New(rand.NewSource(123))

	// 5. Epoch全体の制御
	for epoch := 1; epoch <= epochs; epoch++ {
		trainLoss, err := train.TrainEpoch(
			model,
			mnistDataset{trainData},
			optimizer,
			trainRNG,
		)
		if err != nil {
			log.Fatal(err)
		}

		trainMetrics, err := train.Evaluate(
			model,
			mnistDataset{trainData},
		)
		if err != nil {
			log.Fatal(err)
		}

		testMetrics, err := train.Evaluate(
			model,
			mnistDataset{testData},
		)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf(
			"epoch=%2d loss=%.4f train_acc=%.2f%% test_acc=%.2f%%\n",
			epoch,
			trainLoss,
			trainMetrics.Accuracy*100,
			testMetrics.Accuracy*100,
		)
	}
}
