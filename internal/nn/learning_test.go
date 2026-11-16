package nn_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/inahym196/mnist-cnn/internal/analysis"
	"github.com/inahym196/mnist-cnn/internal/loss"
	"github.com/inahym196/mnist-cnn/internal/nn"
	"github.com/inahym196/mnist-cnn/internal/optim"
	"github.com/inahym196/mnist-cnn/internal/visualize"
)

type Sample struct {
	x      []float64
	target int
}

func evaluteXOR(samples []Sample, l1 *nn.Linear, relu *nn.ReLU, l2 *nn.Linear) (float64, float64) {
	var totalLoss float64
	correct := 0

	for _, sample := range samples {
		h := l1.Forward(sample.x)
		a := relu.Forward(h)
		logits := l2.Forward(a)

		lossValue, _ := loss.CrossEntropy(logits, sample.target)
		totalLoss += lossValue
		predicted := 0
		if logits[1] > logits[0] {
			predicted = 1
		}

		if predicted == sample.target {
			correct++
		}
	}

	return totalLoss / float64(len(samples)), float64(correct) / float64(len(samples))
}

func TestLearnXOR(t *testing.T) {

	in, h, out := 2, 2, 2
	samples := []Sample{
		{[]float64{0, 0}, 0},
		{[]float64{0, 1}, 1},
		{[]float64{1, 0}, 1},
		{[]float64{1, 1}, 0},
	}

	rng := rand.New(rand.NewSource(2))

	l1 := nn.NewLinear(in, h)
	relu := nn.NewReLU()
	l2 := nn.NewLinear(h, out)

	l1.InitHe(rng)
	l2.InitHe(rng)

	sgd := optim.NewSGD(0.05)

	lastEpoch := 5000
	for epoch := range lastEpoch + 1 {
		if epoch == lastEpoch {
			avgLoss, accuracy := evaluteXOR(samples, l1, relu, l2)
			t.Logf("epoch=%4d loss=%.6f accuracy=%3.0f%%", epoch, avgLoss, accuracy*100)

			path := fmt.Sprintf("xor_epoch_%05d.png", epoch)
			if err := saveDecisionBoundary(path, l1, relu, l2); err != nil {
				t.Fatal(err)
			}
		}
		if epoch == lastEpoch {

			model, err := analysis.CaptureModel(l1, relu, l2)
			if err != nil {
				t.Fatal(err)
			}

			t.Log("\n" + analysis.FormatModel(model))

			for i, sample := range samples {
				activation, err := analysis.CaptureActivation(sample.x, l1, relu, l2)
				if err != nil {
					t.Fatal(err)
				}
				t.Log("\n" + analysis.FormatActivation(model, activation))
				path := fmt.Sprintf("xor_%d.svg", i)
				if err := visualize.SaveActivationSVG(path, model, activation, visualize.SVGOptions{}); err != nil {
					t.Fatal(err)
				}
			}
			break
		}
		for _, sample := range samples {
			l1.ZeroGrad()
			l2.ZeroGrad()

			h := l1.Forward(sample.x)
			a := relu.Forward(h)
			logits := l2.Forward(a)

			_, grad := loss.CrossEntropy(logits, sample.target)

			grad = l2.Backward(grad)
			grad = relu.Backward(grad)
			l1.Backward(grad)

			sgd.Step(l1.Parameters()...)
			sgd.Step(l2.Parameters()...)
		}
	}

	for _, sample := range samples {
		h := l1.Forward(sample.x)
		a := relu.Forward(h)
		logits := l2.Forward(a)

		predicted := 0
		if logits[1] > logits[0] {
			predicted = 1
		}

		if predicted != sample.target {
			t.Errorf("input=%v: predicted=%d, want=%d", sample.x, predicted, sample.target)
		}
	}
}
