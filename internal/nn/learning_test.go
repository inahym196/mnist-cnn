package nn_test

import (
	"math/rand"
	"testing"

	"github.com/inahym196/mnist-cnn/internal/loss"
	"github.com/inahym196/mnist-cnn/internal/nn"
	"github.com/inahym196/mnist-cnn/internal/optim"
)

func TestLearnXOR(t *testing.T) {

	in, h, out := 2, 2, 2

	samples := []struct {
		x      []float64
		target int
	}{
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

	for range 1000 {
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
