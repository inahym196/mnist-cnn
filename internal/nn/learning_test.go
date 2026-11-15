package nn_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/inahym196/mnist-cnn/internal/loss"
	"github.com/inahym196/mnist-cnn/internal/nn"
	"github.com/inahym196/mnist-cnn/internal/optim"
)

func weightNorm(l *nn.Linear) float64 {
	weights := l.Parameters()[0].Value

	var sum float64
	for _, w := range weights {
		sum += w * w
	}
	return math.Sqrt(sum)
}

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

func logWeights(t *testing.T, name string, l *nn.Linear, in, out int) {
	t.Helper()

	weights := l.Parameters()[0].Value

	t.Logf("%s weights:", name)
	for j := range out {
		row := weights[j*in : (j+1)*in]
		t.Logf("  neuron[%d]: %+.4f", j, row)
	}
}

func TestLearnXOR(t *testing.T) {

	h := 8
	samples := []Sample{
		{[]float64{0, 0}, 0},
		{[]float64{0, 1}, 1},
		{[]float64{1, 0}, 1},
		{[]float64{1, 1}, 0},
	}

	rng := rand.New(rand.NewSource(42))

	l1 := nn.NewLinear(2, h)
	relu := nn.NewReLU()
	l2 := nn.NewLinear(h, 2)

	l1.InitHe(rng)
	l2.InitHe(rng)

	sgd := optim.NewSGD(0.05)

	for epoch := range 5001 {
		if epoch%2500 == 0 {
			avgLoss, accuracy := evaluteXOR(samples, l1, relu, l2)
			t.Logf(
				"epoch=%4d loss=%.6f accuracy=%3.0f%% L1_norm=%.4f L2_norm=%.4f",
				epoch,
				avgLoss,
				accuracy*100,
				weightNorm(l1),
				weightNorm(l2),
			)
			t.Logf("L1 biases: %+.4f", l1.Parameters()[1].Value)
			logWeights(t, "L1", l1, 2, h)
			t.Logf("L2 biases: %+.4f", l2.Parameters()[1].Value)
			logWeights(t, "L2", l2, h, 2)
		}
		if epoch == 5000 {

			for _, sample := range samples {
				h := l1.Forward(sample.x)
				a := relu.Forward(h)
				logits := l2.Forward(a)

				t.Logf(
					"x=%v hidden=%+.3f logits=%+.3f delta=%.3f",
					sample.x,
					a,
					logits,
					logits[1]-logits[0],
				)
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
