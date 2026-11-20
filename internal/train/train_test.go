package train

import (
	"math/rand"
	"testing"

	"github.com/inahym196/mnist-cnn/internal/nn"
	"github.com/inahym196/mnist-cnn/internal/optim"
)

type testSample struct {
	x      []float64
	target int
}

type testDataset []testSample

func (d testDataset) Len() int {
	return len(d)
}

func (d testDataset) Get(i int) ([]float64, int) {
	return d[i].x, d[i].target
}

type testModel struct {
	linear *nn.Linear
}

func (m *testModel) Forward(x []float64) []float64 {
	return m.linear.Forward(x)
}

func (m *testModel) Backward(grad []float64) []float64 {
	return m.linear.Backward(grad)
}

func (m *testModel) ZeroGrad() {
	m.linear.ZeroGrad()
}

func (m *testModel) Parameters() []*nn.Parameter {
	return m.linear.Parameters()
}

func TestTrainEpoch(t *testing.T) {
	model := &testModel{
		linear: nn.NewLinear(1, 2),
	}

	data := testDataset{
		{x: []float64{1}, target: 1},
	}

	before, err := Evaluate(model, data)
	if err != nil {
		t.Fatal(err)
	}

	optimizer := optim.NewSGD(0.1)
	rng := rand.New(rand.NewSource(1))

	_, err = TrainEpoch(model, data, optimizer, rng)
	if err != nil {
		t.Fatal(err)
	}

	after, err := Evaluate(model, data)
	if err != nil {
		t.Fatal(err)
	}

	if after.Loss >= before.Loss {
		t.Fatalf(
			"loss did not decrease: before=%f after=%f",
			before.Loss, after.Loss,
		)
	}
}

func TestEvaluate(t *testing.T) {
	model := &testModel{
		linear: nn.NewLinear(1, 2),
	}

	// 常にクラス1を予測するように設定する。
	params := model.linear.Parameters()
	params[1].Value[0] = 0
	params[1].Value[1] = 2

	data := testDataset{
		{x: []float64{0}, target: 1},
		{x: []float64{1}, target: 0},
	}

	metrics, err := Evaluate(model, data)
	if err != nil {
		t.Fatal(err)
	}

	if metrics.Accuracy != 0.5 {
		t.Fatalf(
			"accuracy = %f, want 0.5",
			metrics.Accuracy,
		)
	}
}
