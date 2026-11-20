package main

import "github.com/inahym196/mnist-cnn/internal/nn"

type MLP struct {
	L1   *nn.Linear
	ReLU *nn.ReLU
	L2   *nn.Linear
}

func NewMLP() *MLP {
	return &MLP{
		L1:   nn.NewLinear(784, 128),
		ReLU: nn.NewReLU(),
		L2:   nn.NewLinear(128, 10),
	}
}

func (m *MLP) Forward(x []float64) []float64 {
	x = m.L1.Forward(x)
	x = m.ReLU.Forward(x)
	x = m.L2.Forward(x)
	return x
}

func (m *MLP) Backward(grad []float64) []float64 {
	grad = m.L2.Backward(grad)
	grad = m.ReLU.Backward(grad)
	grad = m.L1.Backward(grad)
	return grad
}

func (m *MLP) ZeroGrad() {
	m.L1.ZeroGrad()
	m.L2.ZeroGrad()
}

func (m *MLP) Parameters() []*nn.Parameter {
	params := m.L1.Parameters()
	params = append(params, m.L2.Parameters()...)
	return params
}
