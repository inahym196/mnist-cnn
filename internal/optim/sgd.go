package optim

import "github.com/inahym196/mnist-cnn/internal/nn"

// Stochastic Gradient Descent: 確率的勾配降下法
type SGD struct {
	lr float64 // Learning Rate
}

func NewSGD(lr float64) *SGD {
	return &SGD{lr: lr}
}

func (s *SGD) Step(params ...*nn.Parameter) {
	for _, p := range params {
		for i, grad := range p.Grad {
			p.Value[i] -= s.lr * grad
		}
	}
}
