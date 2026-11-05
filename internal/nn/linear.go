package nn

type Linear struct {
	in, out int

	// 行優先: weights[j*in+i] = W[j][i]
	weights []float64
	biases  []float64
}

func NewLinear(in, out int) *Linear {
	return &Linear{
		in:      in,
		out:     out,
		weights: make([]float64, in*out),
		biases:  make([]float64, out),
	}
}

func (l *Linear) Forward(x []float64) []float64 {
	y := make([]float64, l.out)

	for j := range y {
		sum := l.biases[j]

		for i, v := range x {
			sum += l.weights[j*l.in+i] * v
		}

		y[j] = sum
	}

	return y
}
