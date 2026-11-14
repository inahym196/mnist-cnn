package nn

type Parameter struct {
	Value []float64
	Grad  []float64
}

func NewParameter(size int) Parameter {
	return Parameter{
		Value: make([]float64, size),
		Grad:  make([]float64, size),
	}
}

func (p *Parameter) ZeroGrad() {
	clear(p.Grad)
}
