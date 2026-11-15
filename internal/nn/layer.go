package nn

type Layer interface {
	Forward(x []float64) []float64
}
