package nn

import (
	"math"
	"math/rand"
)

func HeInit(p *Parameter, fanIn int, rng *rand.Rand) {
	stddev := math.Sqrt(2.0 / float64(fanIn))
	for i := range p.Value {
		p.Value[i] = rng.NormFloat64() * stddev
	}
}
