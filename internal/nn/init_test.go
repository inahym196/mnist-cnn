package nn

import (
	"math"
	"math/rand"
	"testing"
)

func TestHeInit(t *testing.T) {
	const (
		fanIn = 100
		size  = 100_000
	)
	p := NewParameter(size)
	rng := rand.New(rand.NewSource(42))

	HeInit(&p, fanIn, rng)

	var sum, sumSq float64
	for _, v := range p.Value {
		sum += v
		sumSq += v * v
	}

	mean := sum / size
	variance := sumSq/size - mean*mean
	wantVariance := 2.0 / fanIn

	if math.Abs(mean) > 0.01 {
		t.Errorf("mean too far from zero: %v", mean)
	}

	if math.Abs(variance-wantVariance) > 0.002 {
		t.Errorf("variance: got %v, want approximately %v", variance, wantVariance)
	}

}
