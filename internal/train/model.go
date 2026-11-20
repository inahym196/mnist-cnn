package train

import (
	"fmt"
	"math/rand"

	"github.com/inahym196/mnist-cnn/internal/loss"
	"github.com/inahym196/mnist-cnn/internal/nn"
)

type Model interface {
	Forward(x []float64) []float64
	Backward(grad []float64) []float64
	ZeroGrad()
	Parameters() []*nn.Parameter
}

type Optimizer interface {
	Step(params ...*nn.Parameter)
}

type Dataset interface {
	Len() int
	Get(i int) ([]float64, int)
}

func TrainEpoch(model Model, data Dataset, optimizer Optimizer, rng *rand.Rand) (float64, error) {

	if data.Len() == 0 {
		return 0, fmt.Errorf("empty training dataset")
	}

	if rng == nil {
		return 0, fmt.Errorf("nil random generator")
	}

	indices := rng.Perm(data.Len())
	params := model.Parameters()

	var totalLoss float64
	for _, i := range indices {
		x, target := data.Get(i)

		model.ZeroGrad()
		logits := model.Forward(x)
		if target < 0 || target >= len(logits) {
			return 0, fmt.Errorf("sample %d: invalid target %d", i, target)
		}

		lossValue, grad := loss.CrossEntropy(logits, target)
		totalLoss += lossValue

		model.Backward(grad)
		optimizer.Step(params...)
	}
	return totalLoss / float64(data.Len()), nil
}
