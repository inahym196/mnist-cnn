package train

import (
	"fmt"

	"github.com/inahym196/mnist-cnn/internal/loss"
)

type Metrics struct {
	Loss     float64
	Accuracy float64
}

func Evaluate(
	model Model,
	data Dataset,
) (Metrics, error) {
	if data.Len() == 0 {
		return Metrics{}, fmt.Errorf("empty evaluation dataset")
	}

	var totalLoss float64
	correct := 0

	for i := 0; i < data.Len(); i++ {
		x, target := data.Get(i)

		logits := model.Forward(x)

		if target < 0 || target >= len(logits) {
			return Metrics{}, fmt.Errorf(
				"sample %d: invalid target %d",
				i, target,
			)
		}

		lossValue, _ := loss.CrossEntropy(logits, target)
		totalLoss += lossValue

		if argmax(logits) == target {
			correct++
		}
	}

	return Metrics{
		Loss:     totalLoss / float64(data.Len()),
		Accuracy: float64(correct) / float64(data.Len()),
	}, nil
}

func argmax(values []float64) int {
	maxIndex := 0

	for i := 1; i < len(values); i++ {
		if values[i] > values[maxIndex] {
			maxIndex = i
		}
	}

	return maxIndex
}
