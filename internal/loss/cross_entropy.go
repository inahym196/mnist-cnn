package loss

import (
	"math"
	"slices"
)

// CrossEntropy はSoftmax込みの損失を計算する。
// logitsは未正規化の分類スコア。
// targetは正解クラスのインデックス。
func CrossEntropy(logits []float64, target int) (float64, []float64) {
	maxLogit := slices.Max(logits)
	gradLogits := make([]float64, len(logits))

	var sum float64
	for i, v := range logits {
		gradLogits[i] = math.Exp(v - maxLogit)
		sum += gradLogits[i]
	}

	for i := range gradLogits {
		gradLogits[i] /= sum
	}

	loss := maxLogit + math.Log(sum) - logits[target]
	gradLogits[target]--
	return loss, gradLogits
}
