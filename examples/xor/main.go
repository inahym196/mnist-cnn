package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"

	"github.com/inahym196/mnist-cnn/internal/analysis"
	"github.com/inahym196/mnist-cnn/internal/loss"
	"github.com/inahym196/mnist-cnn/internal/nn"
	"github.com/inahym196/mnist-cnn/internal/optim"
	"github.com/inahym196/mnist-cnn/internal/visualize"
)

type Sample struct {
	x      []float64
	target int
}

func evaluteXOR(samples []Sample, l1 *nn.Linear, relu *nn.ReLU, l2 *nn.Linear) (float64, float64) {
	var totalLoss float64
	correct := 0

	for _, sample := range samples {
		h := l1.Forward(sample.x)
		a := relu.Forward(h)
		logits := l2.Forward(a)

		lossValue, _ := loss.CrossEntropy(logits, sample.target)
		totalLoss += lossValue
		predicted := 0
		if logits[1] > logits[0] {
			predicted = 1
		}

		if predicted == sample.target {
			correct++
		}
	}

	return totalLoss / float64(len(samples)), float64(correct) / float64(len(samples))
}

func main() {

	in, h, out := 2, 2, 2
	samples := []Sample{
		{[]float64{0, 0}, 0},
		{[]float64{0, 1}, 1},
		{[]float64{1, 0}, 1},
		{[]float64{1, 1}, 0},
	}

	rng := rand.New(rand.NewSource(2))

	l1 := nn.NewLinear(in, h)
	relu := nn.NewReLU()
	l2 := nn.NewLinear(h, out)

	l1.InitHe(rng)
	l2.InitHe(rng)

	sgd := optim.NewSGD(0.05)

	lastEpoch := 5000
	for epoch := range lastEpoch + 1 {
		if epoch == lastEpoch {
			avgLoss, accuracy := evaluteXOR(samples, l1, relu, l2)
			fmt.Printf("epoch=%4d loss=%.6f accuracy=%3.0f%%", epoch, avgLoss, accuracy*100)

			path := fmt.Sprintf("xor_epoch_%05d.png", epoch)
			if err := saveDecisionBoundary(path, l1, relu, l2); err != nil {
				log.Fatal(err)
			}

			model, err := analysis.CaptureModel(l1, relu, l2)
			if err != nil {
				log.Fatal(err)
			}

			fmt.Print("\n" + analysis.FormatModel(model))

			for i, sample := range samples {
				activation, err := analysis.CaptureActivation(sample.x, l1, relu, l2)
				if err != nil {
					log.Fatal(err)
				}
				fmt.Print("\n" + analysis.FormatActivation(model, activation))
				path := fmt.Sprintf("xor_%d.svg", i)
				if err := visualize.SaveActivationSVG(path, model, activation, visualize.SVGOptions{}); err != nil {
					log.Fatal(err)
				}
			}
			break
		}
		for _, sample := range samples {
			l1.ZeroGrad()
			l2.ZeroGrad()

			h := l1.Forward(sample.x)
			a := relu.Forward(h)
			logits := l2.Forward(a)

			_, grad := loss.CrossEntropy(logits, sample.target)

			grad = l2.Backward(grad)
			grad = relu.Backward(grad)
			l1.Backward(grad)

			sgd.Step(l1.Parameters()...)
			sgd.Step(l2.Parameters()...)
		}
	}

	for _, sample := range samples {
		h := l1.Forward(sample.x)
		a := relu.Forward(h)
		logits := l2.Forward(a)

		predicted := 0
		if logits[1] > logits[0] {
			predicted = 1
		}

		if predicted != sample.target {
			log.Fatalf("input=%v: predicted=%d, want=%d", sample.x, predicted, sample.target)
		}
	}
}

func saveDecisionBoundary(path string, l1 *nn.Linear, relu *nn.ReLU, l2 *nn.Linear) error {
	const (
		size = 512
		min  = -0.5
		max  = 1.5
	)
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	class0 := color.RGBA{220, 225, 235, 255}
	class1 := color.RGBA{20, 150, 140, 255}

	for py := range size {
		y := max - float64(py)/float64(size-1)*(max-min)
		for px := range size {
			x := min + float64(px)/float64(size-1)*(max-min)
			h := l1.Forward([]float64{x, y})
			a := relu.Forward(h)
			logits := l2.Forward(a)

			if logits[1] > logits[0] {
				img.Set(px, py, class1)
			} else {
				img.Set(px, py, class0)
			}
		}
	}
	drawPoint := func(x, y float64, c color.Color) {
		px := int((x - min) / (max - min) * (size - 1))
		py := int((max - y) / (max - min) * (size - 1))

		const radius = 5
		for dy := -radius; dy <= radius; dy++ {
			for dx := -radius; dx <= radius; dx++ {
				if dx*dx+dy*dy <= radius*radius {
					img.Set(px+dx, py+dy, c)
				}
			}
		}
	}
	drawPoint(0, 0, color.Black)
	drawPoint(1, 1, color.Black)
	drawPoint(0, 1, color.White)
	drawPoint(1, 0, color.White)

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}
