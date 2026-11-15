package nn_test

import (
	"image"
	"image/color"
	"image/png"
	"os"

	nn "github.com/inahym196/mnist-cnn/internal/nn"
)

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
