package visualize

import "github.com/inahym196/mnist-cnn/internal/analysis"

func SaveActivationSVG(
	path string,
	model analysis.ModelSnapshot,
	activation analysis.ActivationSnapshot,
	opts SVGOptions) error {
	scene, err := BuildScene(model, activation, opts.Layout)
	if err != nil {
		return err
	}
	return SaveSVG(path, scene)
}
