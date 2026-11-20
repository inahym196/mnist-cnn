package mnist

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoad(t *testing.T) {
	dir := t.TempDir()

	writeIDX := func(name string, header []uint32, data []byte) string {
		t.Helper()

		path := filepath.Join(dir, name)
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		for _, v := range header {
			if err := binary.Write(f, binary.BigEndian, v); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := f.Write(data); err != nil {
			t.Fatal(err)
		}
		return path
	}

	pixels := make([]byte, 784)
	pixels[0] = 0
	pixels[1] = 128
	pixels[2] = 255

	imagePath := writeIDX(
		"images.idx",
		[]uint32{2051, 1, 28, 28},
		pixels,
	)
	labelPath := writeIDX(
		"labels.idx",
		[]uint32{2049, 1},
		[]byte{5},
	)

	dataset, err := Load(imagePath, labelPath)
	if err != nil {
		t.Fatal(err)
	}

	if dataset.Len() != 1 {
		t.Fatalf("got %d images, want 1", dataset.Len())
	}

	image := dataset.Images[0]
	if len(image) != 784 {
		t.Fatalf("got %d pixels, want 784", len(image))
	}

	if dataset.Labels[0] != 5 {
		t.Errorf("got label %d, want 5", dataset.Labels[0])
	}

	for i, want := range []float64{0, 128.0 / 255.0, 1} {
		if math.Abs(image[i]-want) > 1e-12 {
			t.Errorf("pixel[%d] = %f, want %f", i, image[i], want)
		}
	}
}
