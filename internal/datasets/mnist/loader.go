package mnist

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type Dataset struct {
	Images [][]float64
	Labels []uint8
}

func (d *Dataset) Len() int { return len(d.Labels) }

func LoadImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, count, rows, cols uint32
	for _, v := range []*uint32{&magic, &count, &rows, &cols} {
		if err := binary.Read(f, binary.BigEndian, v); err != nil {
			return nil, err
		}
	}

	if magic != 2051 {
		return nil, fmt.Errorf("invalid image magic: %d", magic)
	}

	if rows != 28 || cols != 28 {
		return nil, fmt.Errorf("unexpected image dimensions: %dx%d", rows, cols)
	}

	images := make([][]float64, int(count))
	buf := make([]byte, 28*28)

	for i := range images {
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, fmt.Errorf("read image %d: %w", i, err)
		}
		pixels := make([]float64, len(buf))
		for j, p := range buf {
			// 値の範囲をint 0~255 から float 0~1 へ 正規化
			pixels[j] = float64(p) / 255.0
		}
		images[i] = pixels
	}

	return images, nil
}

func LoadLabels(path string) ([]uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, count uint32
	for _, v := range []*uint32{&magic, &count} {
		if err := binary.Read(f, binary.BigEndian, v); err != nil {
			return nil, err
		}
	}

	if magic != 2049 {
		return nil, fmt.Errorf("invalid label magic: %d", magic)
	}

	labels := make([]uint8, int(count))
	if _, err := io.ReadFull(f, labels); err != nil {
		return nil, err
	}

	for i, label := range labels {
		if label > 9 {
			return nil, fmt.Errorf("invalid label %d at index %d", label, i)
		}
	}
	return labels, nil
}

func Load(imagePath, labelPath string) (*Dataset, error) {
	images, err := LoadImages(imagePath)
	if err != nil {
		return nil, err
	}

	labels, err := LoadLabels(labelPath)
	if err != nil {
		return nil, err
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("image count %d != label count %d", len(images), len(labels))
	}

	return &Dataset{
		Images: images,
		Labels: labels,
	}, nil
}
