package main

import (
	t "gorgonia.org/tensor"
	"image"
	"image/png"
	"log"
	"os"
)

func handleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func noErr(X t.Tensor, err error) t.Tensor {
	handleErr(err)
	return X
}

func checkData(X, y t.Tensor, err error, mode string) {
	handleErr(err)
	var sampleSize int
	if mode == "train" {

		sampleSize = 60000
	} else {
		sampleSize = 10000
	}
	if X.Shape()[0] != sampleSize || X.Shape()[1] != 784 {
		log.Fatal("wrong X shape")
	}
	if y.Shape()[0] != sampleSize || y.Shape()[1] != 10 {
		log.Fatal("wrong y shape")
	}
}

func saveImage(inputs Matrix) {
	cols := inputs.Shape()[1]
	imageBackend := make([]uint8, cols)
	for i := 0; i < cols; i++ {
		v := inputs[0][i]
		imageBackend[i] = uint8((v - 0.1) * 0.9 * 255)
	}
	img := &image.Gray{
		Pix:    imageBackend,
		Stride: 28,
		Rect:   image.Rect(0, 0, 28, 28),
	}
	w, _ := os.Create("output.png")
	err := png.Encode(w, img)
	handleErr(err)
}
