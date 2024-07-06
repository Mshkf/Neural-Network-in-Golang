package main

import (
	"encoding/binary"
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

func safeRead(f *os.File, n int) []byte {
	b := make([]byte, n)
	nRead, err := f.Read(b)
	handleErr(err)
	if nRead != n {
		panic("wrong byte count")
	}
	return b
}

func readInt(f *os.File) int {
	buf := safeRead(f, 4)
	return int(binary.BigEndian.Uint32(buf))
}

func readLabelFile(path string) Matrix {
	// for labels we initially apply one hot encoding
	f, err := os.Open(path)
	handleErr(err)
	readInt(f)
	nSamples := readInt(f)
	labels := safeRead(f, nSamples)
	result := NewMatrix(nSamples, 10, "zeros")
	for i := 0; i < nSamples; i++ {
		j := int(labels[i])
		result[i][j] = 1
	}
	err = f.Close()
	handleErr(err)
	return result
}

func readImageFile(path string) Matrix {
	f, err := os.Open(path)
	handleErr(err)
	readInt(f)
	nSamples := readInt(f)
	nRows := readInt(f)
	nCols := readInt(f)
	// we will read about 45 Mb
	// so there in no need to separate reads
	labels := safeRead(f, nSamples*nRows*nCols)
	result := NewMatrix(nSamples, nRows*nCols, "empty")
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nRows*nCols; j++ {
			result[i][j] = float64(labels[i*nRows*nCols+j]) / 255.0
		}
	}
	err = f.Close()
	handleErr(err)
	return result
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
