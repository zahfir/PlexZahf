package imageproc

import (
	"math/rand/v2"
	"plexzahf/internal/kmeans"
	"sort"

	"github.com/lucasb-eyer/go-colorful"
)

type FrameAnalysis struct {
	Colors      [][3]int
	Proportions []float64
	Hues        []float64
	Saturations []float64
}

// sortByProportions sorts the color data arrays by proportion in descending order
func sortByProportions(colors [][3]int, proportions, hues, saturations []float64) ([][3]int, []float64, []float64, []float64) {
	// Create index slice to track original positions
	indices := make([]int, len(proportions))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices by proportion in descending order
	sort.Slice(indices, func(i, j int) bool {
		return proportions[indices[i]] > proportions[indices[j]]
	})

	// Create new sorted arrays
	sortedColors := make([][3]int, len(colors))
	sortedProportions := make([]float64, len(proportions))
	sortedHues := make([]float64, len(hues))
	sortedSaturations := make([]float64, len(saturations))

	// Rearrange all arrays using the sorted indices
	for newIdx, oldIdx := range indices {
		sortedColors[newIdx] = colors[oldIdx]
		sortedProportions[newIdx] = proportions[oldIdx]
		sortedHues[newIdx] = hues[oldIdx]
		sortedSaturations[newIdx] = saturations[oldIdx]
	}

	return sortedColors, sortedProportions, sortedHues, sortedSaturations
}

func GetTopColors(frame []byte, width, height, sampleSize, k int) FrameAnalysis {
	totalPixels := width * height
	if sampleSize > totalPixels {
		sampleSize = totalPixels
	}

	// Sample random pixel indices
	sample := make([][3]uint8, 0, sampleSize)
	for i := 0; i < sampleSize; i++ {
		idx := rand.IntN(totalPixels)
		base := idx * 3
		sample = append(sample, [3]uint8{
			frame[base],
			frame[base+1],
			frame[base+2],
		})
	}

	// Run KMeans clustering
	clusters := kmeans.KMeans(sample, k, 10)

	// Count occurrences
	counts := make([]int, k)
	for _, px := range sample {
		idx := kmeans.NearestCenter(px, clusters)
		counts[idx]++
	}

	total := float64(len(sample))
	proportions := make([]float64, k)
	hues := make([]float64, k)
	saturations := make([]float64, k)

	for i, c := range clusters {
		proportions[i] = float64(counts[i]) / total

		col := colorful.Color{
			R: float64(c[0]) / 255.0,
			G: float64(c[1]) / 255.0,
			B: float64(c[2]) / 255.0,
		}
		h, s, _ := col.Hsl()
		hues[i] = float64(transformH(h))        // Convert to OpenCV hue range
		saturations[i] = float64(transformS(s)) // Convert to OpenCV saturation range
	}

	// Sort all arrays by proportion (descending)
	sortedColors, sortedProportions, sortedHues, sortedSaturations :=
		sortByProportions(clusters, proportions, hues, saturations)

	return FrameAnalysis{
		Colors:      sortedColors,
		Proportions: sortedProportions,
		Hues:        sortedHues,
		Saturations: sortedSaturations,
	}
}

// transformH converts a standard HSL hue value (0-360 degrees) to OpenCV HSV hue range (0-179)
func transformH(hue float64) int {
	// Handle edge case of 360 degrees
	if hue >= 360.0 {
		hue = 0.0
	}

	// Map from 0-360 to 0-179 range
	return int(hue / 2.0)
}

// transformS converts a standard HSL saturation value (0-1) to OpenCV saturation range (0-255)
func transformS(saturation float64) int {
	// Clamp value between 0 and 1
	if saturation < 0.0 {
		saturation = 0.0
	} else if saturation > 1.0 {
		saturation = 1.0
	}

	// Map from 0-1 to 0-255 range
	return int(saturation * 255.0)
}
