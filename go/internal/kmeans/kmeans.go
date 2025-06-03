package kmeans

import (
	"math"
	"math/rand"
)

// KMeans clusters RGB pixels into `k` clusters for `iterations`.
func KMeans(pixels [][3]uint8, k int, iterations int) [][3]int {
	centroids := initCenters(pixels, k)

	assignments := make([]int, len(pixels))

	for iter := 0; iter < iterations; iter++ {
		// Assign pixels to nearest centroid
		for i, px := range pixels {
			assignments[i] = nearestCenter(px, centroids)
		}

		// Recompute centroids
		newCentroids := make([][3]float64, k)
		counts := make([]int, k)

		for i, a := range assignments {
			for c := 0; c < 3; c++ {
				newCentroids[a][c] += float64(pixels[i][c])
			}
			counts[a]++
		}

		// Average centroids
		for i := range newCentroids {
			if counts[i] > 0 {
				for c := 0; c < 3; c++ {
					newCentroids[i][c] /= float64(counts[i])
				}
			}
		}

		// Convert to int and check for convergence
		converged := true
		for i := 0; i < k; i++ {
			for c := 0; c < 3; c++ {
				if math.Abs(centroids[i][c]-newCentroids[i][c]) > 1.0 {
					converged = false
					break
				}
			}
		}

		for i := range centroids {
			centroids[i] = newCentroids[i]
		}

		if converged {
			break
		}
	}

	// Convert to int
	result := make([][3]int, k)
	for i, c := range centroids {
		result[i] = [3]int{int(c[0]), int(c[1]), int(c[2])}
	}
	return result
}

func initCenters(pixels [][3]uint8, k int) [][3]float64 {
	rand.Shuffle(len(pixels), func(i, j int) {
		pixels[i], pixels[j] = pixels[j], pixels[i]
	})
	centers := make([][3]float64, 0, k)
	for i := 0; i < k; i++ {
		p := pixels[i]
		centers = append(centers, [3]float64{
			float64(p[0]),
			float64(p[1]),
			float64(p[2]),
		})
	}
	return centers
}

func nearestCenter(px [3]uint8, centers [][3]float64) int {
	best := 0
	minDist := math.MaxFloat64
	for i, c := range centers {
		d := 0.0
		for j := 0; j < 3; j++ {
			delta := float64(px[j]) - c[j]
			d += delta * delta
		}
		if d < minDist {
			minDist = d
			best = i
		}
	}
	return best
}

func NearestCenter(px [3]uint8, centers [][3]int) int {
	best := 0
	minDist := math.MaxFloat64
	for i, c := range centers {
		d := 0.0
		for j := 0; j < 3; j++ {
			delta := float64(px[j]) - float64(c[j])
			d += delta * delta
		}
		if d < minDist {
			minDist = d
			best = i
		}
	}
	return best
}
