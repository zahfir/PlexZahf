package worker

import (
	"plexzahf/internal/imageproc"
	"plexzahf/internal/video"
)

func WorkerPool(frames [][]video.Frame, width, height, sampleSize, k int, workers int) []imageproc.FrameAnalysis {
	jobs := make(chan []video.Frame, workers)
	results := make(chan []imageproc.FrameAnalysis, len(frames))

	for i := 0; i < workers; i++ {
		go func() {
			for batch := range jobs {
				var res []imageproc.FrameAnalysis
				for _, f := range batch {
					analysis := imageproc.GetTopColors(f, width, height, sampleSize, k)
					res = append(res, analysis)
				}
				results <- res
			}
		}()
	}

	for _, chunk := range frames {
		jobs <- chunk
	}
	close(jobs)

	var all []imageproc.FrameAnalysis
	for i := 0; i < len(frames); i++ {
		all = append(all, <-results...)
	}
	return all
}
