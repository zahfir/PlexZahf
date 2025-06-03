package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"plexzahf/internal/ffmpeg"
	"plexzahf/internal/video"
	"syscall"
)

func main() {
	// Define command line flags
	videoPath := flag.String("video", "", "Path to video file (local or URL)")
	outputDir := flag.String("output", "./results", "Directory to save analysis results")
	startTime := flag.String("start", "", "Start time (format: HH:MM:SS)")
	endTime := flag.String("end", "", "End time (format: HH:MM:SS)")
	sampleRate := flag.Int("sample-rate", 5, "Process every Nth frame")
	pixelsPerFrame := flag.Int("pixels", 5000, "Number of random pixels to sample per frame")

	flag.Parse()

	// Validate required arguments
	if *videoPath == "" {
		fmt.Fprintf(os.Stderr, "Error: Video path is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Setup time range if specified
	var timeRange *ffmpeg.TimeRange
	if *startTime != "" {
		timeRange = &ffmpeg.TimeRange{
			Start: *startTime,
			End:   *endTime,
		}
	}

	// Create a context that can be canceled
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle termination signals
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signalChan
		log.Println("Received termination signal, shutting down...")
		cancel()
	}()

	// Create the frame processor
	processor := video.NewFrameProcessor(
		*videoPath,
		*outputDir,
		timeRange,
		*sampleRate,
		*pixelsPerFrame,
	)

	// Process the video
	log.Printf("Starting analysis of %s", *videoPath)
	if err := processor.ProcessFrames(ctx); err != nil {
		log.Fatalf("Error processing video: %v", err)
	}

	// Get the results
	results := processor.GetResults()
	log.Printf("Analysis complete. Processed %d frames.", len(results))
	if len(results) > 0 {
		first := results[0]
		log.Printf("First frame analysis:")
		log.Printf("  Frame number: %d", first.FrameNumber)
		log.Printf("  Timestamp: %.2fs", first.Timestamp)
	}
}
