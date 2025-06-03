package video

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"plexzahf/internal/ffmpeg"
)

// FrameResult stores analysis results for a processed frame
type FrameResult struct {
	FrameNumber int          `json:"frame_number"`
	Timestamp   float64      `json:"timestamp"`
	AvgColor    [3]uint8     `json:"avg_color,omitempty"`
	Brightness  float64      `json:"brightness,omitempty"`
	TopColors   []ColorCount `json:"top_colors,omitempty"`
}

// ColorCount represents a color and its frequency in the frame
type ColorCount struct {
	Color [3]uint8 `json:"color"`
	Count int      `json:"count"`
}

// FrameBuffer represents a reusable buffer for reading frames
type FrameBuffer struct {
	data []byte
}

// FrameBufferPool manages a pool of frame buffers to minimize GC pressure
type FrameBufferPool struct {
	pool chan *FrameBuffer
}

// NewFrameBufferPool creates a new buffer pool with the specified size
func NewFrameBufferPool(bufferSize, poolSize int) *FrameBufferPool {
	pool := make(chan *FrameBuffer, poolSize)
	for i := 0; i < poolSize; i++ {
		pool <- &FrameBuffer{data: make([]byte, bufferSize)}
	}
	return &FrameBufferPool{pool: pool}
}

// Get retrieves a buffer from the pool
func (p *FrameBufferPool) Get() *FrameBuffer {
	return <-p.pool
}

// Put returns a buffer to the pool
func (p *FrameBufferPool) Put(buffer *FrameBuffer) {
	p.pool <- buffer
}

// FrameProcessor handles the video processing pipeline
type FrameProcessor struct {
	// Configuration
	VideoURL           string
	OutputDir          string
	TimeRange          *ffmpeg.TimeRange
	SampleEveryNFrames int
	PixelsPerFrame     int

	// Internal state
	width        int
	height       int
	framerate    float64
	frameSize    int
	isHDR        bool
	results      []FrameResult
	resultsMutex sync.Mutex
}

// NewFrameProcessor creates a new frame processor instance
func NewFrameProcessor(videoURL, outputDir string, timeRange *ffmpeg.TimeRange, sampleEveryNFrames, pixelsPerFrame int) *FrameProcessor {
	return &FrameProcessor{
		VideoURL:           videoURL,
		OutputDir:          outputDir,
		TimeRange:          timeRange,
		SampleEveryNFrames: sampleEveryNFrames,
		PixelsPerFrame:     pixelsPerFrame,
		results:            make([]FrameResult, 0),
	}
}

// readFullFrame reads a complete frame, handling partial reads
// func readFullFrame(reader io.Reader, buffer []byte, ctx context.Context) (int, error) {
// 	totalRead := 0
// 	for totalRead < len(buffer) {
// 		// Check if context is canceled
// 		select {
// 		case <-ctx.Done():
// 			return totalRead, ctx.Err()
// 		default:
// 		}

// 		n, err := reader.Read(buffer[totalRead:])
// 		if err != nil {
// 			return totalRead, err
// 		}
// 		if n == 0 {
// 			// No data read but no error - this is unusual
// 			return totalRead, io.ErrUnexpectedEOF
// 		}
// 		totalRead += n
// 	}
// 	return totalRead, nil
// }

// saveDebugImage saves a frame as a PNG for debugging
func (fp *FrameProcessor) saveDebugImage(frameNum int, frameData []byte) error {
	img := image.NewRGBA(image.Rect(0, 0, fp.width, fp.height))

	// Copy the raw frame data to the image
	for y := 0; y < fp.height; y++ {
		for x := 0; x < fp.width; x++ {
			idx := (y*fp.width + x) * 3
			img.Set(x, y, color.RGBA{
				R: frameData[idx],
				G: frameData[idx+1],
				B: frameData[idx+2],
				A: 255,
			})
		}
	}

	// Create the output file
	name := fmt.Sprintf("frame_%04d.png", frameNum)
	outputPath := filepath.Join(fp.OutputDir, name)

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating debug image file: %w", err)
	}
	defer file.Close()

	// Write the image
	return png.Encode(file, img)
}

// analyzeFrame performs color analysis on a frame
func (fp *FrameProcessor) analyzeFrame(frameData []byte) (FrameResult, error) {
	result := FrameResult{}

	// Calculate average RGB
	var rSum, gSum, bSum uint64
	for i := 0; i < len(frameData); i += 3 {
		rSum += uint64(frameData[i])
		gSum += uint64(frameData[i+1])
		bSum += uint64(frameData[i+2])
	}

	pixelCount := len(frameData) / 3
	result.AvgColor = [3]uint8{
		uint8(rSum / uint64(pixelCount)),
		uint8(gSum / uint64(pixelCount)),
		uint8(bSum / uint64(pixelCount)),
	}

	// Calculate brightness (simple average of RGB)
	result.Brightness = float64(rSum+gSum+bSum) / (float64(pixelCount) * 3.0 * 255.0)

	// For more advanced color analysis, you could:
	// 1. Sample random pixels instead of processing all
	// 2. Use k-means clustering to find dominant colors
	// 3. Create a color histogram

	return result, nil
}

// Replace your current readPNGFrame function with this:
func readPNGFrame(reader io.Reader) (image.Image, error) {
	// PNG signature to look for
	pngSignature := []byte{137, 80, 78, 71, 13, 10, 26, 10}

	// Use a buffered reader for better performance
	bufferedReader := bufio.NewReaderSize(reader, 1024*1024) // 1MB buffer

	// Look for PNG signature
	for {
		// Read and verify the PNG signature
		signature, err := bufferedReader.Peek(len(pngSignature))
		if err != nil {
			return nil, err
		}

		if bytes.Equal(signature, pngSignature) {
			// Found PNG signature, decode the image
			img, err := png.Decode(bufferedReader)
			return img, err
		}

		// Not a PNG signature, discard a byte and try again
		_, err = bufferedReader.Discard(1)
		if err != nil {
			return nil, err
		}
	}
}

// ProcessFrames extracts and processes frames from the video
func (fp *FrameProcessor) ProcessFrames(ctx context.Context) error {
	// Add at the beginning of ProcessFrames:
	_, erro := os.Stat(fp.VideoURL)
	if erro != nil {
		return fmt.Errorf("cannot access video file '%s': %w", fp.VideoURL, erro)
	}
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(fp.OutputDir, 0755); err != nil {
		return fmt.Errorf("error creating output directory: %w", err)
	}

	startTime := time.Now()

	// Get video information
	log.Println("Getting video information...")
	var err error
	fp.width, fp.height, fp.framerate, err = ffmpeg.GetVideoInfo(ctx, fp.VideoURL)
	if err != nil {
		return fmt.Errorf("error getting video info: %w", err)
	}

	log.Printf("Video dimensions: %dx%d, Frame rate: %.3ffps\n", fp.width, fp.height, fp.framerate)

	// Check if video is HDR
	fp.isHDR, err = ffmpeg.CheckIfHDR(ctx, fp.VideoURL)
	if err != nil {
		log.Printf("Warning: Error checking HDR status: %v", err)
	}
	if fp.isHDR {
		log.Println("HDR content detected, using HDR to SDR conversion")
	}

	// Calculate frame size in bytes (RGB24 format)
	fp.frameSize = fp.width * fp.height * 3

	// Create FFmpeg options
	opts := ffmpeg.FFmpegFrameOptions{
		FFmpegOptions: ffmpeg.FFmpegOptions{
			URL:     fp.VideoURL,
			Width:   fp.width,
			Height:  fp.height,
			FPS:     int(fp.framerate),
			Timeout: 30 * time.Second,
		},
		IsHDR:              fp.isHDR,
		SampleEveryNFrames: fp.SampleEveryNFrames,
		TimeRange:          fp.TimeRange,
	}

	// Start the FFmpeg process
	log.Printf("Starting FFmpeg process to extract every %dth frame...", fp.SampleEveryNFrames)
	cmd, stdout, stderr, err := ffmpeg.CreateFrameExtractionProcess(ctx, opts)
	// In ProcessFrames function, add this after the FFmpeg process starts:
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("FFmpeg: %s", scanner.Text())
		}

		// Add this line to check for scanner errors
		if err := scanner.Err(); err != nil {
			log.Printf("Error reading FFmpeg output: %v", err)
		}
	}()
	if err != nil {
		return fmt.Errorf("error creating FFmpeg process: %w", err)
	}

	// Start monitoring stderr in a goroutine
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("FFmpeg: %s", scanner.Text())
		}
		// Add this line to report scanner errors
		if err := scanner.Err(); err != nil {
			log.Printf("Error reading FFmpeg output: %v", err)
		}
	}()

	// Create a context that will be canceled when this function returns
	processCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Handle process termination
	processErrCh := make(chan error, 1)
	go func() {
		processErrCh <- cmd.Wait()
	}()

	// Create buffer pool for frame reading (reuse buffers to reduce GC pressure)
	bufferPool := NewFrameBufferPool(fp.frameSize, 3) // 3 buffers in the pool

	// Process frames
	frameCount := 0
	resultChan := make(chan FrameResult, 10) // Channel for collecting results

	// Start result collector
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for result := range resultChan {
			fp.resultsMutex.Lock()
			fp.results = append(fp.results, result)
			fp.resultsMutex.Unlock()
		}
	}()

	log.Println("Processing frames...")

	// Main frame processing loop
	var readErrors int
	for {
		// Check if process has ended
		select {
		case err := <-processErrCh:
			if err != nil {
				log.Printf("FFmpeg process ended with error: %v", err)
			} else {
				log.Println("FFmpeg process completed successfully")
			}
			goto ProcessingComplete
		case <-processCtx.Done():
			log.Println("Processing canceled")
			return processCtx.Err()
		default:
			// Continue processing
		}

		// Use PNG decoder with error handling
		img, err := readPNGFrame(stdout)
		if err != nil {
			if err == io.EOF {
				log.Println("End of stream reached")
				break
			}

			readErrors++
			log.Printf("Error reading PNG frame: %v (error #%d)", err, readErrors)

			if readErrors > 5 {
				log.Println("Too many read errors, checking if process is still running...")
				select {
				case err := <-processErrCh:
					log.Printf("Process ended: %v", err)
					goto ProcessingComplete
				default:
					log.Println("Process still running, continuing...")
				}
			}

			// Sleep briefly to avoid tight loop on errors
			time.Sleep(100 * time.Millisecond)
			continue
		}

		// Reset error counter on successful read
		readErrors = 0

		// Convert image.Image to []byte for your existing analysis
		bounds := img.Bounds()
		width, height := bounds.Max.X, bounds.Max.Y
		frameBuffer := bufferPool.Get()

		// Copy pixel data to buffer
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x, y).RGBA()
				idx := (y*width + x) * 3
				frameBuffer.data[idx] = uint8(r >> 8)
				frameBuffer.data[idx+1] = uint8(g >> 8)
				frameBuffer.data[idx+2] = uint8(b >> 8)
			}
		}

		// (Removed impossible err != nil check here)

		// Process the frame in a separate goroutine to maintain reading speed
		wg.Add(1)
		go func(frameNum int, buffer *FrameBuffer) {
			defer wg.Done()
			defer bufferPool.Put(buffer) // Return buffer to pool when done

			// Calculate frame details
			baseResult := FrameResult{
				FrameNumber: frameNum * fp.SampleEveryNFrames,
				Timestamp:   float64(frameNum*fp.SampleEveryNFrames) / fp.framerate,
			}

			// Analyze the frame
			frameAnalysis, err := fp.analyzeFrame(buffer.data)
			if err != nil {
				log.Printf("Error analyzing frame %d: %v", frameNum, err)
			} else {
				baseResult.AvgColor = frameAnalysis.AvgColor
				baseResult.Brightness = frameAnalysis.Brightness
				baseResult.TopColors = frameAnalysis.TopColors
			}

			// Save debug image periodically
			if frameNum%100 == 0 {
				log.Printf("Processed %d frames...", frameNum)

				// Clone the buffer for saving (buffer will be reused)
				dataCopy := make([]byte, len(buffer.data))
				copy(dataCopy, buffer.data)

				if err := fp.saveDebugImage(frameNum, dataCopy); err != nil {
					log.Printf("Error saving debug image: %v", err)
				}
			}

			// Send result to collector
			resultChan <- baseResult
		}(frameCount, frameBuffer)

		frameCount++
	}

ProcessingComplete:
	// Close the result channel and wait for all goroutines to finish
	close(resultChan)
	wg.Wait()

	// Save results
	resultsFile := filepath.Join(fp.OutputDir, "analysis_results.json")

	fp.resultsMutex.Lock()
	data, err := json.MarshalIndent(fp.results, "", "  ")
	fp.resultsMutex.Unlock()

	if err != nil {
		return fmt.Errorf("error marshaling results: %w", err)
	}

	if err := os.WriteFile(resultsFile, data, 0644); err != nil {
		return fmt.Errorf("error writing results file: %w", err)
	}

	elapsedTime := time.Since(startTime).Seconds()
	log.Printf("Processing complete! Analyzed %d frames in %.2f seconds", frameCount, elapsedTime)
	log.Printf("Results saved to %s", resultsFile)

	return nil
}

// GetResults returns a copy of the current analysis results
func (fp *FrameProcessor) GetResults() []FrameResult {
	fp.resultsMutex.Lock()
	defer fp.resultsMutex.Unlock()

	// Return a copy to avoid concurrent access issues
	resultsCopy := make([]FrameResult, len(fp.results))
	copy(resultsCopy, fp.results)
	return resultsCopy
}
