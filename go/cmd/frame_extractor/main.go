// main.go
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"plexzahf/internal/imageproc"
	"strconv"
	"strings"
	"sync"
	"time"
)

// VideoProcessor handles the video processing pipeline
type VideoProcessor struct {
	// Configuration
	VideoURL           string
	OutputDir          string
	TimeRange          *TimeRange
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

// TimeRange represents a start and end time for video processing
type TimeRange struct {
	Start string
	End   string
}

// FrameResult stores analysis results for a processed frame
type FrameResult struct {
	FrameNumber int       `json:"frame_number"`
	Timestamp   float64   `json:"timestamp"`
	Colors      [][3]int  `json:"colors"`
	Proportions []float64 `json:"proportions"`
	Hues        []float64 `json:"hues"`
	Saturations []float64 `json:"saturations"`
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

// checkIfHDR determines if the video has HDR metadata
func (vp *VideoProcessor) checkIfHDR() (bool, error) {
	args := []string{
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=color_transfer,master_display,color_space",
		"-of", "json",
		vp.VideoURL,
	}

	cmd := exec.Command("ffprobe", args...)
	output, err := cmd.Output()
	if err != nil {
		return false, fmt.Errorf("ffprobe error: %w", err)
	}

	var data map[string]interface{}
	if err := json.Unmarshal(output, &data); err != nil {
		return false, fmt.Errorf("error parsing ffprobe output: %w", err)
	}

	streams, ok := data["streams"].([]interface{})
	if !ok || len(streams) == 0 {
		return false, nil
	}

	stream, ok := streams[0].(map[string]interface{})
	if !ok {
		return false, nil
	}

	// Check common HDR indicators
	if transfer, ok := stream["color_transfer"].(string); ok {
		transferLower := strings.ToLower(transfer)
		if strings.Contains(transferLower, "smpte2084") || strings.Contains(transferLower, "arib-std-b67") {
			return true, nil
		}
	}

	if colorspace, ok := stream["color_space"].(string); ok {
		if strings.Contains(strings.ToLower(colorspace), "bt2020") {
			return true, nil
		}
	}

	_, hasMasterDisplay := stream["master_display"]

	return hasMasterDisplay, nil
}

// getVideoInfo extracts width, height, and framerate from the video
func (vp *VideoProcessor) getVideoInfo() error {
	args := []string{
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,avg_frame_rate",
		"-of", "json",
		vp.VideoURL,
	}

	cmd := exec.Command("ffprobe", args...)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("ffprobe error: %w", err)
	}

	var data map[string]interface{}
	if err := json.Unmarshal(output, &data); err != nil {
		return fmt.Errorf("error parsing ffprobe output: %w", err)
	}

	streams, ok := data["streams"].([]interface{})
	if !ok || len(streams) == 0 {
		return fmt.Errorf("no video streams found")
	}

	stream, ok := streams[0].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid stream data")
	}

	// Extract width and height
	if width, ok := stream["width"].(float64); ok {
		vp.width = int(width)
	} else {
		return fmt.Errorf("invalid width")
	}

	if height, ok := stream["height"].(float64); ok {
		vp.height = int(height)
	} else {
		return fmt.Errorf("invalid height")
	}

	// Parse framerate (e.g., "24000/1001" -> ~23.976)
	if framerateStr, ok := stream["avg_frame_rate"].(string); ok {
		if strings.Contains(framerateStr, "/") {
			parts := strings.Split(framerateStr, "/")
			if len(parts) == 2 {
				num, err1 := strconv.ParseFloat(parts[0], 64)
				den, err2 := strconv.ParseFloat(parts[1], 64)
				if err1 == nil && err2 == nil && den != 0 {
					vp.framerate = num / den
				} else {
					return fmt.Errorf("invalid framerate format")
				}
			}
		} else {
			vp.framerate, err = strconv.ParseFloat(framerateStr, 64)
			if err != nil {
				return fmt.Errorf("invalid framerate: %w", err)
			}
		}
	} else {
		return fmt.Errorf("invalid framerate data")
	}

	// Calculate frame size in bytes (RGB24 format)
	vp.frameSize = vp.width * vp.height * 3

	return nil
}

// parseTimeString converts time strings like "00:05:10" to seconds
func parseTimeString(timeStr string) (float64, error) {
	// Handle simple seconds format
	if seconds, err := strconv.ParseFloat(timeStr, 64); err == nil {
		return seconds, nil
	}

	// Handle "HH:MM:SS" format
	parts := strings.Split(timeStr, ":")
	if len(parts) == 3 {
		h, errH := strconv.ParseFloat(parts[0], 64)
		m, errM := strconv.ParseFloat(parts[1], 64)
		s, errS := strconv.ParseFloat(parts[2], 64)

		if errH == nil && errM == nil && errS == nil {
			return h*3600 + m*60 + s, nil
		}
	}

	return 0, fmt.Errorf("invalid time format: %s", timeStr)
}

func readFullFrameSafe(reader io.Reader, buffer []byte, ctx context.Context) (int, error) {
	totalRead := 0
	chunkSize := 16 * 1024 // Read in 16KB chunks

	for totalRead < len(buffer) {
		select {
		case <-ctx.Done():
			log.Printf("[WARN] Context done while reading frame: read %d/%d", totalRead, len(buffer))
			return totalRead, ctx.Err()
		default:
		}

		readLen := chunkSize
		if remaining := len(buffer) - totalRead; remaining < chunkSize {
			readLen = remaining
		}

		n, err := reader.Read(buffer[totalRead : totalRead+readLen])
		if n > 0 {
			totalRead += n
		}

		if err != nil {
			if err == io.EOF {
				log.Printf("[INFO] EOF reached mid-frame: %d/%d bytes", totalRead, len(buffer))
			} else {
				log.Printf("[ERROR] Read error after %d bytes: %v", totalRead, err)
			}
			return totalRead, err
		}
	}

	return totalRead, nil
}

// NewVideoProcessor creates a new video processor instance
func NewVideoProcessor(videoURL, outputDir string, timeRange *TimeRange, sampleEveryNFrames, pixelsPerFrame int) *VideoProcessor {
	return &VideoProcessor{
		VideoURL:           videoURL,
		OutputDir:          outputDir,
		TimeRange:          timeRange,
		SampleEveryNFrames: sampleEveryNFrames,
		PixelsPerFrame:     pixelsPerFrame,
		results:            make([]FrameResult, 0),
	}
}

func (vp *VideoProcessor) createGPUProcessedTempFile() (string, error) {
	tempDir := os.TempDir()
	tempFile := filepath.Join(tempDir, fmt.Sprintf("ffmpeg_temp_%d.hevc", time.Now().UnixNano()))

	args := []string{
		"-hwaccel", "cuda",
		"-hwaccel_output_format", "cuda",
		"-err_detect", "aggressive",
		"-xerror",
		"-loglevel", "verbose",
		"-reconnect", "1",
		"-reconnect_streamed", "1",
		"-reconnect_on_network_error", "1",
		"-reconnect_on_http_error", "4xx,5xx",
		"-reconnect_delay_max", "120",
		"-reconnect_max_retries", "10",
		"-timeout", "10000000",
		"-avioflags", "direct",
	}

	if vp.TimeRange != nil {
		if vp.TimeRange.Start != "" {
			args = append(args, "-ss", vp.TimeRange.Start)
		}
		if vp.TimeRange.End != "" {
			startTime, endTime := 0.0, 0.0
			if s, err := parseTimeString(vp.TimeRange.Start); err == nil {
				startTime = s
			}
			if e, err := parseTimeString(vp.TimeRange.End); err == nil {
				endTime = e
			}
			if endTime > startTime {
				duration := endTime - startTime
				args = append(args, "-t", fmt.Sprintf("%.3f", duration))
			}
		}
	}

	args = append(args, "-i", vp.VideoURL)
	originalWidth, originalHeight := vp.width, vp.height
	targetHeight := 240
	targetWidth := int(float64(targetHeight) * float64(originalWidth) / float64(originalHeight))
	targetWidth = targetWidth - (targetWidth % 2)
	vp.width = targetWidth
	vp.height = targetHeight
	vp.frameSize = vp.width * vp.height * 3

	log.Printf("Scaling video from %dx%d to %dx%d using GPU acceleration",
		originalWidth, originalHeight, targetWidth, targetHeight)

	filterComplex := fmt.Sprintf(
		"fps=%d,scale_cuda=%d:%d",
		vp.SampleEveryNFrames, targetWidth, targetHeight,
	)

	args = append(args,
		"-vf", filterComplex,
		"-c:v", "hevc_nvenc",
		"-an",
		"-progress", "pipe:1",
		"-flush_packets", "1",
		"-f", "hevc",
		"-fflags", "+autobsf",
		tempFile,
	)
	log.Printf("[DEBUG] FFmpeg temp output: %s", tempFile)
	log.Printf("[DEBUG] FFmpeg args: %v", args)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("error creating stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return "", fmt.Errorf("error creating stderr pipe: %w", err)
	}

	go func() {
		scanner := bufio.NewScanner(stdout)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)
		for scanner.Scan() {
			log.Printf("FFmpeg Progress: %s", scanner.Text())
		}
		if err := scanner.Err(); err != nil && err != io.EOF {
			log.Printf("Error reading FFmpeg stdout: %v", err)
		}
		log.Println("FFmpeg stdout closed")
		stdout.Close()
	}()

	go func() {
		scanner := bufio.NewScanner(stderr)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)
		for scanner.Scan() {
			log.Printf("FFmpeg GPU Processing: %s", scanner.Text())
		}
		if err := scanner.Err(); err != nil && err != io.EOF {
			log.Printf("Error reading FFmpeg stderr: %v", err)
		}
		stderr.Close()
	}()

	log.Println("Starting GPU-accelerated preprocessing...")
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("error starting FFmpeg preprocessing: %w", err)
	}

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case err := <-done:
			if err != nil {
				if ctx.Err() == context.DeadlineExceeded {
					return "", fmt.Errorf("FFmpeg process timed out after 60 minutes")
				}
				return "", fmt.Errorf("error in FFmpeg preprocessing: %w", err)
			}
			fileInfo, err := os.Stat(tempFile)
			if err != nil {
				log.Printf("Warning: Unable to get temporary file size: %v", err)
			} else {
				fileSizeBytes := fileInfo.Size()
				fileSizeGB := float64(fileSizeBytes) / (1024 * 1024 * 1024)
				log.Printf("Temporary file size: %.6f GB", fileSizeGB)
			}
			log.Printf("Created GPU-processed file: %s", tempFile)
			return tempFile, nil
		case <-ticker.C:
			// Check if the file size has reached the limit
			fileInfo, err := os.Stat(tempFile)
			if err == nil && fileInfo.Size() >= 10000000000 { // 10 GB
				log.Printf("File reached max size, terminating FFmpeg")
				if err := cmd.Process.Kill(); err != nil {
					log.Printf("Warning: failed to kill FFmpeg process: %v", err)
				}
				fileInfo, err := os.Stat(tempFile)
				if err != nil {
					log.Printf("Warning: Unable to get temporary file size: %v", err)
				} else {
					fileSizeBytes := fileInfo.Size()
					fileSizeGB := float64(fileSizeBytes) / (1024 * 1024 * 1024)
					log.Printf("Temporary file size: %.6f GB", fileSizeGB)
				}
				log.Printf("Created GPU-processed file: %s", tempFile)
				return tempFile, nil
			}
		case <-ctx.Done():
			return "", fmt.Errorf("FFmpeg process timed out after 60 minutes")
		}
	}
}

// Updated version of ProcessFrames that uses two-step processing
func (vp *VideoProcessor) ProcessFramesWithGPU(ctx context.Context) error {
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(vp.OutputDir, 0755); err != nil {
		return fmt.Errorf("error creating output directory: %w", err)
	}

	startTime := time.Now()

	log.Println("Getting video information...")
	if err := vp.getVideoInfo(); err != nil {
		return fmt.Errorf("error getting video info: %w", err)
	}

	log.Printf("Video dimensions: %dx%d, Frame rate: %.3ffps\n", vp.width, vp.height, vp.framerate)

	var err error
	vp.isHDR, err = vp.checkIfHDR()
	if err != nil {
		log.Printf("Warning: Error checking HDR status: %v", err)
	}
	if vp.isHDR {
		log.Println("HDR content detected, using HDR to SDR conversion")
	}

	// Step 1: Create temporary file with GPU acceleration
	tempFile, err := vp.createGPUProcessedTempFile()
	if err != nil {
		return err
	}
	defer os.Remove(tempFile) // Clean up temp file

	// Step 2: Process compressed frames from the temporary file
	// Build a simpler FFmpeg command to read from the temp file
	args := []string{
		"-hwaccel", "cuda",
		"-i", tempFile,
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"-",
	}

	cmd := exec.Command("ffmpeg", args...)

	// The rest of your ProcessFrames method remains the same
	// Create pipes for stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("error creating stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("error creating stderr pipe: %w", err)
	}

	// Start monitoring stderr in a goroutine
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("FFmpeg: %s", scanner.Text())
		}
	}()

	// Start the FFmpeg process
	log.Println("Starting frame extraction from processed file...")
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("error starting FFmpeg: %w", err)
	}

	// From here, use your existing code for the frame processing pipeline
	// Create a context that will be canceled when this function returns
	processCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Handle process termination
	processErrCh := make(chan error, 1)
	go func() {
		processErrCh <- cmd.Wait()
	}()

	// Create buffer pool for frame reading (reuse buffers to reduce GC pressure)
	bufferPool := NewFrameBufferPool(vp.frameSize, 3) // 3 buffers in the pool

	// Process frames
	frameCount := 0
	resultChan := make(chan FrameResult, 10) // Channel for collecting results

	// Start result collector
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for result := range resultChan {
			vp.resultsMutex.Lock()
			vp.results = append(vp.results, result)
			vp.resultsMutex.Unlock()
		}
	}()

	log.Println("Processing frames...")

	// Main frame processing loop
	for {
		// Check if process has ended
		select {
		case err := <-processErrCh:
			if err != nil && err != context.Canceled {
				close(resultChan)
				return fmt.Errorf("FFmpeg process error: %w", err)
			}
			log.Println("FFmpeg process completed")
			goto ProcessingComplete
		case <-processCtx.Done():
			close(resultChan)
			log.Println("Processing canceled")
			return processCtx.Err()
		default:
		}

		// Get a buffer from the pool
		frameBuffer := bufferPool.Get()

		// Read a complete frame with timeout
		readCtx, readCancel := context.WithTimeout(processCtx, 30*time.Second)
		// start := time.Now()
		bytesRead, err := readFullFrameSafe(stdout, frameBuffer.data, readCtx)
		// log.Printf("Read frame %d in %v", frameCount, time.Since(start))

		readCancel()

		if err != nil {
			bufferPool.Put(frameBuffer) // Return buffer to pool
			if err == io.EOF {
				log.Println("End of stream reached")
				break
			} else if err == context.DeadlineExceeded {
				log.Println("Timeout reading frame - stream may be stalled")
				break
			}
			close(resultChan)
			return fmt.Errorf("error reading frame: %w", err)
		}

		if bytesRead < vp.frameSize {
			bufferPool.Put(frameBuffer) // Return buffer to pool
			log.Printf("Incomplete frame (%d bytes). End of stream.", bytesRead)
			break
		}

		// Process the frame in a separate goroutine to maintain reading speed
		wg.Add(1)
		go func(frameNum int, buffer *FrameBuffer) {
			defer wg.Done()
			defer bufferPool.Put(buffer) // Return buffer to pool when done

			// Process frame using existing code
			analysis := imageproc.GetTopColors(buffer.data, vp.width, vp.height, vp.PixelsPerFrame, int(vp.framerate))

			// Calculate frame details
			result := FrameResult{
				FrameNumber: frameNum * vp.SampleEveryNFrames,
				Timestamp:   float64(frameNum*vp.SampleEveryNFrames) / vp.framerate,
				Colors:      analysis.Colors,
				Proportions: analysis.Proportions,
				Hues:        analysis.Hues,
				Saturations: analysis.Saturations,
			}

			// Send result to collector
			resultChan <- result
		}(frameCount, frameBuffer)

		frameCount++
	}

ProcessingComplete:
	// Close the result channel and wait for all goroutines to finish
	close(resultChan)
	wg.Wait()

	// Save results
	resultsFile := filepath.Join(vp.OutputDir, "analysis_results.json")

	vp.resultsMutex.Lock()
	data, err := json.MarshalIndent(vp.results, "", "  ")
	vp.resultsMutex.Unlock()

	if err != nil {
		return fmt.Errorf("error marshaling results: %w", err)
	}

	if err := os.WriteFile(resultsFile, data, 0644); err != nil {
		return fmt.Errorf("error writing results file: %w", err)
	}

	elapsedTime := time.Since(startTime).Seconds()
	log.Printf("Processing complete! Processed file and %d frames in %.2f seconds", frameCount, elapsedTime)
	log.Printf("Results saved to %s", resultsFile)

	return nil
}

func main() {
	// Define command line flags
	videoPath := flag.String("video", "", "Path to video file (local or URL)")
	outputDir := flag.String("output", "./results", "Directory to save analysis results")
	startTime := flag.String("start", "", "Start time (format: HH:MM:SS)")
	endTime := flag.String("end", "", "End time (format: HH:MM:SS)")
	sampleRate := flag.Int("sample-rate", 5, "Process every Nth frame")
	pixelsPerFrame := flag.Int("pixels", 5000, "Number of random pixels to sample per frame")

	flag.Parse()

	if *videoPath == "" {
		fmt.Fprintf(os.Stderr, "Error: Video path is required\n")
		flag.Usage()
		os.Exit(1)
	}

	var timeRange *TimeRange
	if *startTime != "" {
		timeRange = &TimeRange{
			Start: *startTime,
			End:   *endTime,
		}
	}

	processor := NewVideoProcessor(
		*videoPath,
		*outputDir,
		timeRange,
		*sampleRate,
		*pixelsPerFrame,
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := processor.ProcessFramesWithGPU(ctx); err != nil {
		log.Fatalf("Error processing video: %v", err)
	}

}
