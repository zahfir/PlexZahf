package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"plexzahf/internal/ffmpeg"
	"strconv"
)

type Result struct {
	DataSize int    `json:"data_size"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	FPS      int    `json:"fps"`
	Data     string `json:"data"` // Base64 encoded RGB data
}

func main() {
	// Get arguments
	if len(os.Args) < 6 {
		fmt.Println("Usage: ffmpeg_tool.exe <video_path> <width> <height> <fps> <bitrate>")
		os.Exit(1)
	}

	videoPath := os.Args[1]
	width, _ := strconv.Atoi(os.Args[2])
	height, _ := strconv.Atoi(os.Args[3])
	fps, _ := strconv.Atoi(os.Args[4])
	bitrate := os.Args[5]

	ctx := context.Background()
	// No need for cancel function with background context

	// Set up FFmpeg options
	opts := ffmpeg.FFmpegOptions{
		URL:     videoPath,
		Width:   width,
		Height:  height,
		FPS:     fps,
		Bitrate: bitrate,
		Timeout: 0, // 0 indicates no timeout
	}

	// Process the video
	cmd, stdout, stderrBuf, err := ffmpeg.CreateFFmpegProcess(ctx, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating FFmpeg process: %v\n", err)
		os.Exit(1)
	}

	data, err := ffmpeg.ReadStreamToMemory(ctx, cmd, stdout, stderrBuf, 8192, 100*1024*1024)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading stream: %v\n", err)
		os.Exit(1)
	}

	// Encode data as base64 for JSON
	encoded := base64.StdEncoding.EncodeToString(data)

	// Create result struct
	result := Result{
		DataSize: len(data),
		Width:    width,
		Height:   height,
		FPS:      fps,
		Data:     encoded,
	}

	// Output as JSON
	jsonResult, err := json.Marshal(result)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(jsonResult))
}
