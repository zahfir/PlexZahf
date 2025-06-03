package ffmpeg

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strconv"
	"strings"
)

// TimeRange represents start and end time for video processing
type TimeRange struct {
	Start string
	End   string
}

// FFmpegFrameOptions extends FFmpegOptions with frame processing options
type FFmpegFrameOptions struct {
	FFmpegOptions
	IsHDR              bool
	SampleEveryNFrames int
	TimeRange          *TimeRange
}

// CheckIfHDR determines if the video has HDR metadata
func CheckIfHDR(ctx context.Context, videoURL string) (bool, error) {
	args := []string{
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=color_transfer,master_display,color_space",
		"-of", "json",
		videoURL,
	}

	cmd := exec.CommandContext(ctx, "ffprobe", args...)
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

// GetVideoInfo extracts width, height, and framerate from the video
func GetVideoInfo(ctx context.Context, videoURL string) (width, height int, framerate float64, err error) {
	args := []string{
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,avg_frame_rate",
		"-of", "json",
		videoURL,
	}

	cmd := exec.CommandContext(ctx, "ffprobe", args...)
	output, err := cmd.Output()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("ffprobe error: %w", err)
	}

	var data map[string]interface{}
	if err := json.Unmarshal(output, &data); err != nil {
		return 0, 0, 0, fmt.Errorf("error parsing ffprobe output: %w", err)
	}

	streams, ok := data["streams"].([]interface{})
	if !ok || len(streams) == 0 {
		return 0, 0, 0, fmt.Errorf("no video streams found")
	}

	stream, ok := streams[0].(map[string]interface{})
	if !ok {
		return 0, 0, 0, fmt.Errorf("invalid stream data")
	}

	// Extract width and height
	if w, ok := stream["width"].(float64); ok {
		width = int(w)
	} else {
		return 0, 0, 0, fmt.Errorf("invalid width")
	}

	if h, ok := stream["height"].(float64); ok {
		height = int(h)
	} else {
		return 0, 0, 0, fmt.Errorf("invalid height")
	}

	// Parse framerate (e.g., "24000/1001" -> ~23.976)
	if framerateStr, ok := stream["avg_frame_rate"].(string); ok {
		if strings.Contains(framerateStr, "/") {
			parts := strings.Split(framerateStr, "/")
			if len(parts) == 2 {
				num, err1 := strconv.ParseFloat(parts[0], 64)
				den, err2 := strconv.ParseFloat(parts[1], 64)
				if err1 == nil && err2 == nil && den != 0 {
					framerate = num / den
				} else {
					return width, height, 0, fmt.Errorf("invalid framerate format")
				}
			}
		} else {
			framerate, err = strconv.ParseFloat(framerateStr, 64)
			if err != nil {
				return width, height, 0, fmt.Errorf("invalid framerate: %w", err)
			}
		}
	} else {
		return width, height, 0, fmt.Errorf("invalid framerate data")
	}

	return width, height, framerate, nil
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

// CreateFrameExtractionProcess creates an ffmpeg process optimized for
// extracting frames with HDR support
func CreateFrameExtractionProcess(ctx context.Context, opts FFmpegFrameOptions) (*exec.Cmd, io.ReadCloser, *bytes.Buffer, error) {
	// Check if ffmpeg is available
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, nil, nil, fmt.Errorf("ffmpeg not found in $PATH: %w", err)
	}

	args := []string{
		"-loglevel", "info",
		"-stats",
	}

	// Add time range if specified
	if opts.TimeRange != nil {
		if opts.TimeRange.Start != "" {
			args = append(args, "-ss", opts.TimeRange.Start)
		}
		if opts.TimeRange.End != "" {
			// If end is specified, calculate duration
			startTime, endTime := 0.0, 0.0

			// Parse times
			if s, err := parseTimeString(opts.TimeRange.Start); err == nil {
				startTime = s
			}
			if e, err := parseTimeString(opts.TimeRange.End); err == nil {
				endTime = e
			}

			if endTime > startTime {
				duration := endTime - startTime
				args = append(args, "-t", fmt.Sprintf("%.3f", duration))
			}
		}
	}

	// Input file options
	args = append(args,
		"-probesize", "32M",
		"-analyzeduration", "10M",
		"-reconnect", "1",
		"-reconnect_at_eof", "1",
		"-reconnect_streamed", "1",
		"-reconnect_delay_max", "10",
		"-i", opts.URL,
	)

	// Process only every Nth frame (using select filter)
	var filterComplex string
	// if opts.IsHDR {
	// 	filterComplex = fmt.Sprintf(
	// 		"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable"+
	// 			":desat=0:peak=100,zscale=t=bt709:m=bt709:r=tv,format=rgb24,"+
	// 			"select=not(mod(n\\,%d))",
	// 		opts.SampleEveryNFrames,
	// 	)
	// } else {
	// 	// Simpler pipeline for SDR content
	// 	filterComplex = fmt.Sprintf("format=rgb24,select=not(mod(n\\,%d))", opts.SampleEveryNFrames)
	// }

	// DEBUG For HDR content, simplify the filter initially:
	if opts.IsHDR {
		filterComplex = fmt.Sprintf(
			"select=not(mod(n\\,%d))",
			opts.SampleEveryNFrames,
		)
	} else {
		// ADD THIS MISSING ELSE CLAUSE
		filterComplex = fmt.Sprintf(
			"select=not(mod(n\\,%d))",
			opts.SampleEveryNFrames,
		)
	}

	args = append(args,
		"-vf", filterComplex,
		"-vsync", "vfr",
		"-an", // No audio
	)

	// Output format
	// args = append(args,
	// 	"-f", "rawvideo",
	// 	"-pix_fmt", "rgb24",
	// 	"pipe:1",
	// )
	args = append(args,
		"-f", "image2pipe",
		"-pix_fmt", "rgb24",
		"-vcodec", "png",
		"pipe:1",
	)

	log.Printf("FFmpeg command: %v", args)
	// Create the command
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)

	// Create pipes for stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error creating stdout pipe: %w", err)
	}

	// Capture stderr to a buffer
	stderrBuf := &bytes.Buffer{}
	cmd.Stderr = stderrBuf

	// Start the process
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("error starting ffmpeg: %w", err)
	}

	// In CreateFrameExtractionProcess, before returning:
	log.Printf("Full FFmpeg command: %s", strings.Join(cmd.Args, " "))

	return cmd, stdout, stderrBuf, nil
}
