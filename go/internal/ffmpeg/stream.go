package ffmpeg

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"
)

// FFmpegOptions holds video stream parameters.
type FFmpegOptions struct {
	URL     string
	Width   int
	Height  int
	FPS     int
	Bitrate string
	Timeout time.Duration
}

// CreateFFmpegProcess starts an ffmpeg command with safe lifecycle and context.
func CreateFFmpegProcess(ctx context.Context, opts FFmpegOptions) (*exec.Cmd, io.ReadCloser, *bytes.Buffer, error) {
	// Check if ffmpeg is available
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, nil, nil, fmt.Errorf("ffmpeg not found in $PATH: %w", err)
	}

	// pipeName := `\\.\pipe\ffmpeg-test`

	args := []string{
		"-stats",
		"-loglevel", "info",
		"-probesize", "32M",
		"-analyzeduration", "10M",
		"-reconnect", "1",
		"-reconnect_at_eof", "1",
		"-reconnect_streamed", "1",
		"-reconnect_delay_max", "10",
		"-i", opts.URL,
		"-an",
		"-s", fmt.Sprintf("%dx%d", opts.Width, opts.Height),
		"-b:v", opts.Bitrate,
		"-r", fmt.Sprintf("%d", opts.FPS),
		"-preset", "ultrafast",
		"-tune", "fastdecode",
		"-movflags", "+faststart",
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"pipe:1",
	}

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	var stderr bytes.Buffer
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("failed to start ffmpeg: %w", err)
	}

	return cmd, stdout, &stderr, nil
}

// ReadStreamToMemory reads ffmpeg stdout up to a specified limit and cleans up the process.
func ReadStreamToMemory(ctx context.Context, cmd *exec.Cmd, stream io.ReadCloser, stderr *bytes.Buffer, chunkSize int, maxBytes int64) ([]byte, error) {
	defer stream.Close()

	// Bounded reader to avoid memory explosion
	limitedReader := io.LimitReader(stream, maxBytes)
	buffer := &bytes.Buffer{}
	chunk := make([]byte, chunkSize)

	for {
		n, err := limitedReader.Read(chunk)
		if n > 0 {
			buffer.Write(chunk[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("stream read error: %w", err)
		}
	}

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	select {
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("ffmpeg process canceled or timed out")
	case err := <-done:
		if err != nil {
			return nil, fmt.Errorf("ffmpeg error: %v - stderr: %s", err, stderr.String())
		}
		return buffer.Bytes(), nil
	}
}
