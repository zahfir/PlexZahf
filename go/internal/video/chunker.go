package video

type Frame []byte

func ChunkFrames(buffer []byte, width, height, chunkSize int) [][]Frame {
	frameSize := width * height * 3
	var chunks [][]Frame
	for i := 0; i < len(buffer); i += chunkSize * frameSize {
		var batch []Frame
		for j := 0; j < chunkSize; j++ {
			idx := i + j*frameSize
			if idx+frameSize > len(buffer) {
				break
			}
			frame := buffer[idx : idx+frameSize]
			batch = append(batch, frame)
		}
		chunks = append(chunks, batch)
	}
	return chunks
}
