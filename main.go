package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/otiai10/gosseract/v2"
	"gocv.io/x/gocv"
)

type Detection struct {
	Time     float64
	PlateNum string
}

type Config struct {
	VideoPath           string
	OutputPath          string
	ModelPath           string
	ConfThreshold       float32
	NMSThreshold        float32
	ProcessInterval     time.Duration
	DeduplicationWindow time.Duration
}

var (
	videoPath  = flag.String("video", "", "Path to input video file")
	outputPath = flag.String("output", "results.csv", "Path to output CSV file")
	modelPath  = flag.String("model", "models/yolov8n.onnx", "Path to YOLO model")
)

func main() {
	flag.Parse()

	if *videoPath == "" {
		defaultVideo := filepath.Join("C:", "Users", "ignat", "Desktop", "VolgaIt", "видео-участникам.mp4")
		videoPath = &defaultVideo
	}

	config := Config{
		VideoPath:           *videoPath,
		OutputPath:          *outputPath,
		ModelPath:           *modelPath,
		ConfThreshold:       0.4,
		NMSThreshold:        0.5,
		ProcessInterval:     200 * time.Millisecond,
		DeduplicationWindow: 3 * time.Second,
	}

	log.Printf("Starting license plate recognition...")
	log.Printf("Video: %s", config.VideoPath)
	log.Printf("Output: %s", config.OutputPath)

	detections, err := processVideo(config)
	if err != nil {
		log.Fatalf("Error processing video: %v", err)
	}

	if err := saveResults(config.OutputPath, detections); err != nil {
		log.Fatalf("Error saving results: %v", err)
	}

	log.Printf("Processing complete. Found %d license plates", len(detections))
	log.Printf("Results saved to %s", config.OutputPath)
}

func processVideo(config Config) ([]Detection, error) {
	video, err := gocv.VideoCaptureFile(config.VideoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open video: %w", err)
	}
	defer video.Close()

	fps := video.Get(gocv.VideoCaptureFPS)
	if fps == 0 {
		fps = 25.0
	}

	totalFrames := int(video.Get(gocv.VideoCaptureFrameCount))
	log.Printf("Video FPS: %.2f, Total frames: %d", fps, totalFrames)

	detector := NewPlateDetector(config.ModelPath, config.ConfThreshold, config.NMSThreshold)
	recognizer := NewPlateRecognizer()
	defer recognizer.Close()

	var detections []Detection
	frame := gocv.NewMat()
	defer frame.Close()

	frameCount := 0
	framesToSkip := int(float64(config.ProcessInterval.Milliseconds()) / 1000.0 * fps)
	if framesToSkip < 1 {
		framesToSkip = 1
	}

	lastDetections := make(map[string]float64)

	for {
		if ok := video.Read(&frame); !ok || frame.Empty() {
			break
		}

		if frameCount%framesToSkip != 0 {
			frameCount++
			continue
		}

		currentTime := float64(frameCount) / fps

		plates := detector.Detect(frame)

		for _, plate := range plates {
			plateText := recognizer.Recognize(plate)
			if plateText == "" {
				continue
			}

			plateText = normalizeRussianPlate(plateText)
			if !isValidRussianPlate(plateText) {
				continue
			}

			if lastTime, exists := lastDetections[plateText]; exists {
				if currentTime-lastTime < config.DeduplicationWindow.Seconds() {
					continue
				}
			}

			lastDetections[plateText] = currentTime
			detections = append(detections, Detection{
				Time:     currentTime,
				PlateNum: plateText,
			})

			log.Printf("Detected plate: %s at time %s", plateText, formatTime(currentTime))
		}

		frameCount++

		if frameCount%100 == 0 {
			progress := float64(frameCount) / float64(totalFrames) * 100
			log.Printf("Progress: %.1f%% (%d/%d frames)", progress, frameCount, totalFrames)
		}
	}

	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Time < detections[j].Time
	})

	return detections, nil
}

type PlateDetector struct {
	net           gocv.Net
	confThreshold float32
	nmsThreshold  float32
	inputWidth    int
	inputHeight   int
}

func NewPlateDetector(modelPath string, confThreshold, nmsThreshold float32) *PlateDetector {
	net := gocv.ReadNet(modelPath, "")
	if net.Empty() {
		log.Printf("Warning: Could not load YOLO model from %s, using fallback detection", modelPath)
	}

	return &PlateDetector{
		net:           net,
		confThreshold: confThreshold,
		nmsThreshold:  nmsThreshold,
		inputWidth:    640,
		inputHeight:   640,
	}
}

func (pd *PlateDetector) Detect(frame gocv.Mat) []gocv.Mat {
	if pd.net.Empty() {
		return pd.fallbackDetect(frame)
	}

	blob := gocv.BlobFromImage(frame, 1.0/255.0, image.Pt(pd.inputWidth, pd.inputHeight),
		gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	pd.net.SetInput(blob, "")

	prob := pd.net.Forward("")
	defer prob.Close()

	return pd.postProcess(frame, prob)
}

func (pd *PlateDetector) fallbackDetect(frame gocv.Mat) []gocv.Mat {
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(frame, &gray, gocv.ColorBGRToGray)

	blurred := gocv.NewMat()
	defer blurred.Close()
	gocv.GaussianBlur(gray, &blurred, image.Pt(5, 5), 0, 0, gocv.BorderDefault)

	edges := gocv.NewMat()
	defer edges.Close()
	gocv.Canny(blurred, &edges, 30, 200)

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(17, 3))
	defer kernel.Close()

	morphed := gocv.NewMat()
	defer morphed.Close()
	gocv.MorphologyEx(edges, &morphed, gocv.MorphClose, kernel)

	contours := gocv.FindContours(morphed, gocv.RetrievalExternal, gocv.ChainApproxSimple)

	var plates []gocv.Mat

	for _, contour := range contours {
		area := gocv.ContourArea(contour)
		if area < 1000 || area > 50000 {
			continue
		}

		rect := gocv.BoundingRect(contour)
		aspectRatio := float64(rect.Dx()) / float64(rect.Dy())

		if aspectRatio < 2.0 || aspectRatio > 6.0 {
			continue
		}

		if rect.Dx() < 80 || rect.Dy() < 20 {
			continue
		}

		x1 := max(0, rect.Min.X-5)
		y1 := max(0, rect.Min.Y-5)
		x2 := min(frame.Cols(), rect.Max.X+5)
		y2 := min(frame.Rows(), rect.Max.Y+5)

		plateROI := frame.Region(image.Rect(x1, y1, x2, y2))
		plates = append(plates, plateROI.Clone())
		plateROI.Close()
	}

	return plates
}

func (pd *PlateDetector) postProcess(frame gocv.Mat, output gocv.Mat) []gocv.Mat {
	var plates []gocv.Mat

	rows := output.Size()[1]

	var classIDs []int
	var confidences []float32
	var boxes []image.Rectangle

	for i := 0; i < rows; i++ {
		data := output.RowRange(i, i+1)
		confidence := data.GetFloatAt(0, 4)

		if confidence > pd.confThreshold {
			x := data.GetFloatAt(0, 0)
			y := data.GetFloatAt(0, 1)
			w := data.GetFloatAt(0, 2)
			h := data.GetFloatAt(0, 3)

			left := int((x - w/2) * float32(frame.Cols()))
			top := int((y - h/2) * float32(frame.Rows()))
			width := int(w * float32(frame.Cols()))
			height := int(h * float32(frame.Rows()))

			classIDs = append(classIDs, 0)
			confidences = append(confidences, confidence)
			boxes = append(boxes, image.Rect(left, top, left+width, top+height))
		}
		data.Close()
	}

	if len(boxes) > 0 {
		indices := gocv.NMSBoxes(boxes, confidences, pd.confThreshold, pd.nmsThreshold)
		for _, idx := range indices {
			box := boxes[idx]
			x1 := max(0, box.Min.X)
			y1 := max(0, box.Min.Y)
			x2 := min(frame.Cols(), box.Max.X)
			y2 := min(frame.Rows(), box.Max.Y)

			plateROI := frame.Region(image.Rect(x1, y1, x2, y2))
			plates = append(plates, plateROI.Clone())
			plateROI.Close()
		}
	}

	return plates
}

type PlateRecognizer struct {
	client *gosseract.Client
}

func NewPlateRecognizer() *PlateRecognizer {
	client := gosseract.NewClient()
	client.SetLanguage("eng")
	client.SetWhitelist("ABCEHKMOPTXY0123456789")
	client.SetPageSegMode(gosseract.PSM_SINGLE_LINE)

	return &PlateRecognizer{
		client: client,
	}
}

func (pr *PlateRecognizer) Recognize(plateMat gocv.Mat) string {
	defer plateMat.Close()

	if plateMat.Empty() || plateMat.Cols() < 20 || plateMat.Rows() < 10 {
		return ""
	}

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(plateMat, &gray, gocv.ColorBGRToGray)

	resized := gocv.NewMat()
	defer resized.Close()
	targetHeight := 100
	aspectRatio := float64(gray.Cols()) / float64(gray.Rows())
	targetWidth := int(float64(targetHeight) * aspectRatio)
	gocv.Resize(gray, &resized, image.Pt(targetWidth, targetHeight), 0, 0, gocv.InterpolationLinear)

	thresh := gocv.NewMat()
	defer thresh.Close()
	gocv.AdaptiveThreshold(resized, &thresh, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, 11, 2)

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(2, 2))
	defer kernel.Close()

	morphed := gocv.NewMat()
	defer morphed.Close()
	gocv.MorphologyEx(thresh, &morphed, gocv.MorphClose, kernel)

	buf, err := gocv.IMEncode(".png", morphed)
	if err != nil {
		return ""
	}

	pr.client.SetImageFromBytes(buf.GetBytes())
	text, err := pr.client.Text()
	if err != nil {
		return ""
	}

	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, " ", "")
	text = strings.ReplaceAll(text, "\n", "")
	text = strings.ToUpper(text)

	return text
}

func (pr *PlateRecognizer) Close() {
	if pr.client != nil {
		pr.client.Close()
	}
}

func normalizeRussianPlate(text string) string {
	text = strings.ToUpper(text)
	text = strings.ReplaceAll(text, " ", "")
	text = strings.ReplaceAll(text, "-", "")

	replacements := map[rune]rune{
		'0': 'O', 'О': 'O',
		'1': 'I', 'І': 'I',
		'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E',
		'Н': 'H', 'К': 'K', 'М': 'M',
		'Р': 'P', 'Т': 'T', 'У': 'Y', 'Х': 'X',
	}

	var result strings.Builder
	for _, ch := range text {
		if replacement, ok := replacements[ch]; ok {
			result.WriteRune(replacement)
		} else {
			result.WriteRune(ch)
		}
	}

	return result.String()
}

func isValidRussianPlate(text string) bool {
	if len(text) < 8 || len(text) > 9 {
		return false
	}

	validChars := "ABCEHKMOPTXY0123456789"
	for _, ch := range text {
		if !strings.ContainsRune(validChars, ch) {
			return false
		}
	}

	return true
}

func formatTime(seconds float64) string {
	minutes := int(seconds) / 60
	secs := int(seconds) % 60
	ms := int((seconds - float64(int(seconds))) * 100)
	return fmt.Sprintf("%02d:%02d.%02d", minutes, secs, ms)
}

func saveResults(outputPath string, detections []Detection) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = ';'
	defer writer.Flush()

	if err := writer.Write([]string{"time", "plate_num"}); err != nil {
		return err
	}

	for _, det := range detections {
		record := []string{
			formatTime(det.Time),
			det.PlateNum,
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
