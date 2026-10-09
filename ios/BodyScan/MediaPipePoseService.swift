import CoreImage
import Foundation
import MediaPipeTasksVision
import UIKit

final class MediaPipePoseService: NSObject, PoseLandmarkerLiveStreamDelegate {
    var onResult: (([Point3D], Int) -> Void)?
    var onStatus: ((String) -> Void)?

    private let modelURL = URL(string: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")!
    private let workQueue = DispatchQueue(label: "BodyScan.MediaPipe", qos: .userInitiated)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    private var landmarker: PoseLandmarker?
    private var isPreparing = false
    private var lastTimestampMs = -1

    override init() {
        super.init()
        prepare()
    }

    func prepare() {
        guard !isPreparing, landmarker == nil else { return }
        isPreparing = true
        onStatus?("MediaPipe model preparing")

        Task {
            do {
                let modelPath = try await ensureModel()
                let options = PoseLandmarkerOptions()
                options.baseOptions.modelAssetPath = modelPath
                options.runningMode = .liveStream
                options.numPoses = 1
                options.minPoseDetectionConfidence = 0.55
                options.minPosePresenceConfidence = 0.55
                options.minTrackingConfidence = 0.55
                options.poseLandmarkerLiveStreamDelegate = self
                let newLandmarker = try PoseLandmarker(options: options)
                await MainActor.run {
                    self.landmarker = newLandmarker
                    self.isPreparing = false
                    self.onStatus?("MediaPipe ready")
                }
            } catch {
                await MainActor.run {
                    self.isPreparing = false
                    self.onStatus?("MediaPipe failed: \(error.localizedDescription)")
                }
            }
        }
    }

    func process(pixelBuffer: CVPixelBuffer, timestampMs: Int) {
        guard let landmarker else { return }
        guard timestampMs > lastTimestampMs else { return }
        lastTimestampMs = timestampMs

        workQueue.async { [weak self] in
            guard let self else { return }
            do {
                let bgra = try self.makeBGRA(from: pixelBuffer)
                let image = try MPImage(pixelBuffer: bgra, orientation: .right)
                try landmarker.detectAsync(image: image, timestampInMilliseconds: timestampMs)
            } catch {
                DispatchQueue.main.async {
                    self.onStatus?("MediaPipe frame error: \(error.localizedDescription)")
                }
            }
        }
    }

    func poseLandmarker(
        _ poseLandmarker: PoseLandmarker,
        didFinishDetection result: PoseLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if let error {
            DispatchQueue.main.async {
                self.onStatus?("MediaPipe result error: \(error.localizedDescription)")
            }
            return
        }

        guard let world = result?.worldLandmarks.first else { return }
        let points = world.map { landmark in
            Point3D(landmark.x, landmark.y, landmark.z)
        }

        DispatchQueue.main.async {
            self.onResult?(points, timestampInMilliseconds)
        }
    }

    private func ensureModel() async throws -> String {
        let fm = FileManager.default
        let dir = try fm.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("BodyScanModels", isDirectory: true)

        if !fm.fileExists(atPath: dir.path) {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        let target = dir.appendingPathComponent("pose_landmarker_full.task")
        if fm.fileExists(atPath: target.path) {
            return target.path
        }

        let (data, response) = try await URLSession.shared.data(from: modelURL)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw NSError(
                domain: "BodyScan.MediaPipe",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Model download HTTP \(http.statusCode)"]
            )
        }
        try data.write(to: target, options: .atomic)
        return target.path
    }

    private func makeBGRA(from source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)

        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &output
        )
        guard status == kCVReturnSuccess, let output else {
            throw NSError(
                domain: "BodyScan.MediaPipe",
                code: Int(status),
                userInfo: [NSLocalizedDescriptionKey: "Unable to allocate BGRA pixel buffer"]
            )
        }

        ciContext.render(CIImage(cvPixelBuffer: source), to: output)
        return output
    }
}
