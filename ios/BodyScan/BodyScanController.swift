import ARKit
import Combine
import Foundation
import ImageIO
import simd
import UIKit
import Vision

final class BodyScanController: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published var inputHeightCm: Double = 178
    @Published private(set) var statusText = "准备中"
    @Published private(set) var isScanning = false
    @Published private(set) var appleJointCount = 0
    @Published private(set) var mediaPipeJointCount = 0
    @Published private(set) var mediaPipeStatus = "MediaPipe preparing"
    @Published private(set) var personDepthAvailable = false
    @Published private(set) var lidarSceneDepthAvailable = false
    @Published private(set) var pointCloudCount = 0
    @Published private(set) var scanFrameCount = 0
    @Published private(set) var angularCoverage = 0.0
    @Published private(set) var latestMetrics = TailoringMetrics.empty
    @Published private(set) var exportURL: URL?

    private let visionQueue = DispatchQueue(label: "BodyScan.Vision", qos: .userInitiated)
    private let pointCloudQueue = DispatchQueue(label: "BodyScan.PointCloud", qos: .userInitiated)
    private let mediaPipe = MediaPipePoseService()

    private var visionBusy = false
    private var lastVisionTimestamp: TimeInterval = 0
    private var lastPointCloudTimestamp: TimeInterval = 0
    private var lastMediaPipeTimestamp: TimeInterval = 0

    private var scanStartedAt: Date?
    private var poseFrames: [PoseFrameRecord] = []
    private var pointCloud: [Point3D] = []
    private var angularBins = Set<Int>()
    private var mediaPipeFrameCount = 0
    private let angularBinCount = 24
    private let maxPointCloudPoints = 120_000

    override init() {
        super.init()
        session.delegate = self

        mediaPipe.onStatus = { [weak self] text in
            self?.mediaPipeStatus = text
        }

        mediaPipe.onResult = { [weak self] points in
            guard let self else { return }
            self.mediaPipeJointCount = points.count
            self.mediaPipeFrameCount += 1
        }
    }

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            statusText = "此 iPhone 不支持 ARWorldTracking"
            return
        }

        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.isAutoFocusEnabled = true

        var semantics: ARConfiguration.FrameSemantics = []

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.bodyDetection) {
            semantics.insert(.bodyDetection)
        }

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            semantics.insert(.personSegmentationWithDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            semantics.insert(.personSegmentation)
        }

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            semantics.insert(.sceneDepth)
        }

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            semantics.insert(.smoothedSceneDepth)
        }

        config.frameSemantics = semantics
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        statusText = "摄像头已启动"
    }

    func beginScan() {
        guard !isScanning else { return }

        poseFrames.removeAll(keepingCapacity: true)
        pointCloud.removeAll(keepingCapacity: true)
        angularBins.removeAll(keepingCapacity: true)
        mediaPipeFrameCount = 0
        pointCloudCount = 0
        scanFrameCount = 0
        angularCoverage = 0
        latestMetrics = .empty
        exportURL = nil
        scanStartedAt = Date()
        isScanning = true
        statusText = "请缓慢转一圈"
    }

    func finishScan() {
        guard isScanning else { return }
        isScanning = false
        statusText = "扫描结束，正在生成数据"
        exportURL = makeExportFile()
        statusText = "扫描完成"
    }

    func resetScan() {
        isScanning = false
        poseFrames.removeAll()
        pointCloud.removeAll()
        angularBins.removeAll()
        mediaPipeFrameCount = 0
        pointCloudCount = 0
        scanFrameCount = 0
        angularCoverage = 0
        latestMetrics = .empty
        exportURL = nil
        statusText = "已重置"
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let personDepth = frame.estimatedDepthData != nil && frame.segmentationBuffer != nil
        let lidarDepth = frame.sceneDepth != nil || frame.smoothedSceneDepth != nil

        DispatchQueue.main.async {
            self.personDepthAvailable = personDepth
            self.lidarSceneDepthAvailable = lidarDepth
            if self.isScanning {
                self.scanFrameCount += 1
            }
        }

        if frame.timestamp - lastVisionTimestamp >= 0.18 {
            lastVisionTimestamp = frame.timestamp
            analyzeAppleVision(frame)
        }

        if frame.timestamp - lastMediaPipeTimestamp >= 0.25 {
            lastMediaPipeTimestamp = frame.timestamp
            mediaPipe.process(
                pixelBuffer: frame.capturedImage,
                timestampMs: Int(frame.timestamp * 1000)
            )
        }

        if isScanning, frame.timestamp - lastPointCloudTimestamp >= 0.30 {
            lastPointCloudTimestamp = frame.timestamp
            accumulatePersonPointCloud(frame)
        }
    }

    private func analyzeAppleVision(_ frame: ARFrame) {
        guard !visionBusy else { return }
        visionBusy = true

        let pixelBuffer = frame.capturedImage
        let timestamp = frame.timestamp
        let enteredHeight = inputHeightCm

        visionQueue.async { [weak self] in
            guard let self else { return }
            defer { self.visionBusy = false }

            do {
                let request = VNDetectHumanBodyPose3DRequest()
                let handler = VNImageRequestHandler(
                    cvPixelBuffer: pixelBuffer,
                    orientation: .right,
                    options: [:]
                )
                try handler.perform([request])

                guard let observation = request.results?.first else {
                    DispatchQueue.main.async {
                        self.appleJointCount = 0
                    }
                    return
                }

                var jointMap: [String: Point3D] = [:]

                for joint in observation.availableJointNames {
                    do {
                        let transform = try observation.cameraRelativePosition(joint)
                        let t = transform.columns.3
                        jointMap[String(describing: joint)] = Point3D(t.x, t.y, t.z)
                    } catch {
                        continue
                    }
                }

                let metrics = self.metrics(
                    from: observation,
                    calibrationHeightCm: enteredHeight
                )
                let yaw = self.bodyYaw(from: observation)

                if let yaw {
                    let bin = self.yawBin(yaw)
                    self.angularBins.insert(bin)
                }

                let record = PoseFrameRecord(
                    timestamp: timestamp,
                    appleJoints: jointMap,
                    mediaPipeWorldJoints: [],
                    bodyYawDegrees: yaw,
                    bodyHeightEstimateMeters: observation.bodyHeight
                )

                if self.isScanning {
                    self.poseFrames.append(record)
                }

                let coverage = Double(self.angularBins.count) / Double(self.angularBinCount)

                DispatchQueue.main.async {
                    self.appleJointCount = jointMap.count
                    self.latestMetrics = metrics
                    self.angularCoverage = min(1, coverage)
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusText = "Apple 3D Pose error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func metrics(
        from observation: VNHumanBodyPose3DObservation,
        calibrationHeightCm: Double
    ) -> TailoringMetrics {
        let observedHeightCm = Double(observation.bodyHeight) * 100
        let scale: Float = observedHeightCm > 80
            ? Float(calibrationHeightCm / observedHeightCm)
            : 1

        func position(_ joint: VNHumanBodyPose3DObservation.JointName) -> SIMD3<Float>? {
            guard let matrix = try? observation.cameraRelativePosition(joint) else { return nil }
            let t = matrix.columns.3
            return SIMD3<Float>(t.x, t.y, t.z)
        }

        func cm(_ meters: Float) -> Double {
            Double(meters * scale * 100)
        }

        let leftShoulder = position(.leftShoulder)
        let rightShoulder = position(.rightShoulder)
        let leftElbow = position(.leftElbow)
        let rightElbow = position(.rightElbow)
        let leftWrist = position(.leftWrist)
        let rightWrist = position(.rightWrist)

        let shoulderWidth = pairDistance(leftShoulder, rightShoulder).map(cm)
        let shoulderDrop: Double? = {
            guard let l = leftShoulder, let r = rightShoulder else { return nil }
            return cm(abs(l.y - r.y))
        }()

        let leftArm: Double? = {
            guard
                let s = leftShoulder,
                let e = leftElbow,
                let w = leftWrist
            else { return nil }
            return cm(simd_distance(s, e) + simd_distance(e, w))
        }()

        let rightArm: Double? = {
            guard
                let s = rightShoulder,
                let e = rightElbow,
                let w = rightWrist
            else { return nil }
            return cm(simd_distance(s, e) + simd_distance(e, w))
        }()

        return TailoringMetrics(
            visionBodyHeightCm: observedHeightCm > 0 ? observedHeightCm : nil,
            shoulderWidthCm: shoulderWidth,
            shoulderHeightDifferenceCm: shoulderDrop,
            leftArmLengthCm: leftArm,
            rightArmLengthCm: rightArm
        )
    }

    private func bodyYaw(from observation: VNHumanBodyPose3DObservation) -> Double? {
        guard
            let leftShoulder = cameraPosition(.leftShoulder, observation),
            let rightShoulder = cameraPosition(.rightShoulder, observation),
            let root = cameraPosition(.root, observation),
            let centerShoulder = cameraPosition(.centerShoulder, observation)
        else { return nil }

        let shoulderAxis = simd_normalize(rightShoulder - leftShoulder)
        let upAxis = simd_normalize(centerShoulder - root)
        let forward = simd_normalize(simd_cross(shoulderAxis, upAxis))

        let radians = atan2(Double(forward.x), Double(-forward.z))
        var degrees = radians * 180 / .pi
        if degrees < 0 { degrees += 360 }
        return degrees
    }

    private func cameraPosition(
        _ joint: VNHumanBodyPose3DObservation.JointName,
        _ observation: VNHumanBodyPose3DObservation
    ) -> SIMD3<Float>? {
        guard let matrix = try? observation.cameraRelativePosition(joint) else { return nil }
        let t = matrix.columns.3
        return SIMD3<Float>(t.x, t.y, t.z)
    }

    private func pairDistance(
        _ a: SIMD3<Float>?,
        _ b: SIMD3<Float>?
    ) -> Float? {
        guard let a, let b else { return nil }
        return simd_distance(a, b)
    }

    private func yawBin(_ degrees: Double) -> Int {
        let normalized = degrees.truncatingRemainder(dividingBy: 360)
        let width = 360.0 / Double(angularBinCount)
        return max(0, min(angularBinCount - 1, Int(normalized / width)))
    }

    private func accumulatePersonPointCloud(_ frame: ARFrame) {
        guard pointCloud.count < maxPointCloudPoints else { return }

        let depthBuffer: CVPixelBuffer?
        let segmentation = frame.segmentationBuffer

        if let personDepth = frame.estimatedDepthData, segmentation != nil {
            depthBuffer = personDepth
        } else if let lidar = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap {
            depthBuffer = lidar
        } else {
            return
        }

        guard let depthBuffer else { return }

        let cameraTransform = frame.camera.transform
        var intrinsics = frame.camera.intrinsics
        let capturedWidth = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let capturedHeight = Float(CVPixelBufferGetHeight(frame.capturedImage))

        pointCloudQueue.async { [weak self] in
            guard let self else { return }

            let depthWidth = CVPixelBufferGetWidth(depthBuffer)
            let depthHeight = CVPixelBufferGetHeight(depthBuffer)
            guard depthWidth > 0, depthHeight > 0 else { return }

            let scaleX = Float(depthWidth) / capturedWidth
            let scaleY = Float(depthHeight) / capturedHeight
            intrinsics.columns.0.x *= scaleX
            intrinsics.columns.1.y *= scaleY
            intrinsics.columns.2.x *= scaleX
            intrinsics.columns.2.y *= scaleY

            guard CVPixelBufferGetPixelFormatType(depthBuffer) == kCVPixelFormatType_DepthFloat32 else {
                return
            }

            CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
            if let segmentation {
                CVPixelBufferLockBaseAddress(segmentation, .readOnly)
            }

            defer {
                if let segmentation {
                    CVPixelBufferUnlockBaseAddress(segmentation, .readOnly)
                }
                CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            }

            guard let depthBase = CVPixelBufferGetBaseAddress(depthBuffer) else { return }
            let depthRowFloats = CVPixelBufferGetBytesPerRow(depthBuffer) / MemoryLayout<Float32>.size
            let depth = depthBase.assumingMemoryBound(to: Float32.self)

            var maskBase: UnsafeMutablePointer<UInt8>?
            var maskWidth = 0
            var maskHeight = 0
            var maskRowBytes = 0

            if let segmentation,
               let rawMask = CVPixelBufferGetBaseAddress(segmentation) {
                maskBase = rawMask.assumingMemoryBound(to: UInt8.self)
                maskWidth = CVPixelBufferGetWidth(segmentation)
                maskHeight = CVPixelBufferGetHeight(segmentation)
                maskRowBytes = CVPixelBufferGetBytesPerRow(segmentation)
            }

            let fx = intrinsics.columns.0.x
            let fy = intrinsics.columns.1.y
            let cx = intrinsics.columns.2.x
            let cy = intrinsics.columns.2.y
            let sampleStep = 6

            var newPoints: [Point3D] = []
            newPoints.reserveCapacity((depthWidth / sampleStep) * (depthHeight / sampleStep) / 3)

            for y in stride(from: 0, to: depthHeight, by: sampleStep) {
                for x in stride(from: 0, to: depthWidth, by: sampleStep) {
                    if let maskBase, maskWidth > 0, maskHeight > 0 {
                        let mx = min(maskWidth - 1, x * maskWidth / depthWidth)
                        let my = min(maskHeight - 1, y * maskHeight / depthHeight)
                        let maskValue = maskBase[my * maskRowBytes + mx]
                        if maskValue == 0 { continue }
                    }

                    let z = depth[y * depthRowFloats + x]
                    if !z.isFinite || z < 0.5 || z > 5.0 { continue }

                    let cameraX = (Float(x) - cx) / fx * z
                    let cameraY = -(Float(y) - cy) / fy * z
                    let cameraPoint = SIMD4<Float>(cameraX, cameraY, -z, 1)
                    let worldPoint = cameraTransform * cameraPoint

                    newPoints.append(Point3D(
                        worldPoint.x,
                        worldPoint.y,
                        worldPoint.z
                    ))

                    if self.pointCloud.count + newPoints.count >= self.maxPointCloudPoints {
                        break
                    }
                }

                if self.pointCloud.count + newPoints.count >= self.maxPointCloudPoints {
                    break
                }
            }

            self.pointCloud.append(contentsOf: newPoints)

            DispatchQueue.main.async {
                self.pointCloudCount = self.pointCloud.count
            }
        }
    }

    private func makeExportFile() -> URL? {
        let duration = scanStartedAt.map { Date().timeIntervalSince($0) } ?? 0

        let export = ScanExport(
            version: "0.2-native-ios",
            createdAt: Date(),
            inputHeightCm: inputHeightCm,
            durationSeconds: duration,
            capturedFrameCount: scanFrameCount,
            applePoseFrameCount: poseFrames.count,
            mediaPipeFrameCount: mediaPipeFrameCount,
            angularCoveragePercent: angularCoverage * 100,
            pointCloudPointCount: pointCloud.count,
            personDepthAvailable: personDepthAvailable,
            lidarSceneDepthAvailable: lidarSceneDepthAvailable,
            latestMetrics: latestMetrics,
            poseFrames: poseFrames,
            pointCloud: pointCloud
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(export)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("body-scan-v0.2-\(Int(Date().timeIntervalSince1970)).json")
            try data.write(to: url, options: .atomic)
            return url
        } catch {
            statusText = "导出失败: \(error.localizedDescription)"
            return nil
        }
    }
}
