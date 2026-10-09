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

    private struct TimedMediaPipePose {
        var timestamp: TimeInterval
        var points: [Point3D]
    }

    private struct BodyReference {
        var timestamp: TimeInterval
        var rootWorld: SIMD3<Float>
        var yawWorld: Float
        var shoulderHeightMeters: Float
        var calibrationScale: Float
    }

    private let visionQueue = DispatchQueue(label: "BodyScan.Vision", qos: .userInitiated)
    private let pointCloudQueue = DispatchQueue(label: "BodyScan.PointCloud", qos: .userInitiated)
    private let dataLock = NSLock()
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
    private let maxPointCloudPoints = 140_000

    private var latestMediaPipePose: TimedMediaPipePose?
    private var latestBodyReference: BodyReference?
    private var scanReferenceYaw: Float?
    private var visionHeightSamplesMeters: [Float] = []
    private var torsoHeightSamplesMeters: [Float] = []
    private var shoulderWidthSamplesMeters: [Float] = []

    private var latestChestSection: CrossSectionMeasurement?
    private var latestWaistSection: CrossSectionMeasurement?
    private var latestHipSection: CrossSectionMeasurement?

    override init() {
        super.init()
        session.delegate = self

        mediaPipe.onStatus = { [weak self] text in
            self?.mediaPipeStatus = text
        }

        mediaPipe.onResult = { [weak self] points, timestampMs in
            guard let self else { return }
            let timestamp = TimeInterval(timestampMs) / 1000.0

            self.dataLock.lock()
            self.latestMediaPipePose = TimedMediaPipePose(
                timestamp: timestamp,
                points: points
            )
            self.dataLock.unlock()

            self.mediaPipeJointCount = points.count
            if self.isScanning {
                self.mediaPipeFrameCount += 1
            }
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

        visionQueue.sync {}
        pointCloudQueue.sync {}

        poseFrames.removeAll(keepingCapacity: true)
        pointCloud.removeAll(keepingCapacity: true)
        angularBins.removeAll(keepingCapacity: true)
        visionHeightSamplesMeters.removeAll(keepingCapacity: true)
        torsoHeightSamplesMeters.removeAll(keepingCapacity: true)
        shoulderWidthSamplesMeters.removeAll(keepingCapacity: true)

        dataLock.lock()
        scanReferenceYaw = nil
        latestBodyReference = nil
        dataLock.unlock()

        mediaPipeFrameCount = 0
        pointCloudCount = 0
        scanFrameCount = 0
        angularCoverage = 0
        latestMetrics = .empty
        latestChestSection = nil
        latestWaistSection = nil
        latestHipSection = nil
        exportURL = nil
        scanStartedAt = Date()
        isScanning = true
        statusText = "请缓慢转一圈"
    }

    func finishScan() {
        guard isScanning else { return }
        isScanning = false
        statusText = "扫描结束，正在计算身体截面"

        visionQueue.sync {}
        pointCloudQueue.sync {}

        let heightMedian = median(visionHeightSamplesMeters)
        let calibrationScale: Float = {
            guard let heightMedian, heightMedian > 0.8 else { return 1 }
            return Float(inputHeightCm / 100.0) / heightMedian
        }()

        let scaledPointCloud = pointCloud.map {
            Point3D(
                $0.x * calibrationScale,
                $0.y * calibrationScale,
                $0.z * calibrationScale
            )
        }

        let torsoHeight = Double((median(torsoHeightSamplesMeters) ?? 0.55) * calibrationScale)
        let shoulderWidth = Double((median(shoulderWidthSamplesMeters) ?? 0.43) * calibrationScale)

        let sections = BodyAnalysisEngine.analyzeCrossSections(
            canonicalPoints: scaledPointCloud,
            torsoHeightMeters: torsoHeight,
            shoulderWidthMeters: shoulderWidth,
            angularCoverage: angularCoverage
        )

        latestChestSection = sections.chest
        latestWaistSection = sections.waist
        latestHipSection = sections.hip
        pointCloud = scaledPointCloud
        pointCloudCount = scaledPointCloud.count

        var metrics = latestMetrics
        metrics.chestCircumferenceCm = sections.chest?.perimeterCm
        metrics.waistCircumferenceCm = sections.waist?.perimeterCm
        metrics.hipCircumferenceCm = sections.hip?.perimeterCm
        metrics.chestWidthCm = sections.chest?.widthCm
        metrics.chestDepthCm = sections.chest?.depthCm
        metrics.waistWidthCm = sections.waist?.widthCm
        metrics.waistDepthCm = sections.waist?.depthCm
        metrics.hipWidthCm = sections.hip?.widthCm
        metrics.hipDepthCm = sections.hip?.depthCm

        let confidences = [
            sections.chest?.confidence,
            sections.waist?.confidence,
            sections.hip?.confidence
        ].compactMap { $0 }
        if !confidences.isEmpty {
            metrics.sectionConfidence = confidences.reduce(0, +) / Double(confidences.count)
        }

        latestMetrics = metrics
        exportURL = makeExportFile()

        if sections.chest == nil || sections.waist == nil || sections.hip == nil {
            statusText = "扫描完成，但部分截面数据不足"
        } else {
            statusText = "扫描完成：已生成胸/腰/臀截面"
        }
    }

    func resetScan() {
        isScanning = false
        visionQueue.sync {}
        pointCloudQueue.sync {}

        poseFrames.removeAll()
        pointCloud.removeAll()
        angularBins.removeAll()
        visionHeightSamplesMeters.removeAll()
        torsoHeightSamplesMeters.removeAll()
        shoulderWidthSamplesMeters.removeAll()

        dataLock.lock()
        scanReferenceYaw = nil
        latestBodyReference = nil
        dataLock.unlock()

        mediaPipeFrameCount = 0
        pointCloudCount = 0
        scanFrameCount = 0
        angularCoverage = 0
        latestMetrics = .empty
        latestChestSection = nil
        latestWaistSection = nil
        latestHipSection = nil
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

        if isScanning, frame.timestamp - lastPointCloudTimestamp >= 0.28 {
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
        let cameraTransform = frame.camera.transform

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

                let rawApple = self.canonicalAppleJoints(observation)
                let observedHeightMeters = observation.bodyHeight
                let calibrationScale: Float = observedHeightMeters > 0.8
                    ? Float(enteredHeight / 100.0) / observedHeightMeters
                    : 1

                let calibratedApple = self.calibrateAppleJoints(
                    rawApple,
                    scale: calibrationScale
                )

                let mediaPipeSnapshot: TimedMediaPipePose? = {
                    self.dataLock.lock()
                    defer { self.dataLock.unlock() }
                    guard
                        let value = self.latestMediaPipePose,
                        abs(value.timestamp - timestamp) < 0.50
                    else { return nil }
                    return value
                }()

                let fusedResult: FusedSkeletonResult = {
                    if let mediaPipeSnapshot,
                       let fused = BodyAnalysisEngine.fuseSkeletons(
                            apple: rawApple,
                            mediaPipe: mediaPipeSnapshot.points,
                            appleCalibrationScale: calibrationScale
                       ) {
                        return fused
                    }
                    return FusedSkeletonResult(
                        joints: calibratedApple,
                        agreementCm: nil
                    )
                }()

                let metrics = BodyAnalysisEngine.metrics(
                    fused: fusedResult,
                    visionBodyHeightCm: observedHeightMeters > 0
                        ? Double(observedHeightMeters * 100)
                        : nil
                )

                let reference = self.makeBodyReference(
                    observation: observation,
                    cameraTransform: cameraTransform,
                    timestamp: timestamp,
                    calibrationScale: calibrationScale
                )

                if let reference {
                    self.dataLock.lock()
                    self.latestBodyReference = reference
                    if self.isScanning && self.scanReferenceYaw == nil {
                        self.scanReferenceYaw = reference.yawWorld
                    }
                    let referenceYaw = self.scanReferenceYaw
                    self.dataLock.unlock()

                    if self.isScanning, let referenceYaw {
                        let relativeDegrees = Double(
                            self.normalizedPositiveAngle(reference.yawWorld - referenceYaw)
                        ) * 180 / .pi
                        self.angularBins.insert(self.yawBin(relativeDegrees))
                    }

                    if self.isScanning {
                        self.torsoHeightSamplesMeters.append(reference.shoulderHeightMeters)
                    }
                }

                if self.isScanning {
                    if observedHeightMeters > 0.8 {
                        self.visionHeightSamplesMeters.append(observedHeightMeters)
                    }

                    if let l = rawApple["leftShoulder"],
                       let r = rawApple["rightShoulder"] {
                        self.shoulderWidthSamplesMeters.append(simd_distance(l, r))
                    }

                    let record = PoseFrameRecord(
                        timestamp: timestamp,
                        appleJoints: calibratedApple.mapValues { Point3D($0.x, $0.y, $0.z) },
                        mediaPipeWorldJoints: mediaPipeSnapshot?.points ?? [],
                        fusedJoints: fusedResult.joints.mapValues { Point3D($0.x, $0.y, $0.z) },
                        bodyYawDegrees: reference.map { Double($0.yawWorld) * 180 / .pi },
                        bodyHeightEstimateMeters: observedHeightMeters,
                        skeletonAgreementCm: fusedResult.agreementCm
                    )
                    self.poseFrames.append(record)
                }

                let coverage = Double(self.angularBins.count) / Double(self.angularBinCount)

                DispatchQueue.main.async {
                    self.appleJointCount = rawApple.count
                    var mergedMetrics = metrics
                    mergedMetrics.chestCircumferenceCm = self.latestMetrics.chestCircumferenceCm
                    mergedMetrics.waistCircumferenceCm = self.latestMetrics.waistCircumferenceCm
                    mergedMetrics.hipCircumferenceCm = self.latestMetrics.hipCircumferenceCm
                    mergedMetrics.chestWidthCm = self.latestMetrics.chestWidthCm
                    mergedMetrics.chestDepthCm = self.latestMetrics.chestDepthCm
                    mergedMetrics.waistWidthCm = self.latestMetrics.waistWidthCm
                    mergedMetrics.waistDepthCm = self.latestMetrics.waistDepthCm
                    mergedMetrics.hipWidthCm = self.latestMetrics.hipWidthCm
                    mergedMetrics.hipDepthCm = self.latestMetrics.hipDepthCm
                    mergedMetrics.sectionConfidence = self.latestMetrics.sectionConfidence
                    self.latestMetrics = mergedMetrics
                    self.angularCoverage = min(1, coverage)
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusText = "Apple 3D Pose error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func canonicalAppleJoints(
        _ observation: VNHumanBodyPose3DObservation
    ) -> [String: SIMD3<Float>] {
        let definitions: [(String, VNHumanBodyPose3DObservation.JointName)] = [
            ("root", .root),
            ("centerShoulder", .centerShoulder),
            ("leftShoulder", .leftShoulder),
            ("rightShoulder", .rightShoulder),
            ("leftElbow", .leftElbow),
            ("rightElbow", .rightElbow),
            ("leftWrist", .leftWrist),
            ("rightWrist", .rightWrist),
            ("leftHip", .leftHip),
            ("rightHip", .rightHip),
            ("leftKnee", .leftKnee),
            ("rightKnee", .rightKnee),
            ("leftAnkle", .leftAnkle),
            ("rightAnkle", .rightAnkle)
        ]

        var result: [String: SIMD3<Float>] = [:]
        for (name, joint) in definitions {
            if let point = cameraPosition(joint, observation) {
                result[name] = point
            }
        }
        return result
    }

    private func calibrateAppleJoints(
        _ joints: [String: SIMD3<Float>],
        scale: Float
    ) -> [String: SIMD3<Float>] {
        guard
            let leftHip = joints["leftHip"],
            let rightHip = joints["rightHip"]
        else {
            return joints.mapValues { $0 * scale }
        }

        let pelvis = (leftHip + rightHip) * 0.5
        return joints.mapValues { pelvis + ($0 - pelvis) * scale }
    }

    private func makeBodyReference(
        observation: VNHumanBodyPose3DObservation,
        cameraTransform: simd_float4x4,
        timestamp: TimeInterval,
        calibrationScale: Float
    ) -> BodyReference? {
        guard
            let rootCamera = cameraPosition(.root, observation),
            let centerShoulderCamera = cameraPosition(.centerShoulder, observation),
            let leftShoulder = cameraPosition(.leftShoulder, observation),
            let rightShoulder = cameraPosition(.rightShoulder, observation)
        else { return nil }

        let shoulderAxis = simd_normalize(rightShoulder - leftShoulder)
        let upAxis = simd_normalize(centerShoulderCamera - rootCamera)
        var forwardCamera = simd_cross(shoulderAxis, upAxis)
        if simd_length(forwardCamera) < 0.01 { return nil }
        forwardCamera = simd_normalize(forwardCamera)

        let root4 = cameraTransform * SIMD4<Float>(
            rootCamera.x,
            rootCamera.y,
            rootCamera.z,
            1
        )
        let shoulder4 = cameraTransform * SIMD4<Float>(
            centerShoulderCamera.x,
            centerShoulderCamera.y,
            centerShoulderCamera.z,
            1
        )
        let forward4 = cameraTransform * SIMD4<Float>(
            forwardCamera.x,
            forwardCamera.y,
            forwardCamera.z,
            0
        )

        let rootWorld = SIMD3<Float>(root4.x, root4.y, root4.z)
        let shoulderWorld = SIMD3<Float>(shoulder4.x, shoulder4.y, shoulder4.z)
        let forwardWorld = simd_normalize(
            SIMD3<Float>(forward4.x, forward4.y, forward4.z)
        )

        let yaw = atan2(forwardWorld.x, -forwardWorld.z)
        let torsoHeight = abs(shoulderWorld.y - rootWorld.y)

        return BodyReference(
            timestamp: timestamp,
            rootWorld: rootWorld,
            yawWorld: yaw,
            shoulderHeightMeters: torsoHeight,
            calibrationScale: calibrationScale
        )
    }

    private func cameraPosition(
        _ joint: VNHumanBodyPose3DObservation.JointName,
        _ observation: VNHumanBodyPose3DObservation
    ) -> SIMD3<Float>? {
        guard let matrix = try? observation.cameraRelativePosition(joint) else {
            return nil
        }
        let t = matrix.columns.3
        return SIMD3<Float>(t.x, t.y, t.z)
    }

    private func normalizedPositiveAngle(_ angle: Float) -> Float {
        var value = angle.truncatingRemainder(dividingBy: 2 * .pi)
        if value < 0 { value += 2 * .pi }
        return value
    }

    private func yawBin(_ degrees: Double) -> Int {
        let normalized = degrees.truncatingRemainder(dividingBy: 360)
        let width = 360.0 / Double(angularBinCount)
        return max(0, min(angularBinCount - 1, Int(normalized / width)))
    }

    private func accumulatePersonPointCloud(_ frame: ARFrame) {
        guard pointCloud.count < maxPointCloudPoints else { return }
        guard let segmentation = frame.segmentationBuffer else { return }

        let depthBuffer: CVPixelBuffer?
        if let personDepth = frame.estimatedDepthData {
            depthBuffer = personDepth
        } else if let lidar = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap {
            depthBuffer = lidar
        } else {
            return
        }

        guard let depthBuffer else { return }

        let bodyReference: BodyReference? = {
            dataLock.lock()
            defer { dataLock.unlock() }
            guard
                let reference = latestBodyReference,
                abs(reference.timestamp - frame.timestamp) < 0.55
            else { return nil }
            return reference
        }()

        guard let bodyReference else { return }

        let referenceYaw: Float? = {
            dataLock.lock()
            defer { dataLock.unlock() }
            if scanReferenceYaw == nil {
                scanReferenceYaw = bodyReference.yawWorld
            }
            return scanReferenceYaw
        }()

        guard let referenceYaw else { return }

        let deltaYaw = bodyReference.yawWorld - referenceYaw
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
            CVPixelBufferLockBaseAddress(segmentation, .readOnly)

            defer {
                CVPixelBufferUnlockBaseAddress(segmentation, .readOnly)
                CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            }

            guard
                let depthBase = CVPixelBufferGetBaseAddress(depthBuffer),
                let rawMask = CVPixelBufferGetBaseAddress(segmentation)
            else { return }

            let depthRowFloats = CVPixelBufferGetBytesPerRow(depthBuffer) / MemoryLayout<Float32>.size
            let depth = depthBase.assumingMemoryBound(to: Float32.self)

            let maskBase = rawMask.assumingMemoryBound(to: UInt8.self)
            let maskWidth = CVPixelBufferGetWidth(segmentation)
            let maskHeight = CVPixelBufferGetHeight(segmentation)
            let maskRowBytes = CVPixelBufferGetBytesPerRow(segmentation)

            let fx = intrinsics.columns.0.x
            let fy = intrinsics.columns.1.y
            let cx = intrinsics.columns.2.x
            let cy = intrinsics.columns.2.y
            let sampleStep = 5

            let c = cos(-deltaYaw)
            let s = sin(-deltaYaw)

            var newPoints: [Point3D] = []
            newPoints.reserveCapacity((depthWidth / sampleStep) * (depthHeight / sampleStep) / 3)

            for y in stride(from: 0, to: depthHeight, by: sampleStep) {
                for x in stride(from: 0, to: depthWidth, by: sampleStep) {
                    let mx = min(maskWidth - 1, x * maskWidth / depthWidth)
                    let my = min(maskHeight - 1, y * maskHeight / depthHeight)
                    let maskValue = maskBase[my * maskRowBytes + mx]
                    if maskValue < 128 { continue }

                    let z = depth[y * depthRowFloats + x]
                    if !z.isFinite || z < 0.5 || z > 5.0 { continue }

                    let cameraX = (Float(x) - cx) / fx * z
                    let cameraY = -(Float(y) - cy) / fy * z
                    let cameraPoint = SIMD4<Float>(cameraX, cameraY, -z, 1)
                    let worldPoint4 = cameraTransform * cameraPoint
                    let worldPoint = SIMD3<Float>(
                        worldPoint4.x,
                        worldPoint4.y,
                        worldPoint4.z
                    )

                    let rel = worldPoint - bodyReference.rootWorld
                    let canonicalX = c * rel.x + s * rel.z
                    let canonicalZ = -s * rel.x + c * rel.z

                    if abs(canonicalX) > 0.80 || abs(rel.y) > 1.40 || abs(canonicalZ) > 0.70 {
                        continue
                    }

                    newPoints.append(Point3D(
                        canonicalX,
                        rel.y,
                        canonicalZ
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
            version: "0.3-native-ios-body-analysis",
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
            chestSection: latestChestSection,
            waistSection: latestWaistSection,
            hipSection: latestHipSection,
            poseFrames: poseFrames,
            pointCloud: pointCloud
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(export)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "body-scan-v0.3-\(Int(Date().timeIntervalSince1970)).json"
                )
            try data.write(to: url, options: .atomic)
            return url
        } catch {
            statusText = "导出失败: \(error.localizedDescription)"
            return nil
        }
    }

    private func median(_ values: [Float]) -> Float? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[middle - 1] + sorted[middle]) * 0.5
        }
        return sorted[middle]
    }
}
