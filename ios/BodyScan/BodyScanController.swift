import ARKit
import Combine
import Foundation
import simd
import UIKit
import Vision

final class BodyScanController: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()
    @Published var inputHeightCm: Double = 178
    @Published private(set) var statusText = "准备中"
    @Published private(set) var isScanning = false
    @Published private(set) var isProcessing = false
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

    // All mutable scan data and AR/Vision callbacks share one serial queue.
    // UI writes stay on main. No main-thread sync, concurrent array mutation,
    // or post-scan preview result is allowed to overwrite the final measurements.
    private let pipeline = DispatchQueue(label: "BodyScan.Pipeline", qos: .userInitiated)
    private let mediaPipe = MediaPipePoseService()
    private var collecting = false
    private var finished = false
    private var heightCm: Double = 178
    private var startedAt: Date?
    private var lastVisionTimestamp: TimeInterval = 0
    private var lastMediaPipeTimestamp: TimeInterval = 0
    private var latestMP: (points: [PoseLandmark3D], timestamp: TimeInterval)?
    private var poseFrames: [PoseFrameRecord] = []
    private var metricSamples: [TailoringMetrics] = []
    private var torsoHeights: [Float] = []
    private var shoulderWidths: [Float] = []
    private var angularBins = Set<Int>()
    private var referenceYaw: Float?
    private var depthSource = "none"
    private var acceptedDepthFrames = 0
    private var rejectedFrames: [String: Int] = [:]
    private var totalFrames = 0
    private var mpFrames = 0
    private var workingMetrics = TailoringMetrics.empty
    private var pendingSkeleton: (timestamp: TimeInterval, joints: [String: SIMD3<Float>],
                                  sampleIndex: Int, recordIndex: Int)?
    // Equal budgets by angle prevent the first front view using all memory.
    private var anglePoints = [[Point3D]](repeating: [], count: ScanGeometry.yawBinCount)
    private var angleVoxels = [Set<Voxel>](repeating: [], count: ScanGeometry.yawBinCount)
    private let maxPointsPerAngle = 5_000

    private struct Voxel: Hashable {
        let x: Int
        let y: Int
        let z: Int
        init(_ p: SIMD3<Float>) {
            x = Int((p.x / 0.005).rounded())
            y = Int((p.y / 0.005).rounded())
            z = Int((p.z / 0.005).rounded())
        }
    }

    override init() {
        super.init()
        session.delegateQueue = pipeline
        session.delegate = self
        mediaPipe.onStatus = { [weak self] text in
            DispatchQueue.main.async { self?.mediaPipeStatus = text }
        }
        mediaPipe.onResult = { [weak self] points, timestampMs in
            guard let self else { return }
            self.pipeline.async {
                self.latestMP = (points, TimeInterval(timestampMs) / 1000)
                if self.collecting {
                    self.mpFrames += 1
                    self.applyMediaPipe(points, timestamp: TimeInterval(timestampMs) / 1000)
                }
                DispatchQueue.main.async { self.mediaPipeJointCount = points.count }
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
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            config.frameSemantics.insert(.personSegmentationWithDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            config.frameSemantics.insert(.personSegmentation)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        // Temporal smoothing can trail a rotating body; prefer current raw depth.
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        statusText = "摄像头已启动"
    }

    func beginScan() {
        guard !isScanning, !isProcessing else { return }
        guard inputHeightCm.isFinite, (120...230).contains(inputHeightCm) else {
            statusText = "请输入 120–230 cm 的真实身高"
            return
        }
        let input = inputHeightCm
        isProcessing = true
        pipeline.async {
            self.clearScan()
            self.heightCm = input
            self.startedAt = Date()
            self.collecting = true
            DispatchQueue.main.async {
                self.clearDisplay()
                self.isProcessing = false
                self.isScanning = true
                self.statusText = "请保持站直，缓慢转一圈"
            }
        }
    }

    func finishScan() {
        guard isScanning, !isProcessing else { return }
        isProcessing = true
        statusText = "正在检查扫描质量与身体截面"
        pipeline.async {
            self.collecting = false
            self.finished = true
            self.finalizeScan()
        }
    }

    func resetScan() {
        guard !isProcessing else { return }
        isProcessing = true
        pipeline.async {
            self.clearScan()
            DispatchQueue.main.async {
                self.clearDisplay()
                self.isProcessing = false
                self.isScanning = false
                self.statusText = "已重置"
            }
        }
    }

    private func clearScan() {
        collecting = false; finished = false; startedAt = nil
        poseFrames = []; metricSamples = []; torsoHeights = []; shoulderWidths = []
        angularBins = []; referenceYaw = nil; depthSource = "none"
        rejectedFrames = [:]; acceptedDepthFrames = 0; totalFrames = 0; mpFrames = 0
        latestMP = nil; pendingSkeleton = nil; workingMetrics = .empty
        anglePoints = [[Point3D]](repeating: [], count: ScanGeometry.yawBinCount)
        angleVoxels = [Set<Voxel>](repeating: [], count: ScanGeometry.yawBinCount)
    }

    private func clearDisplay() {
        pointCloudCount = 0; scanFrameCount = 0; angularCoverage = 0
        latestMetrics = .empty; exportURL = nil
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let person = frame.estimatedDepthData != nil && frame.segmentationBuffer != nil
        let lidar = frame.sceneDepth != nil
        DispatchQueue.main.async {
            self.personDepthAvailable = person
            self.lidarSceneDepthAvailable = lidar
        }
        guard !finished else { return }
        if collecting { totalFrames += 1 }
        if frame.timestamp - lastMediaPipeTimestamp >= 0.18 {
            lastMediaPipeTimestamp = frame.timestamp
            mediaPipe.process(pixelBuffer: frame.capturedImage, timestampMs: Int(frame.timestamp * 1000))
        }
        guard frame.timestamp - lastVisionTimestamp >= 0.18 else { return }
        lastVisionTimestamp = frame.timestamp
        guard case .normal = frame.camera.trackingState else {
            reject("tracking_not_normal")
            return
        }
        analyze(frame)
    }

    private func reject(_ reason: String) {
        if collecting { rejectedFrames[reason, default: 0] += 1 }
    }

    private func analyze(_ frame: ARFrame) {
        do {
            let request = VNDetectHumanBodyPose3DRequest()
            try VNImageRequestHandler(cvPixelBuffer: frame.capturedImage,
                                      orientation: .right, options: [:]).perform([request])
            guard let observation = request.results?.first,
                  let scale = ScanGeometry.skeletonScale(inputHeightCm: heightCm,
                                                         visionHeightMeters: observation.bodyHeight) else {
                reject("no_valid_pose")
                return
            }
            let cameraJoints = projectedCameraJoints(observation, frame: frame, scale: scale)
            guard cameraJoints.count >= 12 else { reject("joints_out_of_frame"); return }
            let worldJoints = cameraJoints.mapValues { point -> SIMD3<Float> in
                let p = frame.camera.transform * SIMD4<Float>(point.x, point.y, point.z, 1)
                return SIMD3<Float>(p.x, p.y, p.z)
            }
            guard let body = ScanGeometry.bodyFrame(joints: worldJoints) else {
                reject("invalid_or_leaning_pose")
                return
            }
            let mp = latestMP.flatMap { value -> [Point3D]? in
                guard ScanGeometry.synchronized(value.timestamp, frame.timestamp), value.points.count == 33,
                      [11,12,13,14,15,16,23,24,25,26,27,28].allSatisfy({
                          value.points[$0].visibility >= 0.7 && value.points[$0].presence >= 0.7
                      }) else { return nil }
                return value.points.map { $0.point }
            }
            let fused = mp.flatMap {
                BodyAnalysisEngine.fuseSkeletons(apple: worldJoints, mediaPipe: $0, appleCalibrationScale: 1)
            } ?? FusedSkeletonResult(joints: worldJoints, agreementCm: nil)
            let measured = observation.heightEstimation == .measured
            let metrics = BodyAnalysisEngine.metrics(fused: fused,
                visionBodyHeightCm: measured ? Double(observation.bodyHeight * 100) : nil)
            workingMetrics = metrics
            if collecting {
                metricSamples.append(metrics)
                torsoHeights.append(body.torsoHeight)
                if let ls = worldJoints["leftShoulder"], let rs = worldJoints["rightShoulder"] {
                    shoulderWidths.append(simd_distance(ls, rs))
                }
                let transform = (0..<4).flatMap { column in
                    (0..<4).map { row in frame.camera.transform[column][row] }
                }
                let intrinsics = (0..<3).flatMap { column in
                    (0..<3).map { row in frame.camera.intrinsics[column][row] }
                }
                poseFrames.append(PoseFrameRecord(timestamp: frame.timestamp,
                    appleJoints: worldJoints.mapValues { Point3D($0.x, $0.y, $0.z) },
                    mediaPipeWorldJoints: mp ?? [],
                    fusedJoints: fused.joints.mapValues { Point3D($0.x, $0.y, $0.z) },
                    bodyYawDegrees: Double(body.yaw) * 180 / .pi,
                    bodyHeightEstimateMeters: observation.bodyHeight,
                    skeletonAgreementCm: fused.agreementCm, heightIsMeasured: measured,
                    skeletonScale: scale, cameraTransform: transform, cameraIntrinsics: intrinsics))
                pendingSkeleton = (frame.timestamp, worldJoints, metricSamples.count - 1, poseFrames.count - 1)
                // Pose, segmentation, intrinsics and depth from the SAME ARFrame.
                sampleDepth(frame, body: body)
            }
            let count = anglePoints.reduce(0) { $0 + $1.count }
            let coverage = Double(angularBins.count) / Double(ScanGeometry.yawBinCount)
            let frames = totalFrames
            DispatchQueue.main.async {
                self.appleJointCount = cameraJoints.count
                self.latestMetrics = metrics
                self.pointCloudCount = count
                self.angularCoverage = coverage
                self.scanFrameCount = frames
            }
        } catch { reject("vision_error") }
    }

    private func projectedCameraJoints(_ observation: VNHumanBodyPose3DObservation,
                                       frame: ARFrame, scale: Float) -> [String: SIMD3<Float>] {
        let names: [(String, VNHumanBodyPose3DObservation.JointName)] = [
            ("root", .root), ("centerShoulder", .centerShoulder),
            ("leftShoulder", .leftShoulder), ("rightShoulder", .rightShoulder),
            ("leftElbow", .leftElbow), ("rightElbow", .rightElbow),
            ("leftWrist", .leftWrist), ("rightWrist", .rightWrist),
            ("leftHip", .leftHip), ("rightHip", .rightHip),
            ("leftKnee", .leftKnee), ("rightKnee", .rightKnee),
            ("leftAnkle", .leftAnkle), ("rightAnkle", .rightAnkle)
        ]
        let k = frame.camera.intrinsics
        let size = frame.camera.imageResolution
        var joints: [String: SIMD3<Float>] = [:]
        for (name, joint) in names {
            guard let position = try? observation.cameraRelativePosition(joint),
                  let image = try? observation.pointInImage(joint),
                  let raw = ScanGeometry.rawImagePoint(portraitPoint: SIMD2<Float>(Float(image.x), Float(image.y))),
                  let camera = ScanGeometry.unproject(
                    pixel: SIMD2<Float>(raw.x * Float(size.width), raw.y * Float(size.height)),
                    depth: abs(position.columns.3.z) * scale,
                    fx: k.columns.0.x, fy: k.columns.1.y, cx: k.columns.2.x, cy: k.columns.2.y) else { continue }
            joints[name] = camera
        }
        return joints
    }

    // MediaPipe is asynchronous. Fuse its matching Apple frame after it arrives;
    // otherwise the serial Vision stage would always consume the previous pose.
    private func applyMediaPipe(_ landmarks: [PoseLandmark3D], timestamp: TimeInterval) {
        guard let pending = pendingSkeleton,
              ScanGeometry.synchronized(timestamp, pending.timestamp), landmarks.count == 33,
              [11,12,13,14,15,16,23,24,25,26,27,28].allSatisfy({
                  landmarks[$0].visibility >= 0.7 && landmarks[$0].presence >= 0.7
              }), let fused = BodyAnalysisEngine.fuseSkeletons(apple: pending.joints,
                  mediaPipe: landmarks.map(\.point), appleCalibrationScale: 1) else { return }
        let metrics = BodyAnalysisEngine.metrics(fused: fused,
            visionBodyHeightCm: metricSamples[pending.sampleIndex].visionBodyHeightCm)
        metricSamples[pending.sampleIndex] = metrics
        poseFrames[pending.recordIndex].mediaPipeWorldJoints = landmarks.map(\.point)
        poseFrames[pending.recordIndex].fusedJoints = fused.joints.mapValues { Point3D($0.x, $0.y, $0.z) }
        poseFrames[pending.recordIndex].skeletonAgreementCm = fused.agreementCm
        workingMetrics = metrics
        DispatchQueue.main.async { self.latestMetrics = metrics }
    }

    private func sampleDepth(_ frame: ARFrame, body: ScanGeometry.BodyFrame) {
        guard let mask = frame.segmentationBuffer else { reject("no_segmentation"); return }
        let source: String
        let depth: CVPixelBuffer
        let confidence: CVPixelBuffer?
        if let lidar = frame.sceneDepth {
            source = "lidar_scene_depth"; depth = lidar.depthMap; confidence = lidar.confidenceMap
            guard confidence != nil else { reject("no_lidar_confidence"); return }
        } else if let inferred = frame.estimatedDepthData {
            source = "inferred_person_depth"; depth = inferred; confidence = nil
        } else { reject("no_depth"); return }
        guard depthSource == "none" || depthSource == source else { reject("depth_source_changed"); return }
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32,
              CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent8 else {
            reject("unsupported_depth_or_mask_format"); return
        }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        let mw = CVPixelBufferGetWidth(mask), mh = CVPixelBufferGetHeight(mask)
        let imageSize = frame.camera.imageResolution
        guard w > 2, h > 2, mw > 4, mh > 4,
              abs(Double(w) / Double(h) - imageSize.width / imageSize.height) < 0.02 else {
            reject("depth_registration_mismatch"); return
        }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        if let confidence { CVPixelBufferLockBaseAddress(confidence, .readOnly) }
        defer {
            if let confidence { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
            CVPixelBufferUnlockBaseAddress(depth, .readOnly)
        }
        guard let db = CVPixelBufferGetBaseAddress(depth), let mb = CVPixelBufferGetBaseAddress(mask) else {
            reject("empty_depth_buffer"); return
        }
        let depths = db.assumingMemoryBound(to: Float32.self)
        let masks = mb.assumingMemoryBound(to: UInt8.self)
        let ds = CVPixelBufferGetBytesPerRow(depth) / MemoryLayout<Float32>.size
        let ms = CVPixelBufferGetBytesPerRow(mask)
        let conf = confidence.flatMap { CVPixelBufferGetBaseAddress($0)?.assumingMemoryBound(to: UInt8.self) }
        let cw = confidence.map { CVPixelBufferGetWidth($0) } ?? 0
        let ch = confidence.map { CVPixelBufferGetHeight($0) } ?? 0
        let cs = confidence.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
        guard confidence == nil || (conf != nil && cw > 0 && ch > 0) else {
            reject("empty_confidence_buffer"); return
        }
        var k = frame.camera.intrinsics
        let sx = Float(w) / Float(imageSize.width), sy = Float(h) / Float(imageSize.height)
        k.columns.0.x *= sx; k.columns.1.y *= sy
        k.columns.2.x *= sx; k.columns.2.y *= sy
        let yaw = referenceYaw ?? body.yaw
        let bin = ScanGeometry.yawBin(body.yaw - yaw)
        var accepted = 0
        for y in stride(from: 1, to: h - 1, by: 2) {
            for x in stride(from: 1, to: w - 1, by: 2) {
                let mx = x * mw / w, my = y * mh / h
                // Reject mixed foreground/background pixels at mask boundaries.
                guard mx >= 2, my >= 2, mx < mw - 2, my < mh - 2,
                      [(-2,0),(2,0),(0,-2),(0,2),(0,0)].allSatisfy({
                          masks[(my + $0.1) * ms + mx + $0.0] >= 200
                      }) else { continue }
                if let conf, conf[(y * ch / h) * cs + x * cw / w] < 2 { continue }
                let z = depths[y * ds + x]
                guard z.isFinite, [depths[y * ds + x - 1], depths[y * ds + x + 1],
                    depths[(y - 1) * ds + x], depths[(y + 1) * ds + x]].allSatisfy({
                        $0.isFinite && abs($0 - z) < 0.04
                    }), let camera = ScanGeometry.unproject(pixel: SIMD2<Float>(Float(x), Float(y)), depth: z,
                        fx: k.columns.0.x, fy: k.columns.1.y, cx: k.columns.2.x, cy: k.columns.2.y) else { continue }
                let p = frame.camera.transform * SIMD4<Float>(camera.x, camera.y, camera.z, 1)
                let canonical = body.canonicalPoint(SIMD3<Float>(p.x, p.y, p.z))
                guard ScanGeometry.isFinite(canonical), abs(canonical.x) < 0.8,
                      abs(canonical.y) < 1.4, abs(canonical.z) < 0.7 else { continue }
                accepted += 1
                if anglePoints[bin].count < maxPointsPerAngle,
                   angleVoxels[bin].insert(Voxel(canonical)).inserted {
                    anglePoints[bin].append(Point3D(canonical.x, canonical.y, canonical.z))
                }
            }
        }
        if accepted >= 30 {
            referenceYaw = yaw; depthSource = source
            angularBins.insert(bin); acceptedDepthFrames += 1
        } else { reject("too_few_valid_depth_pixels") }
    }

    private func finalizeScan() {
        let points = anglePoints.flatMap { $0 }
        let coverage = Double(angularBins.count) / Double(ScanGeometry.yawBinCount)
        let sections = BodyAnalysisEngine.analyzeCrossSections(canonicalPoints: points,
            torsoHeightMeters: median(torsoHeights.map(Double.init)) ?? .nan,
            shoulderWidthMeters: median(shoulderWidths.map(Double.init)) ?? .nan, angularCoverage: coverage)
        var metrics = workingMetrics
        metrics.shoulderWidthCm = sampleMedian(\.shoulderWidthCm)
        metrics.shoulderHeightDifferenceCm = sampleMedian(\.shoulderHeightDifferenceCm)
        metrics.leftArmLengthCm = sampleMedian(\.leftArmLengthCm)
        metrics.rightArmLengthCm = sampleMedian(\.rightArmLengthCm)
        metrics.skeletonAgreementCm = sampleMedian(\.skeletonAgreementCm)
        metrics.chestCircumferenceCm = sections.chest?.perimeterCm
        metrics.waistCircumferenceCm = sections.waist?.perimeterCm
        metrics.hipCircumferenceCm = sections.hip?.perimeterCm
        metrics.chestWidthCm = sections.chest?.widthCm; metrics.chestDepthCm = sections.chest?.depthCm
        metrics.waistWidthCm = sections.waist?.widthCm; metrics.waistDepthCm = sections.waist?.depthCm
        metrics.hipWidthCm = sections.hip?.widthCm; metrics.hipDepthCm = sections.hip?.depthCm
        let allSections = [sections.chest, sections.waist, sections.hip].compactMap { $0 }
        metrics.sectionConfidence = allSections.count == 3 ? allSections.map(\.confidence).min() : nil
        var warnings = ["研究用估算；采样完整度不是准确率，须与版师实测对照。",
                        "骨架位置仍由视觉模型估计；深度与骨架的配准偏差须用真机验证。"]
        if depthSource != "lidar_scene_depth" {
            warnings.append("没有使用硬件 LiDAR 深度；推断人体深度不能视为真实表面测量。")
        }
        if allSections.count < 3 { warnings.append("截面或角度覆盖不足，请重扫；缺失围度不予补猜。") }
        let duration = startedAt.map { Date().timeIntervalSince($0) } ?? 0
        let export = ScanExport(version: "0.3.1-accuracy-audit", createdAt: Date(), inputHeightCm: heightCm,
            durationSeconds: duration, capturedFrameCount: totalFrames, applePoseFrameCount: poseFrames.count,
            mediaPipeFrameCount: mpFrames, angularCoveragePercent: coverage * 100,
            pointCloudPointCount: points.count, personDepthAvailable: depthSource == "inferred_person_depth",
            lidarSceneDepthAvailable: depthSource == "lidar_scene_depth", latestMetrics: metrics,
            chestSection: sections.chest, waistSection: sections.waist, hipSection: sections.hip,
            poseFrames: poseFrames, pointCloud: points, depthSource: depthSource,
            rejectedFrames: rejectedFrames, acceptedDepthFrames: acceptedDepthFrames,
            measurementWarnings: warnings, depthScaleApplied: 1)
        var url: URL?
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let target = FileManager.default.temporaryDirectory
                .appendingPathComponent("body-scan-v0.3.1-\(UUID().uuidString).json")
            try encoder.encode(export).write(to: target, options: .atomic)
            url = target
        } catch { /* Keep measurements visible and report export failure below. */ }
        let resultURL = url
        let finalMetrics = metrics
        let message = resultURL == nil ? "结果导出失败，请重试"
            : allSections.count == 3 ? "已生成研究用截面，请与版师实测对照" : "采样不足，请重扫；诊断数据可导出"
        DispatchQueue.main.async {
            self.latestMetrics = finalMetrics; self.exportURL = resultURL
            self.angularCoverage = coverage; self.pointCloudCount = points.count
            self.isScanning = false; self.isProcessing = false; self.statusText = message
        }
    }

    private func sampleMedian(_ key: KeyPath<TailoringMetrics, Double?>) -> Double? {
        median(metricSamples.compactMap { $0[keyPath: key] })
    }
    private func median(_ input: [Double]) -> Double? {
        let values = input.filter(\.isFinite).sorted()
        guard !values.isEmpty else { return nil }
        return (values[(values.count - 1) / 2] + values[values.count / 2]) / 2
    }
}
