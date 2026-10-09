import Foundation

struct Point3D: Codable, Hashable {
    var x: Float
    var y: Float
    var z: Float

    init(_ x: Float, _ y: Float, _ z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }
}

struct CrossSectionMeasurement: Codable {
    var yCmRelativePelvis: Double
    var perimeterCm: Double
    var widthCm: Double
    var depthCm: Double
    var sampleCount: Int
    var hullPointCount: Int
    var confidence: Double
}

struct TailoringMetrics: Codable {
    var visionBodyHeightCm: Double?
    var shoulderWidthCm: Double?
    var shoulderHeightDifferenceCm: Double?
    var leftArmLengthCm: Double?
    var rightArmLengthCm: Double?
    var skeletonAgreementCm: Double?

    var chestCircumferenceCm: Double?
    var waistCircumferenceCm: Double?
    var hipCircumferenceCm: Double?

    var chestWidthCm: Double?
    var chestDepthCm: Double?
    var waistWidthCm: Double?
    var waistDepthCm: Double?
    var hipWidthCm: Double?
    var hipDepthCm: Double?

    var sectionConfidence: Double?

    static let empty = TailoringMetrics(
        visionBodyHeightCm: nil,
        shoulderWidthCm: nil,
        shoulderHeightDifferenceCm: nil,
        leftArmLengthCm: nil,
        rightArmLengthCm: nil,
        skeletonAgreementCm: nil,
        chestCircumferenceCm: nil,
        waistCircumferenceCm: nil,
        hipCircumferenceCm: nil,
        chestWidthCm: nil,
        chestDepthCm: nil,
        waistWidthCm: nil,
        waistDepthCm: nil,
        hipWidthCm: nil,
        hipDepthCm: nil,
        sectionConfidence: nil
    )
}

struct PoseFrameRecord: Codable {
    var timestamp: TimeInterval
    var appleJoints: [String: Point3D]
    var mediaPipeWorldJoints: [Point3D]
    var fusedJoints: [String: Point3D]
    var bodyYawDegrees: Double?
    var bodyHeightEstimateMeters: Float?
    var skeletonAgreementCm: Double?
}

struct ScanExport: Codable {
    var version: String
    var createdAt: Date
    var inputHeightCm: Double
    var durationSeconds: Double
    var capturedFrameCount: Int
    var applePoseFrameCount: Int
    var mediaPipeFrameCount: Int
    var angularCoveragePercent: Double
    var pointCloudPointCount: Int
    var personDepthAvailable: Bool
    var lidarSceneDepthAvailable: Bool
    var latestMetrics: TailoringMetrics
    var chestSection: CrossSectionMeasurement?
    var waistSection: CrossSectionMeasurement?
    var hipSection: CrossSectionMeasurement?
    var poseFrames: [PoseFrameRecord]
    var pointCloud: [Point3D]
}
