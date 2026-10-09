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

struct TailoringMetrics: Codable {
    var visionBodyHeightCm: Double?
    var shoulderWidthCm: Double?
    var shoulderHeightDifferenceCm: Double?
    var leftArmLengthCm: Double?
    var rightArmLengthCm: Double?

    static let empty = TailoringMetrics(
        visionBodyHeightCm: nil,
        shoulderWidthCm: nil,
        shoulderHeightDifferenceCm: nil,
        leftArmLengthCm: nil,
        rightArmLengthCm: nil
    )
}

struct PoseFrameRecord: Codable {
    var timestamp: TimeInterval
    var appleJoints: [String: Point3D]
    var mediaPipeWorldJoints: [Point3D]
    var bodyYawDegrees: Double?
    var bodyHeightEstimateMeters: Float?
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
    var poseFrames: [PoseFrameRecord]
    var pointCloud: [Point3D]
}
