import Foundation
import simd

/// All distances here are metres. ARKit depth is already metric: never apply
/// Vision's reference-height correction to depth points.
enum ScanGeometry {
    static let yawBinCount = 24

    static func isFinite(_ p: SIMD3<Float>) -> Bool {
        p.x.isFinite && p.y.isFinite && p.z.isFinite
    }

    static func skeletonScale(inputHeightCm: Double, visionHeightMeters: Float) -> Float? {
        guard inputHeightCm.isFinite, (120...230).contains(inputHeightCm),
              visionHeightMeters.isFinite, (0.8...2.5).contains(visionHeightMeters) else { return nil }
        return Float(inputHeightCm / 100) / visionHeightMeters
    }

    /// Vision receives an EXIF .right image (portrait UI is locked). Undo that
    /// clockwise rotation and the lower-left Vision origin to obtain raw sensor
    /// coordinates. Use the projection to avoid mixing oriented Vision axes
    /// with the landscape AR camera coordinate system.
    static func rawImagePoint(portraitPoint: SIMD2<Float>) -> SIMD2<Float>? {
        guard portraitPoint.x.isFinite, portraitPoint.y.isFinite,
              (0...1).contains(portraitPoint.x), (0...1).contains(portraitPoint.y) else { return nil }
        return SIMD2<Float>(1 - portraitPoint.y, 1 - portraitPoint.x)
    }

    static func unproject(pixel: SIMD2<Float>, depth: Float, fx: Float, fy: Float,
                          cx: Float, cy: Float) -> SIMD3<Float>? {
        guard depth.isFinite, (0.3...5).contains(depth), fx.isFinite, fy.isFinite,
              fx > 0, fy > 0, pixel.x.isFinite, pixel.y.isFinite,
              cx.isFinite, cy.isFinite else { return nil }
        return SIMD3<Float>((pixel.x - cx) / fx * depth,
                            -(pixel.y - cy) / fy * depth, -depth)
    }

    struct BodyFrame {
        var origin: SIMD3<Float>
        var right: SIMD3<Float>
        var forward: SIMD3<Float>
        var torsoHeight: Float
        var yaw: Float { atan2(forward.x, forward.z) }

        func canonicalPoint(_ world: SIMD3<Float>) -> SIMD3<Float> {
            let delta = world - origin
            // Absolute body axes remove the initial turn angle as well as the
            // current turn. Gravity stays vertical; high/low shoulders survive.
            return SIMD3<Float>(simd_dot(delta, right), delta.y, simd_dot(delta, forward))
        }
    }

    static func bodyFrame(joints: [String: SIMD3<Float>]) -> BodyFrame? {
        guard let lh = joints["leftHip"], let rh = joints["rightHip"],
              let ls = joints["leftShoulder"], let rs = joints["rightShoulder"],
              [lh, rh, ls, rs].allSatisfy(isFinite) else { return nil }
        let pelvis = (lh + rh) * 0.5
        let up = (ls + rs) * 0.5 - pelvis
        let side = SIMD3<Float>(rs.x - ls.x, 0, rs.z - ls.z)
        guard simd_length(side) > 0.15, up.y > 0.25,
              up.y / simd_length(up) > cos(Float.pi / 9) else { return nil }
        let right = simd_normalize(side)
        let forward = simd_cross(right, SIMD3<Float>(0, 1, 0))
        return BodyFrame(origin: pelvis, right: right, forward: forward, torsoHeight: up.y)
    }

    static func yawBin(_ radians: Float) -> Int {
        guard radians.isFinite else { return 0 }
        let circle = 2 * Float.pi
        let angle = (radians.truncatingRemainder(dividingBy: circle) + circle)
            .truncatingRemainder(dividingBy: circle)
        return min(yawBinCount - 1, Int(angle / circle * Float(yawBinCount)))
    }

    static func synchronized(_ a: TimeInterval, _ b: TimeInterval) -> Bool {
        a.isFinite && b.isFinite && abs(a - b) <= 0.075
    }
}
