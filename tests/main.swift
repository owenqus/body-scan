import Foundation
import simd

var checks = 0
func check(_ condition: @autoclosure () -> Bool, _ message: String) {
    checks += 1
    guard condition() else { fatalError("FAIL: \(message)") }
    print("PASS: \(message)")
}
func near(_ actual: Double, _ expected: Double, _ tolerance: Double, _ message: String) {
    check(actual.isFinite && abs(actual - expected) <= tolerance,
          "\(message): got \(actual), expected \(expected) ± \(tolerance)")
}

// Independent oracle: numerical integration of the exact ellipse arc length.
func ellipsePerimeter(_ a: Double, _ b: Double) -> Double {
    let count = 100_000
    let step = 2 * Double.pi / Double(count)
    return (0..<count).reduce(0.0) { total, i in
        let theta = (Double(i) + 0.5) * step
        return total + hypot(a * sin(theta), b * cos(theta)) * step
    } * 100
}

// Three different known elliptical cylinders, sampled as surfaces rather than
// generated through the production measurement code. No real-person accuracy
// can be inferred from these geometry regressions.
func cloud(partial: Bool = false, noisy: Bool = false) -> [Point3D] {
    var points: [Point3D] = []
    let regions: [(Double, Double, Double, Double)] = [
        (0.27, 0.43, 0.20, 0.14), // chest
        (0.09, 0.26, 0.16, 0.11), // waist
        (-0.14, 0.065, 0.21, 0.15) // hip
    ]
    for (low, high, a, b) in regions {
        for row in 0...40 {
            let y = low + (high - low) * Double(row) / 40
            for index in 0..<720 {
                let theta = 2 * Double.pi * Double(index) / 720
                if partial && theta > Double.pi { continue }
                let noise = noisy ? 0.001 * sin(Double(index * 71 + row * 23)) : 0
                points.append(Point3D(Float((a + noise) * cos(theta)), Float(y), Float((b + noise) * sin(theta))))
                if noisy && index % 101 == 0 {
                    points.append(Point3D(Float(a * 1.12 * cos(theta)), Float(y), Float(b * 1.12 * sin(theta))))
                }
            }
        }
    }
    return points
}
func analyze(_ points: [Point3D], coverage: Double = 1) -> SectionAnalysis {
    BodyAnalysisEngine.analyzeCrossSections(canonicalPoints: points,
        torsoHeightMeters: 0.55, shoulderWidthMeters: 0.43, angularCoverage: coverage)
}

let full = cloud()
let result = analyze(full)
check(result.chest != nil && result.waist != nil && result.hip != nil, "complete surfaces produce all three sections")
near(result.chest!.perimeterCm, ellipsePerimeter(0.20, 0.14), 0.6, "chest perimeter")
near(result.waist!.perimeterCm, ellipsePerimeter(0.16, 0.11), 0.6, "waist perimeter")
near(result.hip!.perimeterCm, ellipsePerimeter(0.21, 0.15), 0.6, "hip perimeter")
near(result.chest!.widthCm, 40, 0.5, "chest width")
near(result.chest!.depthCm, 28, 0.5, "chest depth")
check(analyze(full, coverage: 0.5).chest == nil, "incomplete turn cannot produce a circumference")
check(analyze(cloud(partial: true)).chest == nil, "missing back surface rejected even if global turn says complete")
let duplicates = Array(repeating: Array(full.prefix(20)), count: 100).flatMap { $0 }
check(analyze(duplicates).chest == nil, "duplicate pixels cannot manufacture surface coverage")
let noisy = analyze(cloud(noisy: true))
check(noisy.chest != nil && noisy.waist != nil && noisy.hip != nil, "small noise and sparse outliers remain measurable")
near(noisy.chest!.perimeterCm, ellipsePerimeter(0.20, 0.14), 0.8, "sparse outliers do not inflate chest hull")
let dirty = full + [Point3D(.nan, 0.35, 0), Point3D(.infinity, 0.15, 0)]
near(analyze(dirty).chest!.perimeterCm, result.chest!.perimeterCm, 0.001, "nonfinite points ignored safely")
check(BodyAnalysisEngine.analyzeCrossSections(canonicalPoints: full, torsoHeightMeters: .nan,
    shoulderWidthMeters: 0.43, angularCoverage: 1).chest == nil, "invalid scale rejected")

// Preserve the true axes and translation through every initial angle, including
// reverse turns and nonzero camera-relative starting headings.
func rotate(_ p: SIMD3<Float>, yaw: Float) -> SIMD3<Float> {
    SIMD3<Float>(cos(yaw) * p.x + sin(yaw) * p.z, p.y,
                 -sin(yaw) * p.x + cos(yaw) * p.z)
}
let pelvis = SIMD3<Float>(0.3, 1, -2.4)
let local: [String: SIMD3<Float>] = [
    "leftHip": SIMD3<Float>(-0.15, 0, 0), "rightHip": SIMD3<Float>(0.15, 0, 0),
    "leftShoulder": SIMD3<Float>(-0.215, 0.55, 0), "rightShoulder": SIMD3<Float>(0.215, 0.55, 0)
]
let probe = SIMD3<Float>(0.18, 0.35, 0.10)
for degrees in [0, 37, 90, 179, 225, 315, -80] {
    let yaw = Float(degrees) * .pi / 180
    let joints = local.mapValues { rotate($0, yaw: yaw) + pelvis }
    let basis = ScanGeometry.bodyFrame(joints: joints)!
    let recovered = basis.canonicalPoint(rotate(probe, yaw: yaw) + pelvis)
    near(Double(simd_distance(recovered, probe)), 0, 0.00001, "absolute body axes at \(degrees) degrees")
}
check(ScanGeometry.yawBin(-.pi / 2) == 18, "reverse-turn yaw bin wraps correctly")
check(ScanGeometry.synchronized(1, 1.05), "50 ms matched pose accepted")
check(!ScanGeometry.synchronized(1, 1.5), "old half-second pose rejected")
check(ScanGeometry.skeletonScale(inputHeightCm: .nan, visionHeightMeters: 1.8) == nil, "NaN height rejected")
check(ScanGeometry.skeletonScale(inputHeightCm: 0, visionHeightMeters: 1.8) == nil, "zero height rejected")
near(Double(ScanGeometry.skeletonScale(inputHeightCm: 162, visionHeightMeters: 1.8)!), 0.9, 0.0001,
     "reference-height correction belongs to skeleton only")
let raw = ScanGeometry.rawImagePoint(portraitPoint: SIMD2<Float>(0.8, 0.7))!
near(Double(raw.x), 0.3, 0.00001, "portrait rotation undone in raw sensor x")
near(Double(raw.y), 0.2, 0.00001, "portrait rotation undone in raw sensor y")
let center = ScanGeometry.unproject(pixel: SIMD2<Float>(320, 240), depth: 2,
                                   fx: 500, fy: 500, cx: 320, cy: 240)!
near(Double(center.z), -2, 0.00001, "metric depth retained exactly")
let offAxis = ScanGeometry.unproject(pixel: SIMD2<Float>(420, 190), depth: 2,
                                    fx: 500, fy: 500, cx: 320, cy: 240)!
near(Double(offAxis.x), 0.4, 0.00001, "pinhole horizontal projection")
near(Double(offAxis.y), 0.2, 0.00001, "AR y axis projection")
check(ScanGeometry.unproject(pixel: .zero, depth: .nan, fx: 1, fy: 1, cx: 0, cy: 0) == nil,
      "invalid depth rejected")

// A conflicting wrist must not drag a plausible Apple wrist by fixed weights.
var skeleton = local
skeleton["leftElbow"] = SIMD3<Float>(-0.26, 0.30, 0)
skeleton["rightElbow"] = SIMD3<Float>(0.26, 0.30, 0)
skeleton["leftWrist"] = SIMD3<Float>(-0.28, 0.05, 0)
skeleton["rightWrist"] = SIMD3<Float>(0.28, 0.05, 0)
skeleton["leftKnee"] = SIMD3<Float>(-0.13, -0.45, 0)
skeleton["rightKnee"] = SIMD3<Float>(0.13, -0.45, 0)
skeleton["leftAnkle"] = SIMD3<Float>(-0.13, -0.85, 0)
skeleton["rightAnkle"] = SIMD3<Float>(0.13, -0.85, 0)
var mp = [Point3D](repeating: Point3D(0, 0, 0), count: 33)
for (name, index) in ["leftShoulder":11,"rightShoulder":12,"leftElbow":13,"rightElbow":14,
                     "leftWrist":15,"rightWrist":16,"leftHip":23,"rightHip":24,
                     "leftKnee":25,"rightKnee":26,"leftAnkle":27,"rightAnkle":28] {
    let p = skeleton[name]!
    mp[index] = Point3D(p.x, p.y, p.z)
}
mp[15] = Point3D(-0.8, 0.05, 0)
let fused = BodyAnalysisEngine.fuseSkeletons(apple: skeleton, mediaPipe: mp, appleCalibrationScale: 1)!
near(Double(simd_distance(fused.joints["leftWrist"]!, skeleton["leftWrist"]!)), 0, 0.00001,
     "conflicting wrist excluded from fusion")
check((fused.agreementCm ?? 0) > 0, "fusion disagreement remains visible")
print("All \(checks) geometry regressions passed. These are synthetic tests, not human accuracy validation.")
