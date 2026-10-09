import Foundation
import simd

struct FusedSkeletonResult {
    var joints: [String: SIMD3<Float>]
    var agreementCm: Double?
}

struct SectionAnalysis {
    var chest: CrossSectionMeasurement?
    var waist: CrossSectionMeasurement?
    var hip: CrossSectionMeasurement?
}

enum BodyAnalysisEngine {
    private static let commonMediaPipeIndices: [String: Int] = [
        "leftShoulder": 11,
        "rightShoulder": 12,
        "leftElbow": 13,
        "rightElbow": 14,
        "leftWrist": 15,
        "rightWrist": 16,
        "leftHip": 23,
        "rightHip": 24,
        "leftKnee": 25,
        "rightKnee": 26,
        "leftAnkle": 27,
        "rightAnkle": 28
    ]

    static func fuseSkeletons(
        apple: [String: SIMD3<Float>],
        mediaPipe: [Point3D],
        appleCalibrationScale: Float
    ) -> FusedSkeletonResult? {
        guard
            let aLeftHip = apple["leftHip"],
            let aRightHip = apple["rightHip"],
            let aLeftShoulder = apple["leftShoulder"],
            let aRightShoulder = apple["rightShoulder"],
            mediaPipe.count > 28
        else { return nil }

        let calibratedApple = apple.mapValues { point -> SIMD3<Float> in
            let pelvis = (aLeftHip + aRightHip) * 0.5
            return pelvis + (point - pelvis) * appleCalibrationScale
        }

        guard let appleBasis = bodyBasis(
            leftHip: calibratedApple["leftHip"]!,
            rightHip: calibratedApple["rightHip"]!,
            leftShoulder: calibratedApple["leftShoulder"]!,
            rightShoulder: calibratedApple["rightShoulder"]!
        ) else { return nil }

        let mp = mediaPipe.map { SIMD3<Float>($0.x, $0.y, $0.z) }
        guard let mpBasis = bodyBasis(
            leftHip: mp[23],
            rightHip: mp[24],
            leftShoulder: mp[11],
            rightShoulder: mp[12]
        ) else { return nil }

        let scale = robustScale(
            apple: calibratedApple,
            mediaPipe: mp
        )

        var mapped: [String: SIMD3<Float>] = [:]
        for (name, index) in commonMediaPipeIndices where index < mp.count {
            let delta = mp[index] - mpBasis.origin
            let local = SIMD3<Float>(
                simd_dot(delta, mpBasis.x),
                simd_dot(delta, mpBasis.y),
                simd_dot(delta, mpBasis.z)
            ) * scale

            mapped[name] = appleBasis.origin
                + appleBasis.x * local.x
                + appleBasis.y * local.y
                + appleBasis.z * local.z
        }

        let names = Set(calibratedApple.keys).union(mapped.keys)
        var fused: [String: SIMD3<Float>] = [:]
        var agreementDistances: [Float] = []

        for name in names {
            switch (calibratedApple[name], mapped[name]) {
            case let (a?, m?):
                let appleWeight: Float = {
                    if name.contains("Shoulder") || name.contains("Hip") { return 0.72 }
                    return 0.62
                }()
                fused[name] = a * appleWeight + m * (1 - appleWeight)
                agreementDistances.append(simd_distance(a, m))
            case let (a?, nil):
                fused[name] = a
            case let (nil, m?):
                fused[name] = m
            default:
                break
            }
        }

        let agreementCm: Double? = agreementDistances.isEmpty
            ? nil
            : Double(agreementDistances.reduce(0, +) / Float(agreementDistances.count) * 100)

        return FusedSkeletonResult(joints: fused, agreementCm: agreementCm)
    }

    static func metrics(
        fused: FusedSkeletonResult,
        visionBodyHeightCm: Double?
    ) -> TailoringMetrics {
        let j = fused.joints
        let shoulderWidth = distance(j["leftShoulder"], j["rightShoulder"]).map { Double($0 * 100) }

        let shoulderDrop: Double? = {
            guard let l = j["leftShoulder"], let r = j["rightShoulder"] else { return nil }
            return Double(abs(l.y - r.y) * 100)
        }()

        let leftArm: Double? = {
            guard
                let s = j["leftShoulder"],
                let e = j["leftElbow"],
                let w = j["leftWrist"]
            else { return nil }
            return Double((simd_distance(s, e) + simd_distance(e, w)) * 100)
        }()

        let rightArm: Double? = {
            guard
                let s = j["rightShoulder"],
                let e = j["rightElbow"],
                let w = j["rightWrist"]
            else { return nil }
            return Double((simd_distance(s, e) + simd_distance(e, w)) * 100)
        }()

        return TailoringMetrics(
            visionBodyHeightCm: visionBodyHeightCm,
            shoulderWidthCm: shoulderWidth,
            shoulderHeightDifferenceCm: shoulderDrop,
            leftArmLengthCm: leftArm,
            rightArmLengthCm: rightArm,
            skeletonAgreementCm: fused.agreementCm,
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

    static func analyzeCrossSections(
        canonicalPoints: [Point3D],
        torsoHeightMeters: Double,
        shoulderWidthMeters: Double,
        angularCoverage: Double
    ) -> SectionAnalysis {
        guard canonicalPoints.count > 500, torsoHeightMeters > 0.25 else {
            return SectionAnalysis(chest: nil, waist: nil, hip: nil)
        }

        let torso = Float(torsoHeightMeters)
        let shoulder = Float(max(0.30, shoulderWidthMeters))
        let halfBand: Float = 0.014

        let chest = bestSection(
            points: canonicalPoints,
            yFractions: strideValues(from: 0.54, through: 0.80, count: 12),
            torsoHeight: torso,
            halfBand: halfBand,
            maxAbsX: max(0.23, shoulder * 0.78),
            maxAbsZ: max(0.20, shoulder * 0.66),
            chooseMaximum: true,
            angularCoverage: angularCoverage
        )

        let waist = bestSection(
            points: canonicalPoints,
            yFractions: strideValues(from: 0.20, through: 0.50, count: 12),
            torsoHeight: torso,
            halfBand: halfBand,
            maxAbsX: max(0.22, shoulder * 0.74),
            maxAbsZ: max(0.19, shoulder * 0.62),
            chooseMaximum: false,
            angularCoverage: angularCoverage
        )

        let hip = bestSection(
            points: canonicalPoints,
            yFractions: strideValues(from: -0.24, through: 0.10, count: 13),
            torsoHeight: torso,
            halfBand: halfBand * 1.15,
            maxAbsX: max(0.25, shoulder * 0.90),
            maxAbsZ: max(0.22, shoulder * 0.72),
            chooseMaximum: true,
            angularCoverage: angularCoverage
        )

        return SectionAnalysis(chest: chest, waist: waist, hip: hip)
    }

    private struct Basis {
        var origin: SIMD3<Float>
        var x: SIMD3<Float>
        var y: SIMD3<Float>
        var z: SIMD3<Float>
    }

    private static func bodyBasis(
        leftHip: SIMD3<Float>,
        rightHip: SIMD3<Float>,
        leftShoulder: SIMD3<Float>,
        rightShoulder: SIMD3<Float>
    ) -> Basis? {
        let origin = (leftHip + rightHip) * 0.5
        let shoulderCenter = (leftShoulder + rightShoulder) * 0.5

        var x = rightShoulder - leftShoulder
        var y = shoulderCenter - origin
        if simd_length(x) < 0.05 || simd_length(y) < 0.10 { return nil }

        x = simd_normalize(x)
        y = simd_normalize(y)

        var z = simd_cross(x, y)
        if simd_length(z) < 0.05 { return nil }
        z = simd_normalize(z)
        x = simd_normalize(simd_cross(y, z))

        return Basis(origin: origin, x: x, y: y, z: z)
    }

    private static func robustScale(
        apple: [String: SIMD3<Float>],
        mediaPipe: [SIMD3<Float>]
    ) -> Float {
        let pairs: [(String, String, Int, Int)] = [
            ("leftShoulder", "rightShoulder", 11, 12),
            ("leftHip", "rightHip", 23, 24),
            ("leftShoulder", "leftElbow", 11, 13),
            ("rightShoulder", "rightElbow", 12, 14),
            ("leftElbow", "leftWrist", 13, 15),
            ("rightElbow", "rightWrist", 14, 16),
            ("leftHip", "leftKnee", 23, 25),
            ("rightHip", "rightKnee", 24, 26)
        ]

        var ratios: [Float] = []
        for (aName, bName, ai, bi) in pairs {
            guard
                let a = apple[aName],
                let b = apple[bName],
                ai < mediaPipe.count,
                bi < mediaPipe.count
            else { continue }

            let appleLength = simd_distance(a, b)
            let mpLength = simd_distance(mediaPipe[ai], mediaPipe[bi])
            if appleLength > 0.03, mpLength > 0.03 {
                ratios.append(appleLength / mpLength)
            }
        }

        guard !ratios.isEmpty else { return 1 }
        ratios.sort()
        return ratios[ratios.count / 2]
    }

    private struct P2: Comparable {
        var x: Float
        var z: Float

        static func < (lhs: P2, rhs: P2) -> Bool {
            lhs.x == rhs.x ? lhs.z < rhs.z : lhs.x < rhs.x
        }
    }

    private static func bestSection(
        points: [Point3D],
        yFractions: [Float],
        torsoHeight: Float,
        halfBand: Float,
        maxAbsX: Float,
        maxAbsZ: Float,
        chooseMaximum: Bool,
        angularCoverage: Double
    ) -> CrossSectionMeasurement? {
        var best: CrossSectionMeasurement?

        for fraction in yFractions {
            let y = torsoHeight * fraction
            guard let section = section(
                points: points,
                y: y,
                halfBand: halfBand,
                maxAbsX: maxAbsX,
                maxAbsZ: maxAbsZ,
                angularCoverage: angularCoverage
            ) else { continue }

            if let current = best {
                if chooseMaximum {
                    if section.perimeterCm > current.perimeterCm { best = section }
                } else {
                    if section.perimeterCm < current.perimeterCm { best = section }
                }
            } else {
                best = section
            }
        }

        return best
    }

    private static func section(
        points: [Point3D],
        y: Float,
        halfBand: Float,
        maxAbsX: Float,
        maxAbsZ: Float,
        angularCoverage: Double
    ) -> CrossSectionMeasurement? {
        var projected: [P2] = []
        projected.reserveCapacity(600)

        for point in points {
            guard abs(point.y - y) <= halfBand else { continue }
            guard abs(point.x) <= maxAbsX, abs(point.z) <= maxAbsZ else { continue }
            projected.append(P2(x: point.x, z: point.z))
        }

        guard projected.count >= 60 else { return nil }

        let xs = projected.map(\.x).sorted()
        let zs = projected.map(\.z).sorted()
        let medianX = xs[xs.count / 2]
        let medianZ = zs[zs.count / 2]

        let radii = projected.map { hypotf($0.x - medianX, $0.z - medianZ) }.sorted()
        let q1 = radii[radii.count / 4]
        let q3 = radii[(radii.count * 3) / 4]
        let upper = q3 + max(0.025, 2.8 * (q3 - q1))

        let filtered = projected.filter {
            hypotf($0.x - medianX, $0.z - medianZ) <= upper
        }
        guard filtered.count >= 50 else { return nil }

        let hull = convexHull(filtered)
        guard hull.count >= 8 else { return nil }

        var perimeter: Float = 0
        for i in hull.indices {
            let a = hull[i]
            let b = hull[(i + 1) % hull.count]
            perimeter += hypotf(a.x - b.x, a.z - b.z)
        }

        let minX = hull.map(\.x).min() ?? 0
        let maxX = hull.map(\.x).max() ?? 0
        let minZ = hull.map(\.z).min() ?? 0
        let maxZ = hull.map(\.z).max() ?? 0

        let samplingScore = min(1.0, Double(filtered.count) / 450.0)
        let hullScore = min(1.0, Double(hull.count) / 28.0)
        let coverageScore = min(1.0, angularCoverage / 0.80)
        let confidence = max(0, min(1, samplingScore * 0.35 + hullScore * 0.25 + coverageScore * 0.40))

        return CrossSectionMeasurement(
            yCmRelativePelvis: Double(y * 100),
            perimeterCm: Double(perimeter * 100),
            widthCm: Double((maxX - minX) * 100),
            depthCm: Double((maxZ - minZ) * 100),
            sampleCount: filtered.count,
            hullPointCount: hull.count,
            confidence: confidence
        )
    }

    private static func convexHull(_ input: [P2]) -> [P2] {
        let points = Array(Set(input.map { QuantizedP2($0) }))
            .map { $0.point }
            .sorted()

        guard points.count > 2 else { return points }

        func cross(_ o: P2, _ a: P2, _ b: P2) -> Float {
            (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x)
        }

        var lower: [P2] = []
        for p in points {
            while lower.count >= 2 &&
                    cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }

        var upper: [P2] = []
        for p in points.reversed() {
            while upper.count >= 2 &&
                    cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }

        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    private struct QuantizedP2: Hashable {
        let qx: Int
        let qz: Int
        let point: P2

        init(_ p: P2) {
            qx = Int((p.x * 1000).rounded())
            qz = Int((p.z * 1000).rounded())
            point = P2(x: Float(qx) / 1000, z: Float(qz) / 1000)
        }

        static func == (lhs: QuantizedP2, rhs: QuantizedP2) -> Bool {
            lhs.qx == rhs.qx && lhs.qz == rhs.qz
        }

        func hash(into hasher: inout Hasher) {
            hasher.combine(qx)
            hasher.combine(qz)
        }
    }

    private static func distance(
        _ a: SIMD3<Float>?,
        _ b: SIMD3<Float>?
    ) -> Float? {
        guard let a, let b else { return nil }
        return simd_distance(a, b)
    }

    private static func strideValues(
        from start: Float,
        through end: Float,
        count: Int
    ) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }
}
