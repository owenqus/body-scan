import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var scanner: BodyScanController

    var body: some View {
        ZStack {
            ARCameraView(scanner: scanner)
                .ignoresSafeArea()

            LinearGradient(
                colors: [.black.opacity(0.72), .clear, .black.opacity(0.78)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 12) {
                header
                Spacer()
                scanGuide
                diagnostics
                controls
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 14)
        }
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Body Scan V0.3")
                    .font(.headline)
                Text(scanner.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text("\(scanner.appleJointCount) Apple 3D joints")
                Text("\(scanner.mediaPipeJointCount) MediaPipe joints")
            }
            .font(.caption2.monospacedDigit())
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private var scanGuide: some View {
        VStack(spacing: 9) {
            ZStack {
                Circle()
                    .stroke(.white.opacity(0.20), lineWidth: 10)

                Circle()
                    .trim(from: 0, to: scanner.angularCoverage)
                    .stroke(
                        .white,
                        style: StrokeStyle(lineWidth: 10, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))

                VStack(spacing: 2) {
                    Text("\(Int(scanner.angularCoverage * 100))%")
                        .font(.title2.bold().monospacedDigit())
                    Text("角度覆盖")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 92, height: 92)

            Text(scanner.isScanning ? "保持原地，缓慢转满一圈" : "手机固定在支架上，开始后缓慢转一圈")
                .font(.subheadline.weight(.semibold))
                .multilineTextAlignment(.center)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private var diagnostics: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                diagnosticChip(
                    title: "人体深度",
                    value: scanner.personDepthAvailable ? "YES" : "NO"
                )
                diagnosticChip(
                    title: "LiDAR",
                    value: scanner.lidarSceneDepthAvailable ? "YES" : "NO"
                )
                diagnosticChip(
                    title: "点云",
                    value: "\(scanner.pointCloudCount)"
                )
            }

            HStack(spacing: 8) {
                metricChip("肩宽", scanner.latestMetrics.shoulderWidthCm)
                metricChip("肩差", scanner.latestMetrics.shoulderHeightDifferenceCm)
                metricChip("左臂", scanner.latestMetrics.leftArmLengthCm)
                metricChip("右臂", scanner.latestMetrics.rightArmLengthCm)
            }

            HStack(spacing: 8) {
                metricChip("胸围", scanner.latestMetrics.chestCircumferenceCm)
                metricChip("腰围", scanner.latestMetrics.waistCircumferenceCm)
                metricChip("臀围", scanner.latestMetrics.hipCircumferenceCm)
                metricChip("骨架差", scanner.latestMetrics.skeletonAgreementCm)
            }

            if let confidence = scanner.latestMetrics.sectionConfidence {
                Text("3D截面置信度 \(Int(confidence * 100))% · 需与版师实测校准")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Text(scanner.mediaPipeStatus)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    private var controls: some View {
        VStack(spacing: 10) {
            HStack {
                Text("真实身高")
                Spacer()
                TextField("cm", value: $scanner.inputHeightCm, format: .number.precision(.fractionLength(0...1)))
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                Text("cm")
                    .foregroundStyle(.secondary)
            }
            .font(.subheadline)
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))

            Button {
                if scanner.isScanning {
                    scanner.finishScan()
                } else {
                    scanner.beginScan()
                }
            } label: {
                Text(scanner.isScanning ? "结束扫描" : "开始 360° 扫描")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
            }
            .buttonStyle(.borderedProminent)
            .tint(scanner.isScanning ? .red : .white)
            .foregroundStyle(scanner.isScanning ? .white : .black)

            HStack {
                Button("重置") {
                    scanner.resetScan()
                }
                .buttonStyle(.bordered)

                Spacer()

                if let url = scanner.exportURL {
                    ShareLink(item: url) {
                        Label("导出原始扫描 JSON", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private func diagnosticChip(title: String, value: String) -> some View {
        VStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.bold().monospacedDigit())
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func metricChip(_ title: String, _ value: Double?) -> some View {
        VStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value.map { String(format: "%.1f", $0) } ?? "—")
                .font(.caption.bold().monospacedDigit())
            Text("cm")
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 7)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}
