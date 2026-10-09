import ARKit
import SceneKit
import SwiftUI

struct ARCameraView: UIViewRepresentable {
    @ObservedObject var scanner: BodyScanController

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = scanner.session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = true
        view.preferredFramesPerSecond = 60
        scanner.startSession()
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    static func dismantleUIView(_ uiView: ARSCNView, coordinator: ()) {
        uiView.session.pause()
    }
}
