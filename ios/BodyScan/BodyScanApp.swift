import SwiftUI

@main
struct BodyScanApp: App {
    @StateObject private var scanner = BodyScanController()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(scanner)
        }
    }
}
