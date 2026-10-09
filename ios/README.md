# Body Scan V0.2 — Native iOS

V0.2 moves the experiment from Safari into a native iPhone app so the scan pipeline can use iOS camera and AR data directly.

## What this first native build captures

- Apple Vision 3D human body pose (17 joints)
- MediaPipe Pose Landmarker world coordinates (33 joints)
- ARKit person segmentation and person depth when supported
- LiDAR scene depth when the iPhone supports it
- Camera intrinsics and AR world tracking for point unprojection
- A sampled 3D point cloud while the subject turns
- Angular coverage during the 360-degree turn
- Early tailoring metrics: shoulder width, left/right shoulder height difference, left/right arm length, Vision body-height estimate

The user enters their real height. V0.2 uses that value to calibrate Apple Vision skeleton distances before displaying tailoring metrics.

## Why both Apple and MediaPipe

Apple Vision supplies a native 3D skeleton relative to the iPhone camera. MediaPipe supplies a second independent set of 33 world landmarks. These are not yet fused into one optimized skeleton in this commit; both streams are captured so we can quantify agreement and later build a fusion/calibration layer.

## Depth strategy

The scanner prefers ARFrame.estimatedDepthData plus segmentationBuffer because those are person-specific when personSegmentationWithDepth is supported. On LiDAR devices, sceneDepth and smoothedSceneDepth are also enabled.

During scanning, the prototype samples depth pixels that belong to the person, unprojects them with camera intrinsics, transforms them into AR world coordinates, and accumulates a point cloud.

## Build

This repo uses XcodeGen so the Xcode project can be generated reproducibly.

On a Mac:

    brew install xcodegen
    cd ios
    xcodegen generate
    open BodyScan.xcodeproj

The project targets iOS 17+ and includes Google's MediaPipe Swift package. The MediaPipe pose model is downloaded from Google's official model storage the first time the app prepares MediaPipe.

## First validation goal

Do not judge UI yet. The first goal is to compare repeated scans against a professional tailor.

1. Scan one person three times without moving the phone.
2. Record shoulder width, shoulder height difference, left/right arm length and point-cloud consistency.
3. Have the tailor measure the same values manually.
4. Compare repeatability first, then absolute error.
5. Only after the skeleton/depth pipeline is stable do we derive chest/waist/hip cross-sections and a tailoring body mesh.

## Next engineering steps

- Fuse Apple 17-joint and MediaPipe 33-joint skeletons.
- Persist key RGB frames and depth frames, not only sampled points.
- Normalize the accumulated point cloud into a body-centered coordinate system.
- Fit a parametric body mesh to the point cloud and silhouettes.
- Cut chest/waist/hip cross-sections from that mesh and calculate real perimeter lengths.
- Feed the calibrated body model into digital fitting and pattern-maker rules.