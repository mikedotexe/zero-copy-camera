# Camera Kelp Demo

An audio-reactive camera effects demo using Metal GPU compute shaders on macOS.

## What to do

Have Rust: [https://rustup.rs](https://rustup.rs)

`cargo run --release`

## What It Does

This demo combines your webcam feed with artistic visual effects:

- **Warped Camera Sheets**: Two spinning camera sheets with various artistic warp patterns (sine waves, spirals, spherical bulge, checkerboard twist, hyperbolic flow)
- **Audio-Reactive Kelp**: Flame-like strands anchored to the screen edges that respond to music/audio in real-time
  - **Left side** = BASS responsive (slow, powerful waves, red/orange colors)
  - **Right side** = TREBLE responsive (fast shimmering, cyan/white colors, aggressive tip whipping)

## Controls

### Quit

- **Esc**

### Camera Modes (Keys 1-3)
- **1** - Centered & Flipped (mirror)
- **2** - Zoomed Out, No Flip
- **3** - Tight Crop, Flipped

### Warp Patterns (Keys 4-8)
- **4** - Sine Wave Ripples
- **5** - Spiral Vortex
- **6** - Spherical Bulge
- **7** - Checkerboard Twist
- **8** - Hyperbolic Flow

### Other Keys
- **G** - Grid Toggle
- **A** - Audio Toggle (turn audio reactivity on/off)
- **UP/DOWN** - Adjust Wind Strength
- **R** - Reset
- **Esc** - Quit

## Requirements

- macOS with Apple Silicon or Intel Mac with Metal support
- Rust toolchain (install from https://rustup.rs)
- Webcam
- Microphone (for audio reactivity)

## Building and Running

```bash
# Build and run in release mode (optimized)
cargo run --release

# Or build first, then run
cargo build --release
./target/release/camera-kelp-demo
```

## Technical Details

**Built with:**
- Rust for the main application loop
- Metal Shading Language (MSL) for GPU compute shaders
- nokhwa for camera capture
- cpal for audio input
- minifb for windowing

**Key Features:**
- Real-time GPU-accelerated rendering at 1280×720
- Live audio spectrum analysis (bass, mid, treble bands)
- Parametric kelp strand rendering with:
  - Quartic tip whipping (t⁴) for treble
  - Squared bass response (bass²) for powerful movement
  - Multi-frequency wave layering
  - Perpendicular distance calculation for sharp rendering
- Artistic camera warps with rotation and various distortion patterns

## How It Works

The demo runs entirely on the GPU using Metal compute shaders. Each frame:

1. Captures camera feed and crops to 1280×720
2. Analyzes audio input and separates into frequency bands
3. GPU shader renders:
   - Two warped/rotated camera sheets in 3D space
   - 25 kelp strands (12 on left responding to bass, 13 on right responding to treble)
   - Each strand uses parametric physics with audio-reactive displacement
   - Fire-like gradient coloring from anchor (red) to tip (yellow/white)
   - Bass creates slow rolling waves with red/orange pulses
   - Treble creates fast shimmers with cyan/white sparkles

The kelp strands are anchored directly to the screen edges (x=0.0 for left, x=1.0 for right) and grow inward approximately 20-37% across the screen, swaying and responding to audio in real-time.

---

**Created by Mike with Claude Code**
