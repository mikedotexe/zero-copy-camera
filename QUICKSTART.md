# Quick Start Guide

Hey Paul! Here's how to get this demo running in 60 seconds.

## Prerequisites

You need Rust installed. If you don't have it:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Run the Demo

```bash
./run.sh
```

That's it! The script will build and launch the demo.

## Controls Quick Reference

**Try these first:**
- Keys **1-3**: Switch camera modes
- Keys **4-8**: Change warp patterns
- **A**: Toggle audio reactivity
- **ESC**: Quit

**Left side kelp** = responds to BASS (red/orange, slow waves)
**Right side kelp** = responds to TREBLE (cyan/white, fast shimmers)

## What You're Seeing

- Two spinning camera sheets with artistic warps
- 25 kelp-like strands anchored to screen edges
- Everything responds to your microphone input in real-time
- Pure GPU rendering using Metal compute shaders

## Share This Project

To create a portable archive:
```bash
./create-archive.sh
```

This creates a timestamped .tar.gz file you can share with others.

## Dive Deeper

Check out `README.md` for full documentation, technical details, and all keyboard controls.

The code is in:
- `src/main.rs` - Rust application (camera capture, audio analysis, main loop)
- `shaders/camera_effects.metal` - GPU shader (all the visual magic)

---

Enjoy! - Mike
