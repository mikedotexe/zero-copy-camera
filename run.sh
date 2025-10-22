#!/bin/bash
# Quick run script for the Camera Kelp Demo

cd "$(dirname "$0")"

echo "Building Camera Kelp Demo..."
cargo build --release

if [ $? -eq 0 ]; then
    echo ""
    echo "Starting demo..."
    echo "Press ESC to quit"
    echo ""
    ./target/release/camera-kelp-demo
else
    echo "Build failed!"
    exit 1
fi
