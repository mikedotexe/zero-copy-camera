#!/bin/bash
# Package the Camera Kelp Demo project for distribution

cd "$(dirname "$0")"

PROJECT_NAME="camera-kelp-demo"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE_NAME="${PROJECT_NAME}-${TIMESTAMP}.tar.gz"

echo "Packaging Camera Kelp Demo..."
echo ""

# Create archive excluding build artifacts and git files
tar -czf "../${ARCHIVE_NAME}" \
    --exclude='target' \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    --exclude='Cargo.lock' \
    -C .. \
    "$(basename "$PWD")"

if [ $? -eq 0 ]; then
    ARCHIVE_SIZE=$(ls -lh "../${ARCHIVE_NAME}" | awk '{print $5}')
    echo "✓ Package created successfully!"
    echo ""
    echo "Archive: ../${ARCHIVE_NAME}"
    echo "Size: ${ARCHIVE_SIZE}"
    echo ""
    echo "To share with Paul:"
    echo "  1. Send him the archive file"
    echo "  2. He can extract it with: tar -xzf ${ARCHIVE_NAME}"
    echo "  3. Then run: cd ${PROJECT_NAME} && ./run.sh"
    echo ""
    echo "Full path: $(cd .. && pwd)/${ARCHIVE_NAME}"
else
    echo "✗ Package creation failed!"
    exit 1
fi
