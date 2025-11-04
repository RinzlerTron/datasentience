#!/bin/bash
# Convert OCI manifest to Docker v2 format for SageMaker compatibility

set -e

AWS_REGION="us-east-1"
REPO="datasentience"
SOURCE_TAG="indexer-fix-v2"
TARGET_TAG="indexer-fix-v2-docker-v2"

echo "Converting OCI manifest to Docker v2 format..."
echo "Source tag: $SOURCE_TAG"
echo "Target tag: $TARGET_TAG"

# Fetch OCI manifest
echo "Fetching OCI manifest..."
aws ecr batch-get-image \
    --repository-name "$REPO" \
    --image-ids imageTag="$SOURCE_TAG" \
    --region "$AWS_REGION" \
    --query 'images[0].imageManifest' \
    --output text > manifest_oci_temp.json

# Convert OCI to Docker v2 format
echo "Converting manifest format..."
sed 's/application\/vnd\.oci\.image\.manifest\.v1+json/application\/vnd.docker.distribution.manifest.v2+json/g' manifest_oci_temp.json | \
sed 's/application\/vnd\.oci\.image\.config\.v1+json/application\/vnd.docker.container.image.v1+json/g' | \
sed 's/application\/vnd\.oci\.image\.layer\.v1\.tar+gzip/application\/vnd.docker.image.rootfs.diff.tar.gzip/g' | \
sed 's/application\/vnd\.oci\.image\.index\.v1+json/application\/vnd.docker.distribution.manifest.v2+json/g' > manifest_docker_v2_temp.json

# Put converted manifest
echo "Uploading Docker v2 manifest..."
aws ecr put-image \
    --repository-name "$REPO" \
    --image-manifest file://manifest_docker_v2_temp.json \
    --image-tag "$TARGET_TAG" \
    --region "$AWS_REGION"

# Verify manifest type
echo "Verifying manifest type..."
MANIFEST_TYPE=$(aws ecr describe-images \
    --repository-name "$REPO" \
    --image-ids imageTag="$TARGET_TAG" \
    --region "$AWS_REGION" \
    --query 'imageDetails[0].imageManifestMediaType' \
    --output text)

echo "Manifest type: $MANIFEST_TYPE"

if [[ "$MANIFEST_TYPE" == "application/vnd.docker.distribution.manifest.v2+json" ]]; then
    echo "✅ Conversion successful! Docker v2 format confirmed."
    echo "✅ Ready to deploy with tag: $TARGET_TAG"
else
    echo "❌ Warning: Manifest type is $MANIFEST_TYPE (expected Docker v2)"
    exit 1
fi

# Cleanup
rm -f manifest_oci_temp.json manifest_docker_v2_temp.json

echo "✅ Conversion complete. Use tag: $TARGET_TAG"

