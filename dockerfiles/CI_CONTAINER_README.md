# CI Container Strategy for Veris Memory

## Overview

This document describes the containerized CI/CD approach for the Veris Memory project, which significantly improves workflow performance and consistency.

## Benefits of Using Pre-built Containers

### 1. **Performance Improvements**
- **Before**: ~3-5 minutes to install dependencies on each PR
- **After**: ~10 seconds to pull pre-built container
- **Result**: 95% reduction in setup time

### 2. **Consistency**
- All CI runs use identical environment
- Eliminates "works on my machine" issues
- Predictable dependency versions

### 3. **Cost Savings**
- Reduced GitHub Actions minutes usage
- Less compute time = lower costs
- Faster feedback for developers

### 4. **Security**
- Weekly rebuilds include security patches
- Centralized dependency management
- Container signing and attestation support

## Container Details

### Image Location
```
ghcr.io/credentum/veris-memory-ci:latest
```

### What's Included
- Python 3.11 with all production dependencies
- All development/testing dependencies
- Node.js and npm for schema validation
- Pre-configured directory structure
- Git safe directory configuration

### Update Schedule
- **Automatic rebuilds**:
  - On changes to `requirements*.txt`
  - On changes to `Dockerfile.ci`
  - Weekly (Sundays at midnight UTC) for security updates
  
### Available Tags
- `latest`: Most recent build from main branch
- `main-<sha>`: Specific commit on main
- `pr-<number>`: PR-specific builds
- `YYYYMMDD`: Date-tagged builds

## Usage in Workflows

### Basic Usage
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/credentum/veris-memory-ci:latest
      credentials:
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/
```

### With Specific Version
```yaml
container:
  image: ghcr.io/credentum/veris-memory-ci:20250826
```

## Local Development

### Pull and Use Locally
```bash
# Pull the container
docker pull ghcr.io/credentum/veris-memory-ci:latest

# Run tests locally using the CI container
docker run --rm -v $(pwd):/app \
  ghcr.io/credentum/veris-memory-ci:latest \
  pytest tests/

# Interactive shell
docker run -it --rm -v $(pwd):/app \
  ghcr.io/credentum/veris-memory-ci:latest \
  bash
```

### Build Locally for Testing
```bash
# Build the container locally
docker build -f dockerfiles/Dockerfile.ci -t veris-memory-ci:local .

# Test with local build
docker run --rm -v $(pwd):/app veris-memory-ci:local pytest
```

## Maintenance

### Adding New Dependencies

1. Update `requirements.txt` or `requirements-dev.txt`
2. Push to main branch or create PR
3. Container will automatically rebuild
4. New container available within ~5 minutes

### Manual Rebuild

If you need to trigger a manual rebuild:

1. Go to Actions tab in GitHub
2. Select "Build CI Container" workflow
3. Click "Run workflow"
4. Select branch and run

### Troubleshooting

#### Container Pull Failures
```bash
# Ensure you're logged in to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Or use GitHub CLI
gh auth token | docker login ghcr.io -u USERNAME --password-stdin
```

#### Permission Issues
Ensure your repository has package write permissions:
- Settings → Actions → General → Workflow permissions
- Enable "Read and write permissions"

## Performance Metrics

### Before Containerization
- Dependency installation: ~180 seconds
- Directory creation: ~5 seconds
- Total setup time: ~185 seconds

### After Containerization
- Container pull: ~10 seconds
- No dependency installation needed
- No directory creation needed
- Total setup time: ~10 seconds

### Result
- **94.6% reduction in setup time**
- **Saves ~3 minutes per CI run**
- **With 100 PRs/month = 5 hours saved**

## Security Considerations

1. **Regular Updates**: Weekly rebuilds ensure security patches
2. **Vulnerability Scanning**: Can add Trivy or Snyk scanning
3. **Minimal Base Image**: Using python:3.11-slim reduces attack surface
4. **No Secrets**: No credentials or secrets in the container
5. **Signed Images**: Support for cosign signing (future enhancement)

## Future Enhancements

- [ ] Multi-architecture support (ARM64 for M1/M2 Macs)
- [ ] Container signing with cosign
- [ ] Vulnerability scanning in build pipeline
- [ ] Size optimization (current: ~500MB, target: <300MB)
- [ ] Development container variant with additional tools

## FAQ

### Q: How often should we rebuild the container?
A: Automatically weekly, and whenever dependencies change. Manual rebuilds as needed for urgent updates.

### Q: Can I use this container for local development?
A: Yes! It ensures your local environment matches CI exactly.

### Q: What if the container registry is down?
A: The workflow will fall back to installing dependencies directly (uncomment the old setup steps as fallback).

### Q: How do we handle different Python versions?
A: Create variant tags like `python3.10`, `python3.11`, `python3.12` if needed.

## Contact

For issues or questions about the CI container:
- Create an issue in the repository
- Tag with `ci/cd` and `containers` labels