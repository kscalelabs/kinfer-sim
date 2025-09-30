# kinfer-sim

Simulation backend for visualizing K-Infer models in real-time.

## Overview

`kinfer-sim` is a Python package that provides a simulation environment for running and visualizing exported K-Infer models. It enables developers to test and validate their trained policies in simulation before deploying them to physical K-Scale robots like K-Bot.

This tool is part of the K-Scale Labs ecosystem, which provides an integrated open-source stack for humanoid robotics—from hardware design to machine learning models.

## Features

- **Real-time visualization** of K-Infer model policies
- **Automatic MJCF downloading** from the K-Scale API
- **Command-line interface** for quick testing
- **Sim-to-real validation** before hardware deployment
- **Support for various robot models** including K-Bot

## Installation

Install via pip:

```bash
pip install kinfer-sim
```

## Quick Start

Run an exported K-Infer model with a single command:

```bash
kinfer-sim examples/kbot_walking.kinfer kbot-headless --use-keyboard
```

This will:
1. Load the specified `.kinfer` model file
2. Automatically download the robot's MJCF description from the K-Scale API
3. Launch the simulation viewer

## Usage

### Basic Command Structure

```bash
kinfer-sim <kinfer-path> <mujoco-model-name> [OPTIONS]
```

### Arguments

**Positional Arguments:**
- `kinfer-path`: Path to your exported K-Infer model (`.kinfer` file)
- `mujoco-model-name`: Name of the robot model (e.g., `kbot`, `zbot`)

**MuJoCo Settings:**
- `--mujoco-scene <str>`: MuJoCo scene to use (default: `smooth`)
- `--no-cache`: Don't use cached metadata
- `--debug`: Enable debug logging
- `--local-model-dir <path>`: Path to local robot directory containing `metadata.json` and `*.mjcf/*.xml` (bypasses K-Scale API)

**Physics Settings:**
- `--dt <float>`: Simulation timestep (default: `0.0001`)
- `--pd-update-frequency <float>`: PD update frequency for actuators in Hz (default: `1000.0`)
- `--no-gravity`: Disable gravity
- `--start-height <float>`: Initial height of the robot in meters (default: `1.1`)
- `--initial-quat <w,x,y,z>`: Initial quaternion as comma-separated values (e.g., `1.0,0.0,0.0,0.0`)
- `--suspend`: Suspend robot base in place to prevent falling
- `--quat-name <str>`: Name of the quaternion sensor (default: `imu_site_quat`)
- `--acc-name <str>`: Name of the accelerometer sensor (default: `imu_acc`)
- `--gyro-name <str>`: Name of the gyroscope sensor (default: `imu_gyro`)

**Rendering Settings:**
- `--no-render`: Disable rendering
- `--render-frequency <float>`: Render frequency in Hz (default: `1.0`)
- `--frame-width <int>`: Frame width in pixels (default: `640`)
- `--frame-height <int>`: Frame height in pixels (default: `480`)
- `--camera <str>`: Camera to use for rendering
- `--save-path <path>`: Path to save logs and videos (default: `logs`)
- `--save-video`: Save video of the simulation
- `--save-logs`: Save simulation logs
- `--free-camera`: Enable free camera control

**Model Settings:**
- `--use-keyboard`: Use keyboard to control the robot with a 16 dim command defined in `kinfer_sim/keyboard.py`

**Randomization Settings (for domain randomization testing):**
- `--command-delay-min <float>`: Minimum command delay in seconds
- `--command-delay-max <float>`: Maximum command delay in seconds
- `--drop-rate <float>`: Drop actions with this probability (default: `0.0`)
- `--joint-pos-delta-noise <float>`: Joint position delta noise in degrees (default: `0.0`)
- `--joint-pos-noise <float>`: Joint position noise in degrees (default: `0.0`)
- `--joint-vel-noise <float>`: Joint velocity noise in degrees/second (default: `0.0`)
- `--joint-zero-noise <float>`: Joint zero noise in degrees (default: `0.0`)
- `--accelerometer-noise <float>`: Accelerometer noise in m/s² (default: `0.0`)
- `--gyroscope-noise <float>`: Gyroscope noise in rad/s (default: `0.0`)
- `--projected-gravity-noise <float>`: Projected gravity noise in m/s² (default: `0.0`)

### Example Workflows

**Enable 16 dimensional keyboard control:**
```bash
kinfer-sim <kinfer-path> kbot-headless --use-keyboard --command-type unified
```

**Test with domain randomization:**
```bash
kinfer-sim <kinfer-path> kbot-headless \
  --drop-rate 0.1 \
  --joint-pos-noise 0.5 \
  --gyroscope-noise 0.01
```

**Use local robot model:**
```bash
kinfer-sim <kinfer-path> kbot-headless --local-model-dir <path to local model>
```

**Suspend robot for testing:**
```bash
kinfer-sim <kinfer-path> kbot-headless --suspend
```

## What is K-Infer?

K-Infer is K-Scale Labs' model export and inference tool. It converts trained reinforcement learning policies into an optimized format (`.kinfer` files) that can be:
- Visualized in simulation using `kinfer-sim`
- Deployed to physical robots using the K-Scale runtime
- Shared and evaluated in the K-Scale benchmarks

## Integration with K-Scale Ecosystem

`kinfer-sim` works seamlessly with other K-Scale tools:

- **K-Sim / IsaacLab / IsaacGym**: GPU-accelerated training environments for learning policies
- **K-Infer**: Model export and inference tool
- **kinfer-sim**: This package - for visualization and validation (you are here!)
- **Firmware**: For deploying policies to physical hardware

### Typical Workflow

1. Train a policy using K-Sim, IsaacLab, or IsaacGym
2. Export the policy using K-Infer to create a `.kinfer` file
3. Validate the policy using `kinfer-sim` (this tool)
4. Deploy to physical robot using firmware

## Documentation and Support

For more information about the K-Scale ecosystem:
- **Main Documentation**: [https://docs.kscale.dev](https://docs.kscale.dev)
- **Simulation Guide**: [https://docs.kscale.dev/robots/k-bot/simulation](https://docs.kscale.dev/robots/k-bot/simulation)
- **K-Scale Labs**: [https://kscale.dev](https://kscale.dev)
- **Discord**: Join our [Discord](https://discord.gg/wZmtKrRYwF) community for real-time help
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/kscalelabs/kinfer-sim/issues)

## Contributing

We welcome contributions! K-Scale Labs is building the world's most accessible platform for embodied intelligence, and we believe in open-source collaboration.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Check out our [GitHub organization](https://github.com/kscalelabs) for more projects to contribute to.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{kinfer_sim,
  title = {kinfer-sim: Simulation backend for K-Infer models},
  author = {K-Scale Labs},
  year = {2025},
  url = {https://github.com/kscalelabs/kinfer-sim}
}
```

---

Built with ❤️ by [K-Scale Labs](https://kscale.dev) - Making robots accessible to everyone.