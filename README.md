# Gosselin Platform on iceoryx2

Real‑time control stack for a 9‑actuator Gosselin platform using zero‑copy IPC with [iceoryx2](https://iceoryx.io/). The stack consists of:

- `orca-motor-server` (Rust): talks to 9 ORCA motors over serial, exposes connection control, consumes position commands, and publishes motor states.
- `gamepad-twist-publisher` (Rust): reads a gamepad and publishes either Twist or preset Pose commands for tele‑operation.
- `gp_kin_twist.py` / `gp_kin_pose.py` (Python): kinematics controllers using the `gosselin-platform` library (JAX) that convert Twist/Pose into actuator position commands.

All components communicate over iceoryx2 publish/subscribe and request/response services.

## Prerequisites

- Install [pixi](https://pixi.sh/latest/) (the only manual dependency). Pixi installs Rust, Python, and the Python packages (including `iceoryx2` and `gosselin-platform`).

## Install

```bash
pixi install --all
```

## Configure

- Serial devices: edit [orca-serial-config.toml](orca-motor-server/orca-serial-config.toml) to match your system. Paths are validated at runtime.
- Platform geometry: [dimension-capstone-revised.toml](dimension-capstone-revised.toml) defines the mechanism dimensions used by the kinematics.
- Initial position: [initial-position.toml](initial-position.toml) defines the home pose and nominal prismatic lengths used by the controllers.

## Run

`pixi run` to see available tasks.

### ORCA Motor Server

```bash
pixi run orca-motor-server
```

### Gamepad Twist/Pose Publisher

```bash
pixi run gamepad-twist-publisher
```

### Kinematics Controller (choose one)

Twist controller:

```bash
pixi run gp-kin-twist
```

Pose controller:

```bash
pixi run gp-kin-pose
```

### MuJoCo simulation with twist controller:

```bash
pixi run -e mujoco-sim gp-kin-twist-mujoco-sim
```

Leave the server and publisher running; start/stop the controller as needed.

### OAK-D RGB‑D Camera Publisher and Subscriber

```bash
pixi run -e oak-camera oak-rgbd-publisher
```

```bash
pixi run -e oak-camera oak-rgbd-subscriber
```

```bash
pixi run -e oak-camera oak-pcl-subscriber
```

## Controls

From `gamepad-twist-publisher/src/main.rs`:

- Twist (topic `/twist`):

  - `vx, vy` ← left stick X/Y
  - `vz` ← right stick Y
  - `wx` ← D‑pad Down − Up
  - `wy` ← D‑pad Right − Left
  - `wz` ← − right stick X

- Pose presets (topic `/pose`):
  - South (A/Cross): set z = 0.30 m
  - North (Y/Triangle): set z = 0.40 m
  - East (B/Circle): set x = +0.10 m, z = 0.35 m
  - West (X/Square): set x = −0.10 m, z = 0.35 m

The Python controllers:

- `gp_kin_twist.py` subscribes to `/twist` and integrates the commanded twist into an SE3 target.
- `gp_kin_pose.py` subscribes to `/pose` and tracks a target SE3 pose with rate limits.

Both compute inverse kinematics and publish `MotorCommandData` in micrometers to the motor server.

## IPC Services

The system uses these iceoryx2 services (see the Rust/Python sources for struct layouts):

- Publish/Subscribe
  - `/twist` → `Twist` (Rust struct; Python `ctypes.Structure`)
  - `/pose` → `Pose`
  - `orca-motor/position_command_um` → `MotorCommandData` (controller → server)
  - `orca-motor/state` → `MotorStates` (server → any listener)
- Request/Response
  - `orca-motor/connect` → `bool` request to (dis)connect motors; `bool` response indicates current connection state
- Event
  - `orca-motor/error_event` → notifies when any motor reports a non‑zero error code

## Notes and Tips

- Always run inside `pixi shell`; it provides Rust and Python toolchains and packages.
- First runs of the Python controllers perform JAX JIT warm‑up and may take a few seconds.
- On Linux/macOS, serial devices look like `/dev/tty*`; on Windows, use `COM*` in `orca-serial-config.toml`.
- Safety: verify a clear workspace before sending commands; the controllers move real hardware.
