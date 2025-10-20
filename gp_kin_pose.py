import asyncio
import ctypes
import time
from ctypes import c_bool, c_int32
from pathlib import Path

import iceoryx2 as iox2
import jax.numpy as jnp
import numpy as np
import tomli_w
import tomllib
from gosselin_platform import GPR9, GPSE3SO23, GPDimension
from jax.flatten_util import ravel_pytree
from jax_dataclasses import pytree_dataclass
from jaxlie import SE3, SO2


@pytree_dataclass(frozen=True)
class GPOrcaInitialPosition:
    """GPOrcaInitialPosition.

    Initial position for the robot, in SI units.

    Attributes:
        x0: **Home** end effector pose and redundant angle. Note that this is **not** the initial position of the platform when power off, but rather the pose of the end effector at the start of the trajectory. When the end effector is at this home pose and redundant angle, all actuators should keep a decent clearance from the mechanical limits on both side (i.e. centered, ideally).
        rho0: Real **effective** prismatic joint length in meters, from U joint to S joint, when the platform is **power off** and retracted to mechanical limits.

        Note that these two properties does **not** need to match.

    """

    x0: GPSE3SO23
    rho0: GPR9

    @property
    def ravel_1d_array(self):
        """Return a 1D array representation of the initial position."""
        return ravel_pytree(self)[0]

    def to_toml(self, path: Path) -> None:
        """Serialize the initial position and save to a TOML file."""
        arr = self.ravel_1d_array.tolist()
        with path.open("wb") as f:
            f.write(tomli_w.dumps({"initial_position": arr}).encode("utf-8"))

    @staticmethod
    def from_toml(path: Path) -> "GPOrcaInitialPosition":
        """Load GPOrcaInitialPosition from a TOML file."""
        with path.open("rb") as f:
            data = tomllib.load(f)
        arr = jnp.array(data["initial_position"])
        return GP_ORCA_INIT_POS_UNRAVEL_FN(arr)


GP_ORCA_INIT_POS_UNRAVEL_FN = ravel_pytree(
    GPOrcaInitialPosition(
        x0=GPSE3SO23(SE3.identity(), SO2.identity((3,))), rho0=GPR9(jnp.zeros(9))
    )
)[1]


class MotorCommandData(ctypes.Structure):
    _fields_ = [("position_um", c_int32 * 9)]

    @staticmethod
    def type_name() -> str:
        return "MotorCommandData"


class MotorState(ctypes.Structure):
    _fields_ = [
        ("position_um", c_int32),
        ("force_mn", c_int32),
        ("power_w", ctypes.c_uint16),
        ("temperature_c", ctypes.c_uint8),
        ("voltage_mv", ctypes.c_uint16),
        ("error", ctypes.c_uint16),
    ]

    @staticmethod
    def type_name() -> str:
        return "MotorState"


class MotorStates(ctypes.Structure):
    _fields_ = [("states", MotorState * 9)]

    @staticmethod
    def type_name() -> str:
        return "MotorStates"


class Pose(ctypes.Structure):
    _fields_ = [
        ("qw", ctypes.c_float),
        ("qx", ctypes.c_float),
        ("qy", ctypes.c_float),
        ("qz", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    @staticmethod
    def type_name() -> str:
        return "Pose"


def clip_by_norm(x, max_norm, axis=-1, eps=1e-12):
    x = jnp.asarray(x)
    norms = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    scale = jnp.minimum(1.0, max_norm / (norms + eps))
    return x * scale


async def main():
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore

    rr_connection_client = (
        node.service_builder(iox2.ServiceName.new("orca-motor/connect"))  # type: ignore
        .request_response(c_bool, c_bool)
        .open_or_create()
        .client_builder()
        .create()
    )

    # Connect to motor
    async with asyncio.timeout(5):
        pending_response = (
            rr_connection_client.loan_uninit().write_payload(c_bool(True)).send()
        )
        while True:
            maybe_connected = pending_response.receive()
            if maybe_connected is not None:
                print(f"Connected: {maybe_connected.payload().contents}")
                is_motor_connected = maybe_connected.payload().contents
                break
            else:
                # print("No response yet...")
                await asyncio.sleep(100e-3)
                # node.wait(iox2.Duration.from_millis(100))  # type: ignore

    if not is_motor_connected:
        raise RuntimeError("Failed to connect to motor.")

    # Load configuration
    DIMENSION = GPDimension.from_toml(Path("./dimension-capstone-revised.toml"))
    INIT_POS = GPOrcaInitialPosition.from_toml(Path("./initial-position.toml"))
    x = INIT_POS.x0
    # JIT warm up
    print("Warming up JIT...")
    for _ in range(100):
        (_, loss), x = DIMENSION.damped_newton_step_fn(
            (x, 0.0), INIT_POS.x0.pose, factor=1e-2
        )
    print("JIT warm up done.")

    try:
        motor_position_publisher = (
            node.service_builder(iox2.ServiceName.new("orca-motor/position_command_um"))  # type: ignore
            .publish_subscribe(MotorCommandData)
            .open_or_create()
            .publisher_builder()
            .create()
        )

        motor_state_subscriber = (
            node.service_builder(iox2.ServiceName.new("orca-motor/state"))  # type: ignore
            .publish_subscribe(MotorStates)
            .open_or_create()
            .subscriber_builder()
            .create()
        )

        pose_subscriber = (
            node.service_builder(iox2.ServiceName.new("/pose"))  # type: ignore
            .publish_subscribe(Pose)
            .open_or_create()
            .subscriber_builder()
            .create()
        )

        # Move to initial position
        actuator_commands_m_init = DIMENSION.ik(x).rho - INIT_POS.rho0.rho
        for frac in jnp.linspace(0, 1, int((_init_time := 5) / (init_dt := 25e-3))):
            motor_command_um_frac = actuator_commands_m_init * 1e6 * frac
            motor_position_publisher.loan_uninit().write_payload(
                MotorCommandData(
                    position_um=tuple(motor_command_um_frac.astype(np.int32).tolist())
                )
            ).send()
            await asyncio.sleep(init_dt)
            maybe_states = motor_state_subscriber.receive()
            if maybe_states is not None:
                states: MotorStates = maybe_states.payload().contents
                positions = [states.states[i].position_um for i in range(9)]
                print(f"Motor positions (um): {positions}")

        # TODO: pose control logic
        target_pose = x.pose
        last_update_instant = time.perf_counter()
        while True:
            maybe_pose = None
            while True:
                temp = pose_subscriber.receive()
                if temp is None:
                    break
                else:
                    maybe_pose = temp
            if maybe_pose is None:
                pass
            else:
                target_pose = SE3(
                    jnp.array(
                        [
                            maybe_pose.payload().contents.qw,
                            maybe_pose.payload().contents.qx,
                            maybe_pose.payload().contents.qy,
                            maybe_pose.payload().contents.qz,
                            maybe_pose.payload().contents.x,
                            maybe_pose.payload().contents.y,
                            maybe_pose.payload().contents.z,
                        ]
                    ),
                )

            current_instant = time.perf_counter()
            dt = current_instant - last_update_instant
            last_update_instant = current_instant

            se3_log = (x.pose.inverse() @ target_pose).log()
            twist = se3_log / dt
            twist_translation_clipped = clip_by_norm(twist[:3], 0.2)
            twist_rotation_clipped = clip_by_norm(twist[3:], jnp.deg2rad(30.0))
            twist_clipped = jnp.concatenate(
                [twist_translation_clipped, twist_rotation_clipped], axis=0
            )
            se3_log = twist_clipped * dt

            (_, loss), x = DIMENSION.damped_newton_step_fn(
                (x, 0.0), x.pose @ SE3.exp(se3_log), factor=1e-2
            )
            actuator_commands_m = DIMENSION.ik(x).rho - INIT_POS.rho0.rho
            motor_command_um = (actuator_commands_m * 1e6).astype(int).tolist()

            motor_position_publisher.loan_uninit().write_payload(
                MotorCommandData(position_um=tuple(motor_command_um))
            ).send()

            maybe_states = None
            while True:
                temp = motor_state_subscriber.receive()
                if temp is None:
                    break
                else:
                    maybe_states = temp
            if maybe_states is not None:
                states: MotorStates = maybe_states.payload().contents
                positions = [states.states[i].position_um for i in range(9)]
                print(f"Motor positions (um): {positions}")
    except Exception as e:
        raise e
    finally:
        # Close motor connection
        async with asyncio.timeout(5):
            pending_response = (
                rr_connection_client.loan_uninit().write_payload(c_bool(False)).send()
            )
            while True:
                maybe_connected = pending_response.receive()
                if maybe_connected is not None:
                    print(f"Connected: {maybe_connected.payload().contents}")
                    break
                else:
                    await asyncio.sleep(100e-3)


if __name__ == "__main__":
    asyncio.run(main())
