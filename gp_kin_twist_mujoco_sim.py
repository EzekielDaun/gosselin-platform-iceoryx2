import ctypes
import time
from pathlib import Path

import iceoryx2 as iox2
import jax.numpy as jnp
import mujoco
import mujoco.viewer as viewer
import numpy as np
import tomli_w
import tomllib
from gosselin_platform import GPR9, GPSE3SO23, GPDimension, mjcf_xml_string
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


class Twist(ctypes.Structure):
    _fields_ = [
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("vz", ctypes.c_float),
        ("wx", ctypes.c_float),
        ("wy", ctypes.c_float),
        ("wz", ctypes.c_float),
    ]

    @staticmethod
    def type_name() -> str:
        return "Twist"


if __name__ == "__main__":
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

    xml_string = mjcf_xml_string(DIMENSION, x, check_viewer=False)
    spec = mujoco.MjSpec.from_string(xml_string)  # type: ignore
    model = spec.compile()  # type: ignore
    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore
    twist_subscriber = (
        node.service_builder(iox2.ServiceName.new("/twist"))  # type: ignore
        .publish_subscribe(Twist)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    with viewer.launch_passive(model, data) as viewer:
        last_update_instant = time.perf_counter()
        while viewer.is_running():
            # twist control logic
            maybe_twist = None
            while True:
                temp = twist_subscriber.receive()
                if temp is None:
                    break
                else:
                    maybe_twist = temp
            if maybe_twist is None:
                pass
            else:
                current_instant = time.perf_counter()
                dt = current_instant - last_update_instant
                last_update_instant = current_instant

                twist: Twist = maybe_twist.payload().contents
                se3_log = (
                    np.array(
                        [
                            twist.vx,
                            twist.vy,
                            twist.vz,
                            twist.wx,
                            twist.wy,
                            twist.wz,
                        ],
                        dtype=np.float64,
                    )
                    * dt
                )

                (_, loss), x = DIMENSION.damped_newton_step_fn(
                    (x, 0.0), x.pose @ SE3.exp(se3_log), factor=1e-2
                )
                data.ctrl = DIMENSION.ik(x).rho
                print(DIMENSION.loss_func(x))
                if (
                    jnp.isnan(loss)
                    or jnp.any(jnp.isnan(jnp.array(data.ctrl)))
                    or jnp.linalg.norm(x.pose.translation() - data.qpos[:3]) > 0.1
                    or jnp.linalg.norm(x.pose.rotation().parameters() - data.qpos[3:7])
                    > 0.1
                ):
                    print("Resetting to initial position.")
                    mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
                    x = INIT_POS.x0
            mujoco.mj_step(model, data)  # type: ignore
            viewer.sync()
