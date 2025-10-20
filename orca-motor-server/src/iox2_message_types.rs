use iceoryx2::prelude::*;

#[derive(Debug, Copy, Clone, ZeroCopySend, Default)]
#[repr(C)]
pub struct MotorState {
    pub position_um: i32,
    pub force_mn: i32,
    pub power_w: u16,
    pub temperature_c: u8,
    pub voltage_mv: u16,
    pub error: u16,
}

#[derive(Debug, Copy, Clone, ZeroCopySend, Default)]
#[type_name("MotorStates")]
#[repr(C)]
pub struct MotorStates {
    pub states: [MotorState; 9],
}

#[derive(Debug, Clone, Copy, ZeroCopySend)]
#[type_name("MotorCommandData")]
#[repr(C)]
pub struct MotorCommandData {
    pub position_um: [i32; 9],
}
