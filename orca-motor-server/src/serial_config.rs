use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{collections::HashSet, fs, path::Path};

#[derive(Debug, Deserialize)]
pub struct OrcaSerialConfig {
    pub baudrate: u32,
    pub motor: [String; 9],
}

impl OrcaSerialConfig {
    pub fn validate(self, check_paths: bool) -> Result<Self> {
        // check device path uniqueness
        let mut dseen = HashSet::new();
        for s in &self.motor {
            if !dseen.insert(s) {
                bail!("duplicated device path: {}", s);
            }
        }

        // optional: check device path existence
        if check_paths {
            for s in &self.motor {
                let p = Path::new(s);
                if !p.exists() {
                    bail!("device not found: {}", s);
                }
            }
        }
        Ok(self)
    }
}

pub fn load_config(path: impl AsRef<Path>, check_paths: bool) -> Result<OrcaSerialConfig> {
    let txt = fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.as_ref().display()))?;
    let cfg: OrcaSerialConfig =
        toml::from_str(&txt).with_context(|| format!("parsing {}", path.as_ref().display()))?;
    cfg.validate(check_paths)
}
