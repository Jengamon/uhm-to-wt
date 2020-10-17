//! Stuff for writing .wt files
use crate::uhm::Frame;
use std::io::Write;
use byteorder::{WriteBytesExt, LittleEndian};

const FRAME_LENGTH: u32 = 2048;

fn clamp_sample(v: f32) -> f32 {
    v.min(1.0).max(-1.0)
}

pub fn write_wt<W: Write>(writer: &mut W, frames: &[Frame], integer_mode: bool) -> std::io::Result<()> {
    writer.write_all(b"vawt")?;
    writer.write_u32::<LittleEndian>(FRAME_LENGTH)?; // All .uhm files work with frames of size 2048
    writer.write_u16::<LittleEndian>(frames.len() as u16)?;

    let flags: u16 = (if integer_mode { 4 } else { 0 }) | 8; // (INTEGER_MODE)? | (MODERN_INTEGER_VALUES)

    writer.write_u16::<LittleEndian>(flags)?;

    for frame in frames {
        for sample in frame {
            if integer_mode {
                writer.write_i16::<LittleEndian>((clamp_sample(*sample) * (i16::MAX as f32)).round() as i16)?;
            } else {
                writer.write_f32::<LittleEndian>(clamp_sample(*sample))?;
            }
        }
    }

    Ok(())
}