//! Diagnostics: what each command buffer records, kept with it (the last
//! `Trail::MOST` notes) until it is submitted; a submission that does not
//! finish is reported with its notes (`engine.rs`, `report_stalls`).

use std::collections::HashMap;
use std::sync::Mutex;

use super::super::{Backend, read_title};
use super::Rec;

/// The culling programs that loop on their batch records (guard ids the
/// translator named: CS_Grass/Simple/Slod_BatchCull).
const CULLERS: [u32; 3] = [0x5819_b88e, 0x3125_4371, 0x05a6_cad5];
/// Dispatches logged per pipeline.
const CULLS_LOGGED: u32 = 4;

/// A buffer V# as base, stride and records (`docs/vulkan-abi.md` 4).
fn vsharp(w: &[u32]) -> String {
    let base = u64::from(w[0]) | u64::from(w[1] & 0xffff) << 32;
    format!("base={base:#x} stride={} records={}", (w[1] >> 16) & 0x3fff, w[2])
}

/// The V# in title memory at `at`, if the CPU reads it.
fn vsharp_at(b: &Backend, at: u64) -> String {
    match read_title(b, at, 16) {
        Ok(bytes) => {
            let w: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            format!("{} words={:08x?}", vsharp(&w), w)
        }
        Err(_) => "unreadable".into(),
    }
}

/// The dwords at `at`, as the CPU sees them (the home of pool pages, not
/// what the card wrote).
fn words_at(b: &Backend, at: u64, n: u64) -> String {
    read_title(b, at, 4 * n).map_or_else(
        |_| "unreadable".into(),
        |bytes| format!("{:?}", bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()),
    )
}

impl Rec {
    /// Notes `what` in the command buffer's trail, with the pipeline's
    /// shader names when they change.
    pub(super) fn note(&mut self, what: impl FnOnce() -> String) {
        self.note_as(what, false);
    }

    /// [`Rec::note`], with the shader names always (`named`): a dispatch,
    /// so that each line's last dispatches name their programs.
    pub(super) fn note_as(&mut self, what: impl FnOnce() -> String, named: bool) {
        if !cfg!(feature = "diagnostics") {
            return;
        }
        let pipeline = self.pipeline;
        let Some(r) = self.cmd.as_mut() else { return };
        let first = r.trail.notes.is_empty() && r.trail.earlier == 0;
        let same = std::mem::replace(&mut self.noted, pipeline) == pipeline;
        let note = if same && !first && !named {
            what()
        } else {
            let names = crate::pipeline::parts(pipeline)
                .map(|(s, _)| crate::pipeline::names(s))
                .unwrap_or_default();
            format!("{} [{names}]", what())
        };
        r.trail.push_with(note, self.last_tables.clone());
    }

    /// Diagnostics: the first [`CULLS_LOGGED`] dispatches of each pipeline
    /// of a culling program: its group counts (`groups`; `None` for an
    /// indirect one, whose args' title address and count address are
    /// `indirect`, read as the CPU sees them), and user data 5-8 read as an
    /// inline V#, as a table address and as a 64-bit address of one.
    pub(super) fn watch_cull(&mut self, b: &Backend, groups: Option<[u32; 3]>, indirect: Option<(u64, Option<u64>)>) {
        static SEEN: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
        if !cfg!(feature = "diagnostics") || !CULLERS.contains(&self.guard_id) {
            return;
        }
        {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let n = seen.get_or_insert_with(HashMap::new).entry(self.pipeline as usize).or_insert(0);
            *n += 1;
            if *n > CULLS_LOGGED {
                return;
            }
        }
        let ud = [self.last_push[9], self.last_push[10], self.last_push[11], self.last_push[12]];
        let args = indirect.map(|(src, count)| {
            format!(
                "args at {src:#x} = {} count at {:?} = {}",
                words_at(b, src, 3),
                count.map(|c| format!("{c:#x}")),
                count.map_or_else(|| "none".into(), |c| words_at(b, c, 1))
            )
        });
        let names = crate::pipeline::parts(self.pipeline).map(|(s, _)| crate::pipeline::names(s)).unwrap_or_default();
        tracing::info!(
            program = %names,
            guard_id = format_args!("{:#010x}", self.guard_id),
            groups = ?groups,
            indirect = args.as_deref().unwrap_or("no"),
            user_data_5_8 = format_args!("{:08x} {:08x} {:08x} {:08x}", ud[0], ud[1], ud[2], ud[3]),
            inline_vsharp = %vsharp(&ud),
            at_table_address = %vsharp_at(b, u64::from(ud[0])),
            at_64bit_address = %vsharp_at(b, u64::from(ud[0]) | u64::from(ud[1]) << 32),
            user_data_map = %self.last_user,
            "Vulkan: a culling dispatch (diagnostics)"
        );
        // What the card reads there (the translator's reading of the
        // batch culling programs): user data 0-8; the V#s at user-data
        // dword 4 (a table address) +0xac0, +0x3a0, +0xb20, +0x2e0; the
        // records at the V# at user-data dword 5 (s88 = records[group]);
        // and for the first groups, 0x120 bytes at the +0xac0 V#'s base
        // plus s88.
        let pushed = self.last_push;
        tracing::info!(
            program = %names,
            user_data_0_8 = format_args!("{:08x?}", &pushed[4..13]),
            "Vulkan: a culling dispatch's user data (diagnostics)"
        );
        let t4 = u64::from(pushed[8]);
        let t5 = u64::from(pushed[9]);
        let base_at = |at: u64| -> Option<u64> {
            let v = read_title(b, at, 8).ok()?;
            let base = u64::from(u32::from_le_bytes([v[0], v[1], v[2], v[3]])) | u64::from(u16::from_le_bytes([v[4], v[5]])) << 32;
            (base != 0).then_some(base)
        };
        let mut targets: Vec<super::super::probe::Target> = [0xac0u64, 0x3a0, 0xb20, 0x2e0]
            .iter()
            .map(|off| (format!("ud4+{off:#x} V#"), t4 + off, 16))
            .collect();
        targets.push(("ud5 V# (records)".into(), t5, 16));
        let records = base_at(t5);
        if let Some(r) = records {
            targets.push(("records[0..16]".into(), r, 64));
        }
        if let (Some(r), Some(batch)) = (records, base_at(t4 + 0xac0)) {
            for g in [0u64, 1, 2] {
                let Ok(w) = read_title(b, r + 4 * g, 4) else { continue };
                let s88 = u64::from(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                targets.push((format!("batch group {g} (s88={s88:#x})"), batch + s88, 0x120));
            }
        }
        super::super::probe::remember(&targets);
        if let Some(cmd) = self.cmd.as_ref().map(|r| r.cmd) {
            super::super::probe::record(b, cmd, targets);
        }
    }
}
