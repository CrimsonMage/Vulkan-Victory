//! Cheap hashing and timing for the paths every draw takes.
//!
//! The title records some 2,200 draws a frame on five or six threads, and
//! each draw passes through a dozen maps and sets keyed by addresses and
//! pointers. Std's maps hash with SipHash, made to resist keys chosen by
//! an attacker; ours are not, and SipHash was a few percent of the
//! recording threads in the profile of 2026-11-25. `Instant::now` under
//! Proton is a call into the Unix side; two per recording call were
//! another few percent.

use std::hash::{BuildHasherDefault, Hasher};

/// A multiplicative hasher in the manner of rustc's FxHash: each word is
/// added and the sum multiplied by an odd constant; the finish rotates the
/// well-mixed high bits down to where the table takes its bucket from.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

const K: u64 = 0xf135_7aea_2e62_a9c5;

impl Fx {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = self.0.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last) ^ (rest.len() as u64) << 59);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

pub(crate) type FastBuild = BuildHasherDefault<Fx>;
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, FastBuild>;
pub(crate) type FastSet<K> = std::collections::HashSet<K, FastBuild>;

/// A tick count for summing short spans: the CPU's time-stamp counter
/// (tens of cycles to read, and the same rate on every core of the CPUs
/// this runs on), else nanoseconds. [`ticks_to_ns`] converts a sum.
#[inline]
pub(crate) fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: RDTSC is in every x86-64 CPU and allowed in user mode.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        epoch().0.elapsed().as_nanos() as u64
    }
}

/// The instant and tick count of the first use, for [`ticks_to_ns`].
fn epoch() -> &'static (std::time::Instant, u64) {
    static EPOCH: std::sync::OnceLock<(std::time::Instant, u64)> = std::sync::OnceLock::new();
    EPOCH.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            (std::time::Instant::now(), ticks())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            (std::time::Instant::now(), 0)
        }
    })
}

/// Called once early (the first list), so that the rate below is measured
/// over the whole run.
pub(crate) fn start_clock() {
    let _ = epoch();
}

/// Nanoseconds of `ticks` [`ticks`], at the rate measured from the first
/// use until now (for reports, not for every span: it reads the clock).
pub(crate) fn ticks_to_ns(ticks: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let &(since, at) = epoch();
        let ns = since.elapsed().as_nanos() as f64;
        let spent = self::ticks().wrapping_sub(at) as f64;
        if spent < 1.0 || ns < 1.0 {
            return 0;
        }
        (ticks as f64 * ns / spent) as u64
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        ticks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keys a pointer apart land in different buckets of a small table,
    /// and byte strings that differ only in length hash differently.
    #[test]
    fn aligned_pointers_spread_and_lengths_count() {
        let mut low = std::collections::HashSet::new();
        for k in 0..64u64 {
            let mut h = Fx::default();
            h.write_u64(0x7ff0_1234_0000 + k * 64);
            low.insert(h.finish() & 63);
        }
        assert!(low.len() > 32, "{} buckets of 64", low.len());
        let hash = |b: &[u8]| {
            let mut h = Fx::default();
            h.write(b);
            h.finish()
        };
        assert_ne!(hash(&[0]), hash(&[0, 0]));
        let mut map: FastMap<(u64, usize), u32> = FastMap::default();
        map.insert((1, 2), 3);
        assert_eq!(map.get(&(1, 2)), Some(&3));
    }

    /// Ticks advance and convert to about the time that passed.
    #[test]
    fn ticks_convert_to_nanoseconds() {
        start_clock();
        let t = ticks();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let ns = ticks_to_ns(ticks().wrapping_sub(t));
        assert!((15_000_000..2_000_000_000).contains(&ns), "{ns} ns");
    }
}
