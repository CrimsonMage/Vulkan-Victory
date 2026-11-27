//! tir_run <container> <memory file> [group] [wave]: steps one console wave
//! of a compute program's thread IR (`vk::interp`) over a memory image and
//! says how it ended, where it looped and which memory it read but the
//! image lacks.
//!
//! The memory file has one line per run of dwords, hexadecimal:
//! `<address> <dword> <dword> ...`; `push <k> <dword> ...` sets user data
//! from dword k, `range <start> <bytes>` the mapped range (default: all).
//! Environment: `WATCH=v1,v2` prints those values each time they change,
//! `LIMIT=n` block visits (default 2^22), `DUMP=1` prints the IR,
//! `IMAGE=size,levels,texel[,g,b,a]` (hexadecimal) what every image reads,
//! `UNDEF=bits` what undefined values read (default 0), `FILL=dword` what
//! memory the file leaves out reads (default 0), `OUT=1` prints the values
//! a pixel or vertex program outputs and what image stores write
//! (watched), `VARY=1` image texels that differ by lane.
use victory_shader::vk::{interp, thread_ir, tir};

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or_else(|_| panic!("not hexadecimal: {s}"))
}

/// The memory file: the memory image and the push dwords.
fn memory_file(path: &str) -> (interp::Memory, Vec<u32>) {
    let mut mem = interp::Memory { range: (0, u64::MAX), ..Default::default() };
    let mut push = vec![0u32; 64];
    for line in std::fs::read_to_string(path).expect("memory file").lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.first().copied() {
            None | Some("#") => {}
            Some("push") => {
                let k = w[1].parse::<usize>().unwrap();
                for (i, x) in w[2..].iter().enumerate() {
                    push[4 + k + i] = hex(x) as u32;
                }
            }
            Some("range") => mem.range = (hex(w[1]), hex(w[2])),
            Some(a) => {
                let words: Vec<u32> = w[1..].iter().map(|x| hex(x) as u32).collect();
                mem.put(hex(a), &words);
            }
        }
    }
    (mem, push)
}

/// The environment's settings (see the module comment).
fn configure(m: &mut interp::Machine, f: &tir::Func) {
    let var = |k: &str| std::env::var(k).ok();
    if let Some(l) = var("LIMIT") {
        m.limit = l.parse().unwrap();
    }
    if let Some(w) = var("WATCH") {
        m.watch = w.split(',').map(|x| x.trim_start_matches('%').parse().unwrap()).collect();
    }
    if let Some(x) = var("IMAGE") {
        let v: Vec<u64> = x.split(',').map(hex).collect();
        m.image_size = [v[0], v[0], 1, 1];
        m.image_levels = v[1] as u32;
        m.image_texel = if v.len() >= 6 { [v[2], v[3], v[4], v[5]] } else { [v[2]; 4] };
    }
    m.image_vary = var("VARY").is_some();
    if let Some(x) = var("FILL") {
        m.mem.fill = hex(&x) as u32;
    }
    if var("OUT").is_some() {
        for (v, n) in f.nodes.iter().enumerate() {
            if let tir::Op::Output { slot, comp } = n.op {
                println!("output slot {slot} comp {comp}: %{v} of {:?}", n.args);
                m.watch.extend(n.args.iter().copied());
            }
            if let tir::Op::Image { kind: tir::ImageKind::Store { .. }, .. } = n.op {
                println!("image store %{v} of {:?}", n.args);
                m.watch.extend(n.args.iter().copied());
            }
        }
    }
    if let Some(x) = var("UNDEF") {
        m.undef = hex(&x);
    }
}

fn report(m: &interp::Machine) {
    for l in &m.log {
        println!("  {l}");
    }
    let mut hot: Vec<(usize, u64)> = m.visits.iter().copied().enumerate().filter(|x| x.1 > 1).collect();
    hot.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("hot blocks: {:?}", &hot[..hot.len().min(16)]);
    println!("last blocks: {:?}", m.trail);
    for (v, n) in &m.notes {
        println!("note %{v}: {n}");
    }
    for (a, (k, v)) in &m.mem.missing {
        println!("missing {a:#x} read {k}x (first by %{v})");
    }
    for (a, w) in &m.mem.stored {
        println!("stored {a:#x} = {w:08x}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let container = std::fs::read(&args[1]).expect("container");
    let (f, _iface, native) = thread_ir(&container).expect("translation");
    if std::env::var_os("DUMP").is_some() {
        tir::dump(&f);
    }
    let (mem, push) = memory_file(&args[2]);
    let lanes = if native.wave64 { 64 } else { 32 };
    let mut m = interp::Machine::new(&f, lanes, mem);
    m.push = push;
    m.group = [args.get(3).map_or(0, |s| s.parse().unwrap()), 0, 0];
    m.wave = args.get(4).map_or(0, |s| s.parse().unwrap());
    configure(&mut m, &f);
    let stop = m.run();
    println!("stop: {stop:?} (wave{lanes}, {} blocks)", f.blocks.len());
    report(&m);
}
