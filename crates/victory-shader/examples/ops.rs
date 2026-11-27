//! ops <container>... : the console instructions each program uses, by
//! name with their count (`ALL=1`: every instruction with its offset and
//! words instead), to see which opcodes a program that misbehaves has that
//! the proven ones lack.
use std::collections::BTreeMap;

fn main() {
    let all = std::env::var_os("ALL").is_some();
    for path in std::env::args().skip(1) {
        let container = std::fs::read(&path).expect("container");
        let Ok((_, _, native)) = victory_shader::vk::thread_ir(&container) else {
            println!("{path}: not translated");
            continue;
        };
        let Ok(code) = victory_shader::gfx10::decode(&native.code) else {
            println!("{path}: not decoded");
            continue;
        };
        let mut count: BTreeMap<String, u32> = BTreeMap::new();
        for i in &code {
            let name = i.name().map_or_else(|| format!("{:?}:{:#x}", i.encoding, i.opcode), str::to_string);
            if all {
                println!("{:6x} {name} {}", i.offset, i.words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" "));
            }
            *count.entry(name).or_default() += 1;
        }
        let list: Vec<String> = count.iter().map(|(n, c)| format!("{n} {c}")).collect();
        println!("{path}: wave{} {} instructions: {}", if native.wave64 { 64 } else { 32 }, code.len(), list.join(", "));
    }
}
