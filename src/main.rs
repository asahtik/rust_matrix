mod helpers;
mod structs;

use structs::dense::matrix::*;

fn main() {
    let mut m = mat![3,3; data:[u32]=1,2,3,4,5,6,7,8,9];
    let m1 = m.get(0..2, 0..2);
    let m2 = m.get(1.., 1..);
    let m3 = m.get(.., ..);
    let m4 = m.get(0..=1, 0..=1);
    let m5 = m.get(..=1, ..=1);
    let m6 = m.get(..3, ..3);
    let m7 = m.get(1, 2);

    println!("{}", m);
    println!("{}", m1);
    println!("{}", m2);
    println!("{}", m3);
    println!("{}", m4);
    println!("{}", m5);
    println!("{}", m6);
    println!("{}", m7);
    println!("{}", m.scale(2));
    println!("{}", (m2 - m4).0.unwrap());
}