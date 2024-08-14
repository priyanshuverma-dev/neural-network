use std::time::Instant;

mod net;
use net::Net;

fn main() {
    fn original_fn(x: f64) -> f64 {
        x * x * x + x * x + x
    }

    let training_data: Vec<(f64, f64)> = (1..=100)
        .step_by(7)
        .map(|x| (x as f64) / 100.0)
        .map(|x| (x, original_fn(x)))
        .collect();

    let validation_data: Vec<(f64, f64)> = (20..=60)
        .map(|x| (x as f64) / 100.0)
        .map(|x| (x, original_fn(x)))
        .collect();

    let start = Instant::now();
    let mut net = Net::new(20);
    net.train(&training_data, 100000, 100, 0.000001);
    eprintln!("Training duration: {}s", start.elapsed().as_secs());
    eprintln!("Validation error: {}", net.cost(&validation_data));

    for x in 0..1000 {
        let x = x as f64 / 1000.0;
        println!("{}\t{}\t{}", x, original_fn(x), net.eval(x));
    }
}
