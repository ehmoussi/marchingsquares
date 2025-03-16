#[inline(always)]
pub fn close(p1_x: f64, p1_y: f64, p2_x: f64, p2_y: f64, tol: f64) -> bool {
    (p1_x - p2_x).abs() < tol && (p1_y - p2_y).abs() < tol
}

fn _print_segment(i: usize, segments: &Vec<f64>) {
    print!(
        "[({p1_x}, {p1_y})",
        p1_x = segments[4 * i],
        p1_y = segments[4 * i + 1]
    );
    print!(
        ", ({p2_x}, {p2_y})]",
        p2_x = segments[4 * i + 2],
        p2_y = segments[4 * i + 3],
    );
}

fn _print_segment_by_position(
    r0: &usize,
    c0: &usize,
    nb_cols: &usize,
    positions: &Vec<usize>,
    segments: &Vec<f64>,
    neighbors: &Vec<usize>,
) -> () {
    println!("--------------------------------");
    let index = r0 * (nb_cols - 1) + c0;
    if let (Some(&start), Some(&end)) = (positions.get(index), positions.get(index + 1)) {
        for i in start..end {
            _print_segment(i, &segments);
            println!("");
            for &neighbor_index in neighbors.iter() {
                if let (Some(&start_nb), Some(&end_nb)) = (
                    positions.get(neighbor_index),
                    positions.get(neighbor_index + 1),
                ) {
                    for j in start_nb..end_nb {
                        print!("    --> ");
                        _print_segment(j, segments);
                        println!();
                    }
                }
            }
        }
    }
}
