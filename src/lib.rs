use numpy::ndarray::{Array, ArrayD, ArrayViewD};
use numpy::{IntoPyArray, IxDyn, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

#[inline]
fn get_fraction(from_value: f64, to_value: f64, level: f64) -> f64 {
    if to_value == from_value {
        return 0.0;
    }
    return (level - from_value) / (to_value - from_value);
}

#[pyfunction]
#[pyo3(signature=(p1_x, p1_y, p2_x, p2_y, tol=1e-10))]
#[inline]
fn close(p1_x: f64, p1_y: f64, p2_x: f64, p2_y: f64, tol: f64) -> bool {
    (p1_x - p2_x).abs() < tol && (p1_y - p2_y).abs() < tol
}

#[inline]
fn top_x(r: usize, _: usize, _: f64, _: f64, _: f64) -> f64 {
    r as f64
}

#[inline]
fn top_y(_: usize, c: usize, ul: f64, ur: f64, level: f64) -> f64 {
    c as f64 + get_fraction(ul, ur, level)
}

#[inline]
fn bottom_x(r: usize, _: usize, _: f64, _: f64, _: f64) -> f64 {
    r as f64
}

#[inline]
fn bottom_y(_: usize, c: usize, ll: f64, lr: f64, level: f64) -> f64 {
    c as f64 + get_fraction(ll, lr, level)
}

#[inline]
fn left_x(r: usize, _: usize, ul: f64, ll: f64, level: f64) -> f64 {
    r as f64 + get_fraction(ul, ll, level)
}

#[inline]
fn left_y(_: usize, c: usize, _: f64, _: f64, _: f64) -> f64 {
    c as f64
}

#[inline]
fn right_x(r: usize, _: usize, ur: f64, lr: f64, level: f64) -> f64 {
    r as f64 + get_fraction(ur, lr, level)
}

#[inline]
fn right_y(_: usize, c: usize, _: f64, _: f64, _: f64) -> f64 {
    c as f64
}

fn _get_contour_segments(
    array: &ArrayViewD<'_, f64>,
    level: f64,
    vertex_connect_high: bool,
    mask: &ArrayViewD<'_, u8>,
) -> (Vec<f64>, Vec<u8>) {
    let shape = array.shape();
    let nb_rows = shape[0];
    let nb_cols = shape[1];
    let mut segments = Vec::with_capacity(nb_rows * nb_cols);
    let mut square_cases: Vec<u8> = Vec::with_capacity((nb_rows - 1) * (nb_cols - 1));
    for r0 in 0..(nb_rows - 1) {
        let r1 = r0 + 1;
        for c0 in 0..(nb_cols - 1) {
            let c1 = c0 + 1;
            let (ul, ll, ur, lr): (f64, f64, f64, f64);
            unsafe {
                let is_masked = mask.uget([r0, c0])
                    * mask.uget([r1, c0])
                    * mask.uget([r0, c1])
                    * mask.uget([r1, c1]);
                if is_masked == 0 {
                    square_cases.push(0);
                    continue;
                }
                ul = *array.uget([r0, c0]);
                ll = *array.uget([r1, c0]);
                ur = *array.uget([r0, c1]);
                lr = *array.uget([r1, c1]);
            }
            let square_case = 1 * u8::from(ul > level) as u8
                | 2 * (ur > level) as u8
                | 4 * (ll > level) as u8
                | 8 * (lr > level) as u8;
            square_cases.push(square_case);
            let square_segments = match square_case {
                1 => vec![
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ],
                2 => vec![
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ],
                3 => vec![
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ],
                4 => vec![
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ],
                5 => vec![
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ],
                6 => match vertex_connect_high {
                    true => vec![
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        // seg 2
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                    ],
                    false => vec![
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        // seg 2
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                    ],
                },
                7 => vec![
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ],
                8 => vec![
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ],
                9 => match vertex_connect_high {
                    true => vec![
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        // seg 2
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                    ],
                    false => vec![
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        // seg 2
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                    ],
                },
                10 => vec![
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ],
                11 => vec![
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ],
                12 => vec![
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ],
                13 => vec![
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ],
                14 => vec![
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ],
                0 | 15 => vec![], // No segments pass through the square
                other_case => unreachable!("Unexpected square case: {}", other_case),
            };
            segments.extend(square_segments);
        }
    }
    (segments, square_cases)
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

#[inline(always)]
fn add_top_neighbor(
    r0: &usize,
    c0: &usize,
    nb_cols_m1: &usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    is_head: bool,
) {
    let top_index = (r0 - 1) * nb_cols_m1 + c0;
    match (
        square_cases.get(top_index),
        positions.get(top_index),
        is_head,
    ) {
        (Some(8 | 10 | 11), Some(index), true) => {
            head_neighbors.push(Some(*index));
        }
        (Some(4 | 5 | 7), Some(index), false) => {
            tail_neighbors.push(Some(*index));
        }
        (Some(6), Some(index), false) => {
            tail_neighbors.push(Some(*index + 1));
        }
        (Some(9), Some(index), true) => {
            head_neighbors.push(Some(*index + 1));
        }
        (_, _, true) => {
            head_neighbors.push(None);
        }
        (_, _, false) => {
            tail_neighbors.push(None);
        }
    }
}

#[inline(always)]
fn add_left_neighbor(
    r0: &usize,
    c0: &usize,
    nb_cols_m1: &usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    is_head: bool,
) {
    let left_index = r0 * nb_cols_m1 + (c0 - 1);
    match (
        square_cases.get(left_index),
        positions.get(left_index),
        false,
        is_head,
    ) {
        (Some(2 | 3 | 7), Some(index), _, true) => {
            head_neighbors.push(Some(*index));
        }
        (Some(8 | 12 | 13), Some(index), _, false) => {
            tail_neighbors.push(Some(*index));
        }
        (Some(6), Some(index), true, true) => {
            head_neighbors.push(Some(*index + 1));
        }
        (Some(6), Some(index), false, true) => {
            head_neighbors.push(Some(*index));
        }
        (Some(9), Some(index), true, false) => {
            tail_neighbors.push(Some(*index));
        }
        (Some(9), Some(index), false, false) => {
            tail_neighbors.push(Some(*index + 1));
        }
        (_, _, _, true) => {
            head_neighbors.push(None);
        }
        (_, _, _, false) => {
            tail_neighbors.push(None);
        }
    }
}

#[inline(always)]
fn add_right_neighbor(
    r0: &usize,
    c0: &usize,
    nb_cols_m1: &usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    // is_fully_connected: bool,
    is_head: bool,
) {
    let right_index = r0 * nb_cols_m1 + (c0 + 1);
    match (
        square_cases.get(right_index),
        positions.get(right_index),
        false,
        is_head,
    ) {
        (Some(4 | 12 | 14), Some(index), _, true) => {
            head_neighbors.push(Some(*index));
        }
        (Some(1 | 3 | 11), Some(index), _, false) => {
            tail_neighbors.push(Some(*index));
        }
        (Some(6), Some(index), true, true) => {
            head_neighbors.push(Some(*index));
        }
        (Some(6), Some(index), false, true) => {
            head_neighbors.push(Some(*index + 1));
        }
        (Some(9), Some(index), true, false) => {
            tail_neighbors.push(Some(*index + 1));
        }
        (Some(9), Some(index), false, false) => {
            tail_neighbors.push(Some(*index));
        }
        (_, _, _, true) => {
            head_neighbors.push(None);
        }
        (_, _, _, false) => {
            tail_neighbors.push(None);
        }
    }
}

#[inline(always)]
fn add_bottom_neighbor(
    r0: &usize,
    c0: &usize,
    nb_cols_m1: &usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    is_head: bool,
) {
    let bottom_index = (r0 + 1) * nb_cols_m1 + c0;
    match (
        square_cases.get(bottom_index),
        positions.get(bottom_index),
        is_head,
    ) {
        (Some(1 | 5 | 13), Some(index), true) => {
            head_neighbors.push(Some(*index));
        }
        (Some(2 | 10 | 14), Some(index), false) => {
            tail_neighbors.push(Some(*index));
        }
        (Some(6), Some(index), false) => {
            tail_neighbors.push(Some(*index));
        }
        (Some(9), Some(index), true) => {
            head_neighbors.push(Some(*index));
        }
        (_, _, true) => {
            head_neighbors.push(None);
        }
        (_, _, false) => {
            tail_neighbors.push(None);
        }
    }
}

fn build_neighbors(
    square_cases: &Vec<u8>,
    _segments: &Vec<f64>,
    _nb_rows: usize,
    nb_cols: usize,
    vertex_connect_high: bool,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut segment_positions = vec![0; square_cases.len() + 1];
    for (i, square_case) in square_cases.iter().enumerate() {
        let nb_segments = match square_case {
            1 | 2 | 3 | 4 | 5 | 7 | 8 | 10 | 11 | 12 | 13 | 14 => 1,
            6 | 9 => 2,
            0 | 15 => 0,
            other_case => unreachable!("Unexpected square case: {}", other_case),
        };
        segment_positions[i + 1] = segment_positions[i] + nb_segments;
    }
    let mut head_neighbors = Vec::with_capacity(_segments.len() / 4);
    let mut tail_neighbors = Vec::with_capacity(_segments.len() / 4);
    let nb_cols_m1 = nb_cols - 1;
    for (square_index, square_case) in square_cases.iter().enumerate() {
        let r0 = square_index / nb_cols_m1;
        let c0 = square_index % nb_cols_m1;
        // let diff = head_neighbors.len();
        match square_case {
            0 | 15 => (),
            1 => {
                add_top_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_left_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            2 => {
                add_right_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_top_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            3 => {
                add_right_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_left_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            4 => {
                add_left_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_bottom_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            5 => {
                add_top_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_bottom_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            6 => match vertex_connect_high {
                true => {
                    add_left_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_top_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                    // seg 2
                    add_right_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_bottom_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                }
                false => {
                    add_right_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_top_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                    // seg 2
                    add_left_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_bottom_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                }
            },
            7 => {
                add_right_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_bottom_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            8 => {
                add_bottom_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_right_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            9 => match vertex_connect_high {
                true => {
                    add_top_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_right_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                    // seg 2
                    add_bottom_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_left_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                }
                false => {
                    add_top_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_left_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                    // seg 2
                    add_bottom_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_right_neighbor(
                        &r0,
                        &c0,
                        &nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                }
            },
            10 => {
                add_bottom_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_top_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            11 => {
                add_bottom_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_left_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            12 => {
                add_left_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_right_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            13 => {
                add_top_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_right_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            14 => {
                add_left_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_top_neighbor(
                    &r0,
                    &c0,
                    &nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            _ => unreachable!("Unexpected square case: {}", square_case),
        }
        // _print_segment_by_position(
        //     current_index,
        //     &segment_positions,
        //     &segments,
        //     &square_neighbors,
        // );
        // match square_case {
        //     0 | 15 => assert_eq!(head_neighbors.len(), diff, "{square_case}"),
        //     6 | 9 => assert_eq!(head_neighbors.len(), diff + 2, "{square_case}"),
        //     _ => assert_eq!(head_neighbors.len(), diff + 1, "{square_case}"),
        // }
    }
    assert_eq!(tail_neighbors.len(), _segments.len() / 4);
    assert_eq!(head_neighbors.len(), _segments.len() / 4);
    (head_neighbors, tail_neighbors)
}

fn assemble_contours(
    segments: &Vec<f64>,
    head_neighbors: &Vec<Option<usize>>,
    tail_neighbors: &Vec<Option<usize>>,
    tol: f64,
) -> Vec<ArrayD<f64>> {
    let mut contours = Vec::with_capacity(segments.len() / 4);
    let mut visited = vec![false; segments.len() / 4];
    for first_index in 0..(segments.len() / 4) {
        if visited[first_index] {
            continue;
        }
        let mut contour = Vec::new();
        let mut tail_index = first_index;
        let mut head_index = first_index;
        visited[first_index] = true;
        // first point
        contour.push(segments[4 * first_index]);
        contour.push(segments[4 * first_index + 1]);
        // second point
        contour.push(segments[4 * first_index + 2]);
        contour.push(segments[4 * first_index + 3]);
        let mut nb_points = 0;
        while contour.len() > nb_points {
            nb_points = contour.len();
            match (
                find_next_segment(&segments, &visited, &head_neighbors, head_index, tol),
                find_previous_segment(&segments, &visited, &tail_neighbors, tail_index, tol),
            ) {
                (Some(next_index), None) => {
                    contour.push(segments[4 * next_index + 2]);
                    contour.push(segments[4 * next_index + 3]);
                    head_index = next_index;
                    visited[next_index] = true;
                }
                (None, Some(prev_index)) => {
                    // inserted in reverse to make x the first
                    contour.insert(0, segments[4 * prev_index + 1]);
                    contour.insert(0, segments[4 * prev_index + 0]);
                    tail_index = prev_index;
                    visited[prev_index] = true;
                }
                (Some(next_index), Some(prev_index)) => {
                    if next_index <= prev_index {
                        contour.push(segments[4 * next_index + 2]);
                        contour.push(segments[4 * next_index + 3]);
                        head_index = next_index;
                        visited[next_index] = true;
                    } else {
                        // inserted in reverse to make x the first
                        contour.insert(0, segments[4 * prev_index + 1]);
                        contour.insert(0, segments[4 * prev_index + 0]);
                        tail_index = prev_index;
                        visited[prev_index] = true;
                    }
                }
                (None, None) => (),
            }
        }
        let shape = IxDyn(&[contour.len() / 2, 2]);
        contours.push(ArrayD::from_shape_vec(shape, contour).unwrap());
    }
    return contours;
}

#[inline(always)]
fn find_next_segment(
    segments: &Vec<f64>,
    visited: &Vec<bool>,
    neighbors: &Vec<Option<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    unsafe {
        if let Some(next_index) = neighbors.get_unchecked(index) {
            if !visited.get_unchecked(*next_index)
                && close(
                    // first point of the next_index-th segment
                    *segments.get_unchecked(4 * next_index + 0),
                    *segments.get_unchecked(4 * next_index + 1),
                    // second point of the index-th segment
                    *segments.get_unchecked(4 * index + 2),
                    *segments.get_unchecked(4 * index + 3),
                    tol,
                )
            {
                return Some(*next_index);
            }
        }
    }
    None
}

#[inline(always)]
fn find_previous_segment(
    segments: &Vec<f64>,
    visited: &Vec<bool>,
    neighbors: &Vec<Option<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    unsafe {
        if let Some(prev_index) = neighbors.get_unchecked(index) {
            if !visited.get_unchecked(*prev_index)
                && close(
                    // first point of the index-th segment
                    *segments.get_unchecked(4 * index + 0),
                    *segments.get_unchecked(4 * index + 1),
                    // second point of the prev_index-th segment
                    *segments.get_unchecked(4 * prev_index + 2),
                    *segments.get_unchecked(4 * prev_index + 3),
                    tol,
                )
            {
                return Some(*prev_index);
            }
        }
    }
    None
}

#[pyfunction]
#[pyo3(signature=(array, level, mask, vertex_connect_high=false))]
fn get_contour_segments<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    level: f64,
    mask: Option<PyReadonlyArrayDyn<'py, u8>>,
    vertex_connect_high: bool,
) -> Bound<'py, PyArrayDyn<f64>> {
    let array = array.as_array();
    assert_eq!(
        array.shape().len(),
        2,
        "Only 2d dimension array can be used"
    );
    let segments = match mask {
        Some(mask) => {
            let mask = mask.as_array();
            debug_assert_eq!(
                array.len(),
                mask.len(),
                "The array and the mask should have the same length: {array_len}!={mask_len}",
                array_len = array.len(),
                mask_len = mask.len()
            );
            _get_contour_segments(&array, level, vertex_connect_high, &mask).0
        }
        None => {
            let mask = Array::from_shape_vec(array.shape(), vec![1u8; array.len()]).unwrap();
            let mask = mask.view().into_dyn();
            _get_contour_segments(&array, level, vertex_connect_high, &mask).0
        }
    };
    let shape = IxDyn(&[segments.len() / 4, 2, 2]);
    ArrayD::from_shape_vec(shape, segments)
        .unwrap()
        .into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature=(array, level, mask, is_fully_connected=false, tol=1e-16))]
fn marching_squares<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    level: f64,
    mask: Option<PyReadonlyArrayDyn<'py, u8>>,
    is_fully_connected: bool,
    tol: f64,
) -> Vec<Bound<'py, PyArrayDyn<f64>>> {
    let array = array.as_array();
    assert_eq!(
        array.shape().len(),
        2,
        "Only 2d dimension array can be used"
    );
    let (segments, square_cases) = match mask {
        Some(mask) => {
            let mask = mask.as_array();
            debug_assert_eq!(
                array.len(),
                mask.len(),
                "The array and the mask should have the same length: {array_len}!={mask_len}",
                array_len = array.len(),
                mask_len = mask.len()
            );
            _get_contour_segments(&array, level, is_fully_connected, &mask)
        }
        None => {
            let mask = Array::from_shape_vec(array.shape(), vec![1u8; array.len()]).unwrap();
            let mask = mask.view().into_dyn();
            _get_contour_segments(&array, level, is_fully_connected, &mask)
        }
    };
    let shape = array.shape();
    let (nb_rows, nb_cols) = (shape[0], shape[1]);
    let (head_neighbors, tail_neighbors) = build_neighbors(
        &square_cases,
        &segments,
        nb_rows,
        nb_cols,
        is_fully_connected,
    );
    let contours = assemble_contours(&segments, &head_neighbors, &tail_neighbors, tol);
    contours.into_iter().map(|c| c.into_pyarray(py)).collect()
}

// Marching squares algorithm
#[pymodule]
fn _marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(close, m)?)?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
