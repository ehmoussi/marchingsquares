use numpy::ndarray::{Array, ArrayViewD};
use numpy::PyReadonlyArrayDyn;
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
    nb_rows: usize,
    nb_cols: usize,
    level: f64,
    vertex_connect_high: bool,
    mask: &ArrayViewD<'_, u8>,
) -> (Vec<f64>, Vec<u8>) {
    let mut segments = Vec::with_capacity(nb_rows * nb_cols);
    let mut square_cases: Vec<u8> = Vec::with_capacity((nb_rows - 1) * (nb_cols - 1));
    for r0 in 0..(nb_rows - 1) {
        let r1 = r0 + 1;
        for c0 in 0..(nb_cols - 1) {
            let c1 = c0 + 1;
            let (ul, ll, ur, lr): (f64, f64, f64, f64);
            let is_masked: u8;
            let ul_index = r0 * nb_cols + c0;
            let ll_index = r1 * nb_cols + c0;
            let ur_index = r0 * nb_cols + c1;
            let lr_index = r1 * nb_cols + c1;
            ul = array[ul_index];
            ll = array[ll_index];
            ur = array[ur_index];
            lr = array[lr_index];
            is_masked = mask[ul_index] * mask[ll_index] * mask[ur_index] * mask[lr_index];
            let square_case = is_masked
                * (1 * u8::from(ul > level) as u8
                    | 2 * (ur > level) as u8
                    | 4 * (ll > level) as u8
                    | 8 * (lr > level) as u8);
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
fn top_index(r0: &usize, c0: &usize, nb_cols: &usize) -> Option<usize> {
    r0.checked_add_signed(-1)
        .and_then(|r| Some(r * (nb_cols - 1)))
        .and_then(|idx| Some(idx + c0))
}

#[inline(always)]
fn left_index(r0: &usize, c0: &usize, nb_cols: &usize) -> Option<usize> {
    c0.checked_add_signed(-1)
        .and_then(|c| Some(r0 * (nb_cols - 1) + c))
}

#[inline(always)]
fn right_index(r0: &usize, c0: &usize, nb_cols: &usize) -> Option<usize> {
    Some(r0 * (nb_cols - 1) + (c0 + 1))
}

#[inline(always)]
fn bottom_index(r0: &usize, c0: &usize, nb_cols: &usize) -> Option<usize> {
    Some((r0 + 1) * (nb_cols - 1) + c0)
}

#[inline(always)]
fn add_neighbor(index: Option<usize>, neighbors: &mut Vec<usize>, positions: &Vec<usize>) {
    if let Some(index) = index {
        if let (Some(&start), Some(&end)) = (positions.get(index), positions.get(index + 1)) {
            for i in start..end {
                neighbors.push(i);
            }
        }
    }
}

fn build_neighbors(
    square_cases: &Vec<u8>,
    _segments: &Vec<f64>,
    nb_cols: usize,
    vertex_connect_high: bool,
) -> Vec<Vec<usize>> {
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
    let nb_segments = segment_positions[segment_positions.len() - 1];
    let mut neighbors = Vec::with_capacity(nb_segments / 4);
    for (square_index, square_case) in square_cases.iter().enumerate() {
        let r0 = square_index / (nb_cols - 1);
        let c0 = square_index % (nb_cols - 1);
        let mut square_neighbors = Vec::new();
        match square_case {
            0 | 15 => (),
            1 => {
                add_neighbor(
                    top_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    left_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            2 => {
                add_neighbor(
                    right_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    top_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            3 => {
                add_neighbor(
                    right_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    left_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            4 => {
                add_neighbor(
                    left_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    bottom_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            5 => {
                add_neighbor(
                    top_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    bottom_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            6 => match vertex_connect_high {
                true => {
                    add_neighbor(
                        left_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        top_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    // seg 2
                    add_neighbor(
                        right_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        bottom_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                }
                false => {
                    add_neighbor(
                        right_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        top_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    // seg 2
                    add_neighbor(
                        left_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        bottom_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                }
            },
            7 => {
                add_neighbor(
                    right_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    bottom_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            8 => {
                add_neighbor(
                    bottom_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    right_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            9 => match vertex_connect_high {
                true => {
                    add_neighbor(
                        top_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        right_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    // seg 2
                    add_neighbor(
                        bottom_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        left_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                }
                false => {
                    add_neighbor(
                        top_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        left_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    // seg 2
                    add_neighbor(
                        bottom_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                    add_neighbor(
                        right_index(&r0, &c0, &nb_cols),
                        &mut square_neighbors,
                        &segment_positions,
                    );
                }
            },
            10 => {
                add_neighbor(
                    bottom_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    top_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            11 => {
                add_neighbor(
                    bottom_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    left_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            12 => {
                add_neighbor(
                    left_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    right_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            13 => {
                add_neighbor(
                    top_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    right_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
            }
            14 => {
                add_neighbor(
                    left_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
                );
                add_neighbor(
                    top_index(&r0, &c0, &nb_cols),
                    &mut square_neighbors,
                    &segment_positions,
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
        match square_case {
            0 | 15 => (),
            1..6 => neighbors.push(square_neighbors),
            7..9 => neighbors.push(square_neighbors),
            10..15 => neighbors.push(square_neighbors),
            6 | 9 => {
                neighbors.push(square_neighbors.clone());
                neighbors.push(square_neighbors);
            }
            _ => unreachable!("Unexpected square case: {}", square_case),
        }
    }
    neighbors
}

fn assemble_contours(segments: &Vec<f64>, neighbors: &Vec<Vec<usize>>, tol: f64) -> Vec<Vec<f64>> {
    let mut contours = Vec::with_capacity(segments.len() / 4);
    let mut visited = vec![false; segments.len()];
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
                find_next_segment(&segments, &visited, &neighbors, head_index, tol),
                find_previous_segment(&segments, &visited, &neighbors, tail_index, tol),
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
        contours.push(contour);
    }
    return contours;
}

#[inline(always)]
fn find_next_segment(
    segments: &Vec<f64>,
    visited: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    if let Some(neighbors_index) = neighbors.get(index) {
        for &next_index in neighbors_index {
            if !visited[next_index] {
                if close(
                    // first point of the next_index-th segment
                    segments[4 * next_index + 0],
                    segments[4 * next_index + 1],
                    // second point of the index-th segment
                    segments[4 * index + 2],
                    segments[4 * index + 3],
                    tol,
                ) {
                    return Some(next_index);
                }
            }
        }
    }
    None
}

#[inline(always)]
fn find_previous_segment(
    segments: &Vec<f64>,
    visited: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    if let Some(neighbors_index) = neighbors.get(index) {
        for &prev_index in neighbors_index {
            if !visited[prev_index] {
                if close(
                    // first point of the index-th segment
                    segments[4 * index + 0],
                    segments[4 * index + 1],
                    // second point of the prev_index-th segment
                    segments[4 * prev_index + 2],
                    segments[4 * prev_index + 3],
                    tol,
                ) {
                    return Some(prev_index);
                }
            }
        }
    }
    None
}

#[pyfunction]
#[pyo3(signature=(array, shape, level, mask, vertex_connect_high=false))]
fn get_contour_segments<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    shape: (usize, usize),
    level: f64,
    mask: Option<PyReadonlyArrayDyn<'py, u8>>,
    vertex_connect_high: bool,
) -> Bound<'py, PyArrayDyn<f64>> {
    let array = array.as_array();
    debug_assert_eq!(
        array.len(),
        shape.0 * shape.1,
        "The shape of the array is incorrect {array_len}!={shape_0}*{shape_1}",
        array_len = array.len(),
        shape_0 = shape.0,
        shape_1 = shape.1
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
            _get_contour_segments(&array, shape.0, shape.1, level, vertex_connect_high, &mask).0
        }
        None => {
            let mask = Array::from_vec(vec![1u8; array.len()]);
            let mask = mask.view().into_dyn();
            _get_contour_segments(&array, shape.0, shape.1, level, vertex_connect_high, &mask).0
        }
    };
    Array::from_vec(segments).into_dyn().into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature=(array, shape, level, mask, is_fully_connected=false, tol=1e-16))]
fn marching_squares<'py>(
    array: PyReadonlyArrayDyn<'py, f64>,
    shape: (usize, usize),
    level: f64,
    mask: Option<PyReadonlyArrayDyn<'py, u8>>,
    is_fully_connected: bool,
    tol: f64,
) -> Vec<Vec<f64>> {
    let array = array.as_array();
    debug_assert_eq!(
        array.len(),
        shape.0 * shape.1,
        "The shape of the array is incorrect {array_len}!={shape_0}*{shape_1}",
        array_len = array.len(),
        shape_0 = shape.0,
        shape_1 = shape.1
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
            _get_contour_segments(&array, shape.0, shape.1, level, is_fully_connected, &mask)
        }
        None => {
            let mask = Array::from_vec(vec![1u8; array.len()]);
            let mask = mask.view().into_dyn();
            _get_contour_segments(&array, shape.0, shape.1, level, is_fully_connected, &mask)
        }
    };
    let (_nb_rows, nb_cols) = shape;
    let neighbors = build_neighbors(&square_cases, &segments, nb_cols, is_fully_connected);
    assemble_contours(&segments, &neighbors, tol)
}

// Marching squares algorithm
#[pymodule]
fn _marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(close, m)?)?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
