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
    array: &Vec<f64>,
    nb_rows: usize,
    nb_cols: usize,
    level: f64,
    vertex_connect_high: bool,
    mask: &Vec<u8>,
) -> (Vec<f64>, Vec<Option<usize>>) {
    let mut segments = Vec::with_capacity(nb_rows * nb_cols);
    let mut indices = vec![None; 2 * nb_rows * nb_cols];
    let mut current_index: usize = 0;
    for r0 in 0..(nb_rows - 1) {
        let r1 = r0 + 1;
        for c0 in 0..(nb_cols - 1) {
            let c1 = c0 + 1;
            let (ul, ll, ur, lr): (f64, f64, f64, f64);
            let is_masked: u8;
            unsafe {
                let ul_index = r0 * nb_cols + c0;
                let ll_index = r1 * nb_cols + c0;
                let ur_index = r0 * nb_cols + c1;
                let lr_index = r1 * nb_cols + c1;
                ul = *array.get_unchecked(ul_index);
                ll = *array.get_unchecked(ll_index);
                ur = *array.get_unchecked(ur_index);
                lr = *array.get_unchecked(lr_index);
                is_masked = *mask.get_unchecked(ul_index)
                    * *mask.get_unchecked(ll_index)
                    * *mask.get_unchecked(ur_index)
                    * *mask.get_unchecked(lr_index);
            }
            if is_masked == 0 {
                current_index += 2;
                continue;
            }
            let square_case = 1 * u8::from(ul > level) as u8
                | 2 * (ur > level) as u8
                | 4 * (ll > level) as u8
                | 8 * (lr > level) as u8;
            match square_case {
                1 | 2 | 3 | 4 | 5 | 7 | 8 | 10 | 11 | 12 | 13 | 14 => {
                    indices[current_index] = Some(segments.len() / 4);
                    current_index += 2;
                }
                6 | 9 => {
                    indices[current_index] = Some(segments.len() / 4);
                    current_index += 1;
                    indices[current_index] = Some((segments.len() / 4) + 1);
                    current_index += 1;
                }
                0 | 15 => current_index += 2,
                other_case => unreachable!("Unexpected case: {}", other_case),
            }
            match square_case {
                1 => segments.extend([
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ]),
                2 => segments.extend([
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ]),
                3 => segments.extend([
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ]),
                4 => segments.extend([
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ]),
                5 => segments.extend([
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ]),
                6 => match vertex_connect_high {
                    true => segments.extend([
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        // seg 2
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                    ]),
                    false => segments.extend([
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        // seg 2
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                    ]),
                },
                7 => segments.extend([
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ]),
                8 => segments.extend([
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ]),
                9 => match vertex_connect_high {
                    true => segments.extend([
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        // seg 2
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                    ]),
                    false => segments.extend([
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        // seg 2
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                    ]),
                },
                10 => segments.extend([
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ]),
                11 => segments.extend([
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ]),
                12 => segments.extend([
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ]),
                13 => segments.extend([
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ]),
                14 => segments.extend([
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ]),
                0 | 15 => (), // No segments pass through the square
                other_case => unreachable!("Unexpected case: {}", other_case),
            }
        }
        // add the last column
        current_index += 2;
    }
    assert_eq!(
        current_index + 2 * nb_cols, // add the last row
        2 * nb_rows * nb_cols
    );
    (segments, indices)
}

fn assemble_contours(
    segments: &Vec<f64>,
    indices: &Vec<Option<usize>>,
    nb_cols: usize,
    tol: f64,
) -> Vec<Vec<f64>> {
    let mut contours = Vec::with_capacity(segments.len() / 4);
    let mut visited = vec![false; segments.len()];
    let mut neighbors = Vec::with_capacity(segments.len() / 4);
    for (index, seg_index) in indices.iter().enumerate() {
        match seg_index {
            Some(_) => {
                let mut neighbors_index = Vec::with_capacity(8);
                for (i, j) in [(-1, 0), (0, -1), (0, 1), (1, 0)].iter() {
                    for k in 0..2 {
                        if let Some(n_index) = index.checked_add_signed(
                            i * 2 * (nb_cols as isize) + 2 * j + k - (index as isize % 2),
                        ) {
                            if let Some(neighbor_index) = indices[n_index] {
                                neighbors_index.push(neighbor_index);
                            }
                        }
                    }
                }
                neighbors.push(neighbors_index);
            }
            None => (),
        }
    }
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

#[inline]
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

#[inline]
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
fn get_contour_segments(
    array: Vec<f64>,
    shape: (usize, usize),
    level: f64,
    mask: Vec<u8>,
    vertex_connect_high: bool,
) -> Vec<f64> {
    assert_eq!(
        array.len(),
        shape.0 * shape.1,
        "The shape of the array is incorrect {array_len}!={shape_0}*{shape_1}",
        array_len = array.len(),
        shape_0 = shape.0,
        shape_1 = shape.1
    );
    assert_eq!(
        array.len(),
        mask.len(),
        "The array and the mask should have the same length: {array_len}!={mask_len}",
        array_len = array.len(),
        mask_len = mask.len()
    );
    _get_contour_segments(&array, shape.0, shape.1, level, vertex_connect_high, &mask).0
}

#[pyfunction]
#[pyo3(signature=(array, shape, level, mask, is_fully_connected=false, tol=1e-16))]
fn marching_squares(
    array: Vec<f64>,
    shape: (usize, usize),
    level: f64,
    mask: Vec<u8>,
    is_fully_connected: bool,
    tol: f64,
) -> Vec<Vec<f64>> {
    assert_eq!(
        array.len(),
        shape.0 * shape.1,
        "The shape of the array is incorrect {array_len}!={shape_0}*{shape_1}",
        array_len = array.len(),
        shape_0 = shape.0,
        shape_1 = shape.1
    );
    assert_eq!(
        array.len(),
        mask.len(),
        "The array and the mask should have the same length: {array_len}!={mask_len}",
        array_len = array.len(),
        mask_len = mask.len()
    );
    let (nb_rows, nb_cols) = shape;
    let (segments, indices) =
        _get_contour_segments(&array, shape.0, shape.1, level, is_fully_connected, &mask);
    debug_assert_eq!(indices.len(), 2 * nb_rows * nb_cols);
    assemble_contours(&segments, &indices, nb_cols, tol)
}

// Marching squares algorithm
#[pymodule]
fn _marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(close, m)?)?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
