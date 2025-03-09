use pyo3::{exceptions::PyValueError, prelude::*};

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

#[inline]
fn marching_square(
    r0: usize,
    c0: usize,
    r1: usize,
    c1: usize,
    ul: f64,
    ur: f64,
    ll: f64,
    lr: f64,
    level: f64,
    vertex_connect_high: bool,
) -> Vec<Option<f64>> {
    let square_case = 1 * u8::from(ul > level) as u8
        | 2 * (ur > level) as u8
        | 4 * (ll > level) as u8
        | 8 * (lr > level) as u8;
    match square_case {
        1 => vec![
            Some(top_x(r0, c0, ul, ur, level)),
            Some(top_y(r0, c0, ul, ur, level)),
            Some(left_x(r0, c0, ul, ll, level)),
            Some(left_y(r0, c0, ul, ll, level)),
            None,
            None,
            None,
            None,
        ],
        2 => vec![
            Some(right_x(r0, c1, ur, lr, level)),
            Some(right_y(r0, c1, ur, lr, level)),
            Some(top_x(r0, c0, ul, ur, level)),
            Some(top_y(r0, c0, ul, ur, level)),
            None,
            None,
            None,
            None,
        ],
        3 => vec![
            Some(right_x(r0, c1, ur, lr, level)),
            Some(right_y(r0, c1, ur, lr, level)),
            Some(left_x(r0, c0, ul, ll, level)),
            Some(left_y(r0, c0, ul, ll, level)),
            None,
            None,
            None,
            None,
        ],
        4 => vec![
            Some(left_x(r0, c0, ul, ll, level)),
            Some(left_y(r0, c0, ul, ll, level)),
            Some(bottom_x(r1, c0, ll, lr, level)),
            Some(bottom_y(r1, c0, ll, lr, level)),
            None,
            None,
            None,
            None,
        ],
        5 => vec![
            Some(top_x(r0, c0, ul, ur, level)),
            Some(top_y(r0, c0, ul, ur, level)),
            Some(bottom_x(r1, c0, ll, lr, level)),
            Some(bottom_y(r1, c0, ll, lr, level)),
            None,
            None,
            None,
            None,
        ],
        6 => match vertex_connect_high {
            true => vec![
                Some(left_x(r0, c0, ul, ll, level)),
                Some(left_y(r0, c0, ul, ll, level)),
                Some(top_x(r0, c0, ul, ur, level)),
                Some(top_y(r0, c0, ul, ur, level)),
                // seg 2
                Some(right_x(r0, c1, ur, lr, level)),
                Some(right_y(r0, c1, ur, lr, level)),
                Some(bottom_x(r1, c0, ll, lr, level)),
                Some(bottom_y(r1, c0, ll, lr, level)),
            ],
            false => vec![
                Some(right_x(r0, c1, ur, lr, level)),
                Some(right_y(r0, c1, ur, lr, level)),
                Some(top_x(r0, c0, ul, ur, level)),
                Some(top_y(r0, c0, ul, ur, level)),
                // seg 2
                Some(left_x(r0, c0, ul, ll, level)),
                Some(left_y(r0, c0, ul, ll, level)),
                Some(bottom_x(r1, c0, ll, lr, level)),
                Some(bottom_y(r1, c0, ll, lr, level)),
            ],
        },
        7 => vec![
            Some(right_x(r0, c1, ur, lr, level)),
            Some(right_y(r0, c1, ur, lr, level)),
            Some(bottom_x(r1, c0, ll, lr, level)),
            Some(bottom_y(r1, c0, ll, lr, level)),
            None,
            None,
            None,
            None,
        ],
        8 => vec![
            Some(bottom_x(r1, c0, ll, lr, level)),
            Some(bottom_y(r1, c0, ll, lr, level)),
            Some(right_x(r0, c1, ur, lr, level)),
            Some(right_y(r0, c1, ur, lr, level)),
            None,
            None,
            None,
            None,
        ],
        9 => match vertex_connect_high {
            true => vec![
                Some(top_x(r0, c0, ul, ur, level)),
                Some(top_y(r0, c0, ul, ur, level)),
                Some(right_x(r0, c1, ur, lr, level)),
                Some(right_y(r0, c1, ur, lr, level)),
                // seg 2
                Some(bottom_x(r1, c0, ll, lr, level)),
                Some(bottom_y(r1, c0, ll, lr, level)),
                Some(left_x(r0, c0, ul, ll, level)),
                Some(left_y(r0, c0, ul, ll, level)),
            ],
            false => vec![
                Some(top_x(r0, c0, ul, ur, level)),
                Some(top_y(r0, c0, ul, ur, level)),
                Some(left_x(r0, c0, ul, ll, level)),
                Some(left_y(r0, c0, ul, ll, level)),
                // seg 2
                Some(bottom_x(r1, c0, ll, lr, level)),
                Some(bottom_y(r1, c0, ll, lr, level)),
                Some(right_x(r0, c1, ur, lr, level)),
                Some(right_y(r0, c1, ur, lr, level)),
            ],
        },
        10 => vec![
            Some(bottom_x(r1, c0, ll, lr, level)),
            Some(bottom_y(r1, c0, ll, lr, level)),
            Some(top_x(r0, c0, ul, ur, level)),
            Some(top_y(r0, c0, ul, ur, level)),
            None,
            None,
            None,
            None,
        ],
        11 => vec![
            Some(bottom_x(r1, c0, ll, lr, level)),
            Some(bottom_y(r1, c0, ll, lr, level)),
            Some(left_x(r0, c0, ul, ll, level)),
            Some(left_y(r0, c0, ul, ll, level)),
            None,
            None,
            None,
            None,
        ],
        12 => vec![
            Some(left_x(r0, c0, ul, ll, level)),
            Some(left_y(r0, c0, ul, ll, level)),
            Some(right_x(r0, c1, ur, lr, level)),
            Some(right_y(r0, c1, ur, lr, level)),
            None,
            None,
            None,
            None,
        ],
        13 => vec![
            Some(top_x(r0, c0, ul, ur, level)),
            Some(top_y(r0, c0, ul, ur, level)),
            Some(right_x(r0, c1, ur, lr, level)),
            Some(right_y(r0, c1, ur, lr, level)),
            None,
            None,
            None,
            None,
        ],
        14 => vec![
            Some(left_x(r0, c0, ul, ll, level)),
            Some(left_y(r0, c0, ul, ll, level)),
            Some(top_x(r0, c0, ul, ur, level)),
            Some(top_y(r0, c0, ul, ur, level)),
            None,
            None,
            None,
            None,
        ],
        0 | 15 => {
            // No segments pass through the square
            vec![None; 8]
        }
        other_case => unreachable!("Unexpected case: {}", other_case),
    }
}

#[inline]
fn to_grid_mask(
    mask: Option<&Vec<bool>>,
    array: &Vec<f64>,
    nb_rows: usize,
    nb_cols: usize,
) -> Result<Vec<u8>, String> {
    let mut grid_mask = vec![1; (nb_rows - 1) * (nb_cols - 1)];
    match mask {
        Some(mask) => {
            if array.len() != mask.len() {
                return Err(format!(
            "The array and the mask must have the same length, {array_len:?} != {mask_len:?}",
            array_len = array.len(),
            mask_len = mask.len(),
        ));
            }
            for r0 in 0..(nb_rows - 1) {
                let r1 = r0 + 1;
                for c0 in 0..(nb_cols - 1) {
                    let c1 = c0 + 1;
                    unsafe {
                        grid_mask[r0 * (nb_cols - 1) + c0] = *mask.get_unchecked(r0 * nb_cols + c0)
                            as u8
                            * *mask.get_unchecked(r0 * nb_cols + c1) as u8
                            * *mask.get_unchecked(r1 * nb_cols + c0) as u8
                            * *mask.get_unchecked(r1 * nb_cols + c1) as u8;
                    }
                }
            }
        }
        None => (),
    }
    Ok(grid_mask)
}

fn _get_contour_segments(
    array: &Vec<f64>,
    nb_rows: usize,
    nb_cols: usize,
    level: f64,
    vertex_connect_high: bool,
    grid_mask: &Vec<u8>,
) -> Result<(Vec<f64>, Vec<Option<usize>>), String> {
    if array.len() != nb_rows * nb_cols {
        return Err(format!(
            "The array and the given shape are incompatible: {array_len:?} != ({nb_rows} * {nb_cols})",
            array_len = array.len(),
        ));
    }
    let mut segments = Vec::with_capacity(nb_rows * nb_cols);
    let mut indices = vec![None; 2 * nb_rows * nb_cols];
    let mut current_index: usize = 0;
    for r0 in 0..(nb_rows - 1) {
        let r1 = r0 + 1;
        for c0 in 0..(nb_cols - 1) {
            let c1 = c0 + 1;
            let ul_index = r0 * nb_cols + c0;
            let ur_index = r0 * nb_cols + c1;
            let ll_index = r1 * nb_cols + c0;
            let lr_index = r1 * nb_cols + c1;
            let (ul, ll, ur, lr): (f64, f64, f64, f64);
            let is_masked: bool;
            unsafe {
                ul = *array.get_unchecked(ul_index);
                // .expect("The iterator should be bound by the number of columns of the array");
                ll = *array.get_unchecked(ll_index);
                // .expect("The iterator should be bound by the number of columns of the array");
                ur = *array.get_unchecked(ur_index);
                // .expect("The iterator should end one column before the end");
                lr = *array.get_unchecked(lr_index);
                // .expect("The iterator should end one column before the end");
                is_masked = *grid_mask.get_unchecked(r0 * (nb_cols - 1) + c0) == 0;
            }
            match is_masked {
                true => current_index += 2,
                false => {
                    let segment =
                        marching_square(r0, c0, r1, c1, ul, ur, ll, lr, level, vertex_connect_high);
                    // Segment 1
                    if let (Some(s1_p1_x), Some(s1_p1_y), Some(s1_p2_x), Some(s1_p2_y)) =
                        (segment[0], segment[1], segment[2], segment[3])
                    {
                        indices[current_index] = Some(segments.len() / 4);
                        segments.push(s1_p1_x);
                        segments.push(s1_p1_y);
                        segments.push(s1_p2_x);
                        segments.push(s1_p2_y);
                    }
                    current_index += 1;
                    // Segment 2
                    if let (Some(s2_p1_x), Some(s2_p1_y), Some(s2_p2_x), Some(s2_p2_y)) =
                        (segment[4], segment[5], segment[6], segment[7])
                    {
                        indices[current_index] = Some(segments.len() / 4);
                        segments.push(s2_p1_x);
                        segments.push(s2_p1_y);
                        segments.push(s2_p2_x);
                        segments.push(s2_p2_y);
                    }
                    current_index += 1;
                }
            }
        }
        // add the last column
        current_index += 2;
    }
    assert_eq!(
        current_index + 2 * nb_cols, // add the last row
        2 * nb_rows * nb_cols
    );
    Ok((segments, indices))
}

fn assemble_contours(
    segments: &Vec<f64>,
    indices: &Vec<Option<usize>>,
    nb_cols: usize,
    tol: f64,
) -> Vec<Vec<Vec<f64>>> {
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
        contour.push(vec![
            segments[4 * first_index],
            segments[4 * first_index + 1],
        ]);
        contour.push(vec![
            segments[4 * first_index + 2],
            segments[4 * first_index + 3],
        ]);
        let mut nb_points = 0;
        while contour.len() > nb_points {
            nb_points = contour.len();
            match (
                find_next_segment(&segments, &visited, &neighbors, head_index, tol),
                find_previous_segment(&segments, &visited, &neighbors, tail_index, tol),
            ) {
                (Some(next_index), None) => {
                    contour.push(vec![
                        segments[4 * next_index + 2],
                        segments[4 * next_index + 3],
                    ]);
                    head_index = next_index;
                    visited[next_index] = true;
                }
                (None, Some(prev_index)) => {
                    contour.insert(
                        0,
                        vec![segments[4 * prev_index + 0], segments[4 * prev_index + 1]],
                    );
                    tail_index = prev_index;
                    visited[prev_index] = true;
                }
                (Some(next_index), Some(prev_index)) => {
                    if next_index <= prev_index {
                        contour.push(vec![
                            segments[4 * next_index + 2],
                            segments[4 * next_index + 3],
                        ]);
                        head_index = next_index;
                        visited[next_index] = true;
                    } else {
                        contour.insert(
                            0,
                            vec![segments[4 * prev_index + 0], segments[4 * prev_index + 1]],
                        );
                        tail_index = prev_index;
                        visited[prev_index] = true;
                    }
                }
                (None, None) => (),
            }
        }
        if contour.len() > 0 {
            contours.push(contour);
        }
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
#[pyo3(signature=(array, shape, level, vertex_connect_high=false, mask=None))]
fn get_contour_segments(
    array: Vec<f64>,
    shape: (usize, usize),
    level: f64,
    vertex_connect_high: bool,
    mask: Option<Vec<bool>>,
) -> PyResult<Vec<f64>> {
    let grid_mask =
        to_grid_mask(mask.as_ref(), &array, shape.0, shape.1).map_err(PyValueError::new_err)?;
    match _get_contour_segments(
        &array,
        shape.0,
        shape.1,
        level,
        vertex_connect_high,
        &grid_mask,
    ) {
        Ok((segments, _)) => Ok(segments),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature=(array, shape, level, is_fully_connected=false, mask=None, tol=1e-10))]
fn marching_squares(
    array: Vec<f64>,
    shape: (usize, usize),
    level: f64,
    is_fully_connected: bool,
    mask: Option<Vec<bool>>,
    tol: f64,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let (nb_rows, nb_cols) = shape;
    let grid_mask =
        to_grid_mask(mask.as_ref(), &array, nb_rows, nb_cols).map_err(PyValueError::new_err)?;
    match _get_contour_segments(
        &array,
        shape.0,
        shape.1,
        level,
        is_fully_connected,
        &grid_mask,
    ) {
        Ok((segments, indices)) => {
            debug_assert_eq!(indices.len(), 2 * nb_rows * nb_cols);
            Ok(assemble_contours(&segments, &indices, nb_cols, tol))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

// Marching squares algorithm
#[pymodule]
fn marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(close, m)?)?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
