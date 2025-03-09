use pyo3::{exceptions::PyValueError, prelude::*};

#[inline]
fn get_fraction(from_value: f64, to_value: f64, level: f64) -> f64 {
    if to_value == from_value {
        return 0.0;
    }
    return (level - from_value) / (to_value - from_value);
}

#[pyclass]
#[derive(Debug, PartialEq, Clone)]
struct Point {
    #[pyo3(get, set)]
    x: f64,
    #[pyo3(get, set)]
    y: f64,
}

#[pymethods]
impl Point {
    fn close(&self, other: &Point, tol: f64) -> bool {
        (self.x - other.x).abs() < tol && (self.y - other.y).abs() < tol
    }

    #[staticmethod]
    fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    #[staticmethod]
    #[inline]
    fn top(r: usize, c: usize, ul: f64, ur: f64, level: f64) -> Point {
        Point {
            x: r as f64,
            y: c as f64 + get_fraction(ul, ur, level),
        }
    }

    #[staticmethod]
    #[inline]
    fn bottom(r: usize, c: usize, ll: f64, lr: f64, level: f64) -> Point {
        Point {
            x: r as f64,
            y: c as f64 + get_fraction(ll, lr, level),
        }
    }

    #[staticmethod]
    #[inline]
    fn left(r: usize, c: usize, ul: f64, ll: f64, level: f64) -> Point {
        Point {
            x: r as f64 + get_fraction(ul, ll, level),
            y: c as f64,
        }
    }

    #[staticmethod]
    #[inline]
    fn right(r: usize, c: usize, ur: f64, lr: f64, level: f64) -> Point {
        Point {
            x: r as f64 + get_fraction(ur, lr, level),
            y: c as f64,
        }
    }
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
) -> (Option<(Point, Point)>, Option<(Point, Point)>) {
    let square_case = 1 * u8::from(ul > level) as u8
        | 2 * (ur > level) as u8
        | 4 * (ll > level) as u8
        | 8 * (lr > level) as u8;
    match square_case {
        1 => (
            Some((
                Point::top(r0, c0, ul, ur, level),
                Point::left(r0, c0, ul, ll, level),
            )),
            None,
        ),
        2 => (
            Some((
                Point::right(r0, c1, ur, lr, level),
                Point::top(r0, c0, ul, ur, level),
            )),
            None,
        ),
        3 => (
            Some((
                Point::right(r0, c1, ur, lr, level),
                Point::left(r0, c0, ul, ll, level),
            )),
            None,
        ),
        4 => (
            Some((
                Point::left(r0, c0, ul, ll, level),
                Point::bottom(r1, c0, ll, lr, level),
            )),
            None,
        ),
        5 => (
            Some((
                Point::top(r0, c0, ul, ur, level),
                Point::bottom(r1, c0, ll, lr, level),
            )),
            None,
        ),
        6 => match vertex_connect_high {
            true => (
                Some((
                    Point::left(r0, c0, ul, ll, level),
                    Point::top(r0, c0, ul, ur, level),
                )),
                Some((
                    Point::right(r0, c1, ur, lr, level),
                    Point::bottom(r1, c0, ll, lr, level),
                )),
            ),
            false => (
                Some((
                    Point::right(r0, c1, ur, lr, level),
                    Point::top(r0, c0, ul, ur, level),
                )),
                Some((
                    Point::left(r0, c0, ul, ll, level),
                    Point::bottom(r1, c0, ll, lr, level),
                )),
            ),
        },
        7 => (
            Some((
                Point::right(r0, c1, ur, lr, level),
                Point::bottom(r1, c0, ll, lr, level),
            )),
            None,
        ),
        8 => (
            Some((
                Point::bottom(r1, c0, ll, lr, level),
                Point::right(r0, c1, ur, lr, level),
            )),
            None,
        ),
        9 => match vertex_connect_high {
            true => (
                Some((
                    Point::top(r0, c0, ul, ur, level),
                    Point::right(r0, c1, ur, lr, level),
                )),
                Some((
                    Point::bottom(r1, c0, ll, lr, level),
                    Point::left(r0, c0, ul, ll, level),
                )),
            ),
            false => (
                Some((
                    Point::top(r0, c0, ul, ur, level),
                    Point::left(r0, c0, ul, ll, level),
                )),
                Some((
                    Point::bottom(r1, c0, ll, lr, level),
                    Point::right(r0, c1, ur, lr, level),
                )),
            ),
        },
        10 => (
            Some((
                Point::bottom(r1, c0, ll, lr, level),
                Point::top(r0, c0, ul, ur, level),
            )),
            None,
        ),
        11 => (
            Some((
                Point::bottom(r1, c0, ll, lr, level),
                Point::left(r0, c0, ul, ll, level),
            )),
            None,
        ),
        12 => (
            Some((
                Point::left(r0, c0, ul, ll, level),
                Point::right(r0, c1, ur, lr, level),
            )),
            None,
        ),
        13 => (
            Some((
                Point::top(r0, c0, ul, ur, level),
                Point::right(r0, c1, ur, lr, level),
            )),
            None,
        ),
        14 => (
            Some((
                Point::left(r0, c0, ul, ll, level),
                Point::top(r0, c0, ul, ur, level),
            )),
            None,
        ),
        0 | 15 => {
            // No segments pass through the square
            (None, None)
        }
        other_case => unreachable!("Unexpected case: {}", other_case),
    }
}

#[pyfunction]
#[pyo3(signature=(array, shape, level, vertex_connect_high=false, mask=None))]
fn get_contour_segments(
    array: Vec<f64>,
    shape: (usize, usize),
    level: f64,
    vertex_connect_high: bool,
    mask: Option<Vec<bool>>,
) -> PyResult<Vec<(Point, Point)>> {
    let grid_mask =
        &to_grid_mask(mask.as_ref(), &array, shape.0, shape.1).map_err(PyValueError::new_err)?;
    match _get_contour_segments(
        &array,
        shape.0,
        shape.1,
        level,
        vertex_connect_high,
        grid_mask,
    ) {
        Ok((segments, _)) => Ok(segments),
        Err(msg) => Err(PyValueError::new_err(msg)),
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
) -> Result<(Vec<(Point, Point)>, Vec<Option<usize>>), String> {
    if array.len() != nb_rows * nb_cols {
        return Err(format!(
            "The array and the given shape are incompatible: {array_len:?} != ({nb_rows} * {nb_cols})",
            array_len = array.len(),
        ));
    }
    let mut segments: Vec<(Point, Point)> = Vec::with_capacity(2 * nb_rows * nb_cols);
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
                    let (seg_1, seg_2) =
                        marching_square(r0, c0, r1, c1, ul, ur, ll, lr, level, vertex_connect_high);
                    // Segment 1
                    if let Some(segment_1) = seg_1 {
                        indices[current_index] = Some(segments.len());
                        segments.push(segment_1);
                    }
                    current_index += 1;
                    // Segment 2
                    if let Some(segment_2) = seg_2 {
                        indices[current_index] = Some(segments.len());
                        segments.push(segment_2);
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
    segments: &Vec<(Point, Point)>,
    indices: &Vec<Option<usize>>,
    nb_cols: usize,
    tol: f64,
) -> Vec<Vec<Point>> {
    let mut contours = Vec::with_capacity(segments.len());
    let mut visited = vec![false; segments.len()];
    let mut neighbors = Vec::with_capacity(segments.len());
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
    for first_index in 0..segments.len() {
        if visited[first_index] {
            continue;
        }
        let mut contour = Vec::new();
        let mut tail_index = first_index;
        let mut head_index = first_index;
        visited[first_index] = true;
        if let Some(segment) = segments.get(first_index) {
            contour.push(segment.0.clone());
            contour.push(segment.1.clone());
        }
        let mut nb_points = 0;
        while contour.len() > nb_points {
            nb_points = contour.len();
            match (
                find_next_segment(&segments, &visited, &neighbors, head_index, tol),
                find_previous_segment(&segments, &visited, &neighbors, tail_index, tol),
            ) {
                (Some(next_index), None) => {
                    if let Some(next_segment) = segments.get(next_index) {
                        contour.push(next_segment.1.clone());
                        head_index = next_index;
                        visited[next_index] = true;
                    } else {
                        unreachable!("The returned index should be an available segment index.");
                    }
                }
                (None, Some(prev_index)) => {
                    if let Some(prev_segment) = segments.get(prev_index) {
                        contour.insert(0, prev_segment.0.clone());
                        tail_index = prev_index;
                        visited[prev_index] = true;
                    } else {
                        unreachable!("The returned index should be an available segment index.");
                    }
                }
                (Some(next_index), Some(prev_index)) => {
                    if next_index <= prev_index {
                        if let Some(next_segment) = segments.get(next_index) {
                            contour.push(next_segment.1.clone());
                            head_index = next_index;
                            visited[next_index] = true;
                        } else {
                            unreachable!(
                                "The returned index should be an available segment index."
                            );
                        }
                    } else {
                        if let Some(prev_segment) = segments.get(prev_index) {
                            contour.insert(0, prev_segment.0.clone());
                            tail_index = prev_index;
                            visited[prev_index] = true;
                        } else {
                            unreachable!(
                                "The returned index should be an available segment index."
                            );
                        }
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
    segments: &Vec<(Point, Point)>,
    visited: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    if let (Some(segment), Some(neighbors_index)) = (segments.get(index), neighbors.get(index)) {
        for &next_index in neighbors_index {
            if !visited[next_index] {
                if let Some(next_segment) = segments.get(next_index) {
                    if segment.1.close(&next_segment.0, tol) {
                        return Some(next_index);
                    }
                }
            }
        }
    }
    None
}

#[inline]
fn find_previous_segment(
    segments: &Vec<(Point, Point)>,
    visited: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    if let (Some(segment), Some(neighbors_index)) = (segments.get(index), neighbors.get(index)) {
        for &prev_index in neighbors_index {
            if !visited[prev_index] {
                if let Some(prev_segment) = segments.get(prev_index) {
                    if prev_segment.1.close(&segment.0, tol) {
                        return Some(prev_index);
                    }
                }
            }
        }
    }
    None
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
) -> PyResult<Vec<Vec<Point>>> {
    let (nb_rows, nb_cols) = shape;
    let grid_mask =
        to_grid_mask(mask.as_ref(), &array, nb_rows, nb_cols).map_err(PyValueError::new_err)?;
    _get_contour_segments(
        &array,
        shape.0,
        shape.1,
        level,
        is_fully_connected,
        &grid_mask,
    )
    .map_err(PyValueError::new_err)
    .map(|(segments, indices)| {
        debug_assert_eq!(indices.len(), 2 * nb_rows * nb_cols);
        assemble_contours(&segments, &indices, nb_cols, tol)
    })
}

// Marching squares algorithm
#[pymodule]
fn marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
