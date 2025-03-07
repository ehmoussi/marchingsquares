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
#[pyo3(signature=(array, level, vertex_connect_high=false, mask=None))]
fn get_contour_segments(
    array: Vec<Vec<f64>>,
    level: f64,
    vertex_connect_high: bool,
    mask: Option<Vec<Vec<bool>>>,
) -> PyResult<Vec<(Point, Point)>> {
    match _get_contour_segments(&array, level, vertex_connect_high, mask.as_ref()) {
        Ok(segments) => Ok(segments.into_iter().filter_map(|s| s).collect()),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

fn _get_contour_segments(
    array: &Vec<Vec<f64>>,
    level: f64,
    vertex_connect_high: bool,
    mask: Option<&Vec<Vec<bool>>>,
) -> Result<Vec<Option<(Point, Point)>>, String> {
    let nb_rows = array.len();
    let nb_cols = match array.get(0) {
        Some(first_row) => first_row.len(),
        None => 0,
    };
    if let Some(m) = mask {
        let mask_nb_cols = match m.get(0) {
            Some(mask_row) => mask_row.len(),
            None => 0,
        };
        let array_shape = (nb_rows, nb_cols);
        let mask_shape = (m.len(), mask_nb_cols);
        if array_shape != mask_shape {
            return Err(format!(
            "The array and the mask must have the same shape, {array_shape:?} != {mask_shape:?}",
        ));
        }
    }
    let mut segments: Vec<Option<(Point, Point)>> = Vec::with_capacity(2 * nb_rows * nb_cols);
    for r0 in 0..(array.len() - 1) {
        let current_row = array
            .get(r0)
            .expect("The iterator should be bound by the length of the array");
        if current_row.len() != nb_cols {
            return Err(format!(
                "The array don't have the same number of columns. The row {r0} has {current_row_nb_cols} instead of {nb_cols} columns",
                current_row_nb_cols=current_row.len(),
            ));
        }
        let r1 = r0 + 1;
        let next_row = array
            .get(r1)
            .expect("The iterator should end one row before the end");
        if next_row.len() != nb_cols {
            return Err(format!(
                "The array don't have the same number of columns. The row {r1} has {next_row_nb_cols} instead of {nb_cols} columns",
                next_row_nb_cols=next_row.len(),
            ));
        }
        let (current_row_mask, next_row_mask) = match mask {
            Some(m) => (
                Some(
                    m.get(r0)
                        .expect("The mask should have the same number of rows as the array"),
                ),
                Some(
                    m.get(r1)
                        .expect("The mask should have the same number of rows as the array"),
                ),
            ),
            None => (None, None),
        };
        for c0 in 0..(nb_cols - 1) {
            let c1 = c0 + 1;
            let ul = current_row
                .get(c0)
                .expect("The iterator should be bound by the number of columns of the array");
            let ll = next_row
                .get(c0)
                .expect("The iterator should be bound by the number of columns of the array");
            let ur = current_row
                .get(c1)
                .expect("The iterator should end one column before the end");
            let lr = next_row
                .get(c1)
                .expect("The iterator should end one column before the end");
            if let (Some(cr_mask), Some(nr_mask)) = (current_row_mask, next_row_mask) {
                let mask_ul = cr_mask
                    .get(c0)
                    .expect("The iterator should be bound by the number of columns of the array");
                let mask_ll = nr_mask
                    .get(c0)
                    .expect("The iterator should be bound by the number of columns of the array");
                let mask_ur = cr_mask
                    .get(c1)
                    .expect("The iterator should end one column before the end");
                let mask_lr = nr_mask
                    .get(c1)
                    .expect("The iterator should end one column before the end");
                if !(*mask_ul && *mask_ll && *mask_ur && *mask_lr) {
                    segments.push(None);
                    segments.push(None);
                    continue;
                }
            }
            let (seg_1, seg_2) = marching_square(
                r0,
                c0,
                r1,
                c1,
                *ul,
                *ur,
                *ll,
                *lr,
                level,
                vertex_connect_high,
            );
            segments.push(seg_1);
            segments.push(seg_2);
        }
        segments.push(None);
        segments.push(None);
    }
    for _ in 0..nb_cols {
        segments.push(None);
        segments.push(None);
    }
    Ok(segments)
}

fn assemble_contours(
    segments: &Vec<Option<(Point, Point)>>,
    nb_cols: usize,
    tol: f64,
) -> Vec<Vec<Point>> {
    let mut contours = Vec::with_capacity(segments.len());
    let mut visited = vec![false; segments.len()];
    let mut neighbors = Vec::with_capacity(segments.len());
    for index in 0..segments.len() {
        let mut neighbors_index = Vec::with_capacity(8);
        for (i, j) in [(-1, 0), (0, -1), (0, 1), (1, 0)].iter() {
            for k in 0..2 {
                if let Some(neighbor_index) = index.checked_add_signed(
                    i * 2 * (nb_cols as isize) + 2 * j + k - (index as isize % 2),
                ) {
                    neighbors_index.push(neighbor_index);
                }
            }
        }
        neighbors.push(neighbors_index);
    }
    for first_index in 0..segments.len() {
        if segments[first_index].is_none() {
            visited[first_index] = true;
            continue;
        } else if visited[first_index] {
            continue;
        }
        let mut contour = Vec::new();
        let mut tail_index = first_index;
        let mut head_index = first_index;
        visited[first_index] = true;
        if let Some(seg) = segments.get(first_index) {
            if let Some(segment) = seg {
                contour.push(segment.0.clone());
                contour.push(segment.1.clone());
            }
        }
        let mut nb_points = 0;
        while contour.len() > nb_points {
            nb_points = contour.len();
            match (
                find_next_segment(&segments, &visited, &neighbors, head_index, tol),
                find_previous_segment(&segments, &visited, &neighbors, tail_index, tol),
            ) {
                (Some(next_index), None) => {
                    if let Some(next_seg) = segments.get(next_index) {
                        if let Some(next_segment) = next_seg {
                            contour.push(next_segment.1.clone());
                            head_index = next_index;
                            visited[next_index] = true;
                        } else {
                            unreachable!(
                                "The returned index should be an available segment index."
                            );
                        }
                    } else {
                        unreachable!("The returned index should be an available segment index.");
                    }
                }
                (None, Some(prev_index)) => {
                    if let Some(prev_seg) = segments.get(prev_index) {
                        if let Some(prev_segment) = prev_seg {
                            contour.insert(0, prev_segment.0.clone());
                            tail_index = prev_index;
                            visited[prev_index] = true;
                        } else {
                            unreachable!(
                                "The returned index should be an available segment index."
                            );
                        }
                    } else {
                        unreachable!("The returned index should be an available segment index.");
                    }
                }
                (Some(next_index), Some(prev_index)) => {
                    if next_index <= prev_index {
                        if let Some(next_seg) = segments.get(next_index) {
                            if let Some(next_segment) = next_seg {
                                contour.push(next_segment.1.clone());
                                head_index = next_index;
                                visited[next_index] = true;
                            } else {
                                unreachable!(
                                    "The returned index should be an available segment index."
                                );
                            }
                        } else {
                            unreachable!(
                                "The returned index should be an available segment index."
                            );
                        }
                    } else {
                        if let Some(prev_seg) = segments.get(prev_index) {
                            if let Some(prev_segment) = prev_seg {
                                contour.insert(0, prev_segment.0.clone());
                                tail_index = prev_index;
                                visited[prev_index] = true;
                            } else {
                                unreachable!(
                                    "The returned index should be an available segment index."
                                );
                            }
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
    segments: &Vec<Option<(Point, Point)>>,
    visited: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    if let (Some(seg), Some(neighbors_index)) = (segments.get(index), neighbors.get(index)) {
        if let Some(segment) = seg {
            for &next_index in neighbors_index {
                if !visited[next_index] {
                    if let Some(next_seg) = segments.get(next_index) {
                        if let Some(next_segment) = next_seg {
                            if segment.1.close(&next_segment.0, tol) {
                                return Some(next_index);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[inline]
fn find_previous_segment(
    segments: &Vec<Option<(Point, Point)>>,
    visited: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    index: usize,
    tol: f64,
) -> Option<usize> {
    if let (Some(seg), Some(neighbors_index)) = (segments.get(index), neighbors.get(index)) {
        if let Some(segment) = seg {
            for &prev_index in neighbors_index {
                if !visited[prev_index] {
                    if let Some(prev_seg) = segments.get(prev_index) {
                        if let Some(prev_segment) = prev_seg {
                            if prev_segment.1.close(&segment.0, tol) {
                                return Some(prev_index);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[pyfunction]
#[pyo3(signature=(array, level, is_fully_connected=false, mask=None, tol=1e-10))]
fn marching_squares(
    array: Vec<Vec<f64>>,
    level: f64,
    is_fully_connected: bool,
    mask: Option<Vec<Vec<bool>>>,
    tol: f64,
) -> PyResult<Vec<Vec<Point>>> {
    let nb_rows = array.len();
    let nb_cols = match array.get(0) {
        Some(a) => a.len(),
        None => 0,
    };
    match _get_contour_segments(&array, level, is_fully_connected, mask.as_ref()) {
        Ok(segments) => {
            debug_assert_eq!(segments.len(), 2 * nb_rows * nb_cols);
            Ok(assemble_contours(&segments, nb_cols, tol))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

// Marching squares algorithm
#[pymodule]
fn marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
