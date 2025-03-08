use pyo3::prelude::*;

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

#[pyfunction]
#[pyo3(signature=(array, level, vertex_connect_high=false, mask=None))]
fn get_contour_segments(
    array: Vec<Vec<f64>>,
    level: f64,
    vertex_connect_high: bool,
    mask: Option<Vec<Vec<bool>>>,
) -> Vec<(Point, Point)> {
    _get_contour_segments(&array, level, vertex_connect_high, mask.as_ref())
        .into_iter()
        .filter_map(|s| s)
        .collect()
}

fn _get_contour_segments(
    array: &Vec<Vec<f64>>,
    level: f64,
    vertex_connect_high: bool,
    mask: Option<&Vec<Vec<bool>>>,
) -> Vec<Option<(Point, Point)>> {
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
        assert_eq!(
            (nb_rows, nb_cols),
            (m.len(), mask_nb_cols),
            "The array and the mask must have the same shape"
        );
    }
    let mut segments: Vec<Option<(Point, Point)>> = Vec::with_capacity(2 * nb_rows * nb_cols);
    for r0 in 0..(array.len() - 1) {
        let current_row = array
            .get(r0)
            .expect("The iterator should be bound by the length of the array");
        assert_eq!(
            current_row.len(),
            nb_cols,
            "The array don't have the same number of columns. The row {} has {} instead of {}",
            r0,
            current_row.len(),
            nb_cols
        );
        let r1 = r0 + 1;
        let next_row = array
            .get(r1)
            .expect("The iterator should end one row before the end");
        assert_eq!(
            next_row.len(),
            nb_cols,
            "The array don't have the same number of columns. The row {} has {} instead of {}",
            r1,
            next_row.len(),
            nb_cols
        );
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
            let square_case = 1 * u8::from(*ul > level) as u8
                | 2 * (*ur > level) as u8
                | 4 * (*ll > level) as u8
                | 8 * (*lr > level) as u8;
            match square_case {
                1 => {
                    segments.push(Some((
                        Point::top(r0, c0, *ul, *ur, level),
                        Point::left(r0, c0, *ul, *ll, level),
                    )));
                    segments.push(None)
                }
                2 => {
                    segments.push(Some((
                        Point::right(r0, c1, *ur, *lr, level),
                        Point::top(r0, c0, *ul, *ur, level),
                    )));
                    segments.push(None)
                }
                3 => {
                    segments.push(Some((
                        Point::right(r0, c1, *ur, *lr, level),
                        Point::left(r0, c0, *ul, *ll, level),
                    )));
                    segments.push(None)
                }
                4 => {
                    segments.push(Some((
                        Point::left(r0, c0, *ul, *ll, level),
                        Point::bottom(r1, c0, *ll, *lr, level),
                    )));
                    segments.push(None)
                }
                5 => {
                    segments.push(Some((
                        Point::top(r0, c0, *ul, *ur, level),
                        Point::bottom(r1, c0, *ll, *lr, level),
                    )));
                    segments.push(None)
                }
                6 => {
                    if vertex_connect_high {
                        segments.push(Some((
                            Point::left(r0, c0, *ul, *ll, level),
                            Point::top(r0, c0, *ul, *ur, level),
                        )));
                        segments.push(Some((
                            Point::right(r0, c1, *ur, *lr, level),
                            Point::bottom(r1, c0, *ll, *lr, level),
                        )));
                    } else {
                        segments.push(Some((
                            Point::right(r0, c1, *ur, *lr, level),
                            Point::top(r0, c0, *ul, *ur, level),
                        )));
                        segments.push(Some((
                            Point::left(r0, c0, *ul, *ll, level),
                            Point::bottom(r1, c0, *ll, *lr, level),
                        )));
                    }
                }
                7 => {
                    segments.push(Some((
                        Point::right(r0, c1, *ur, *lr, level),
                        Point::bottom(r1, c0, *ll, *lr, level),
                    )));
                    segments.push(None)
                }
                8 => {
                    segments.push(Some((
                        Point::bottom(r1, c0, *ll, *lr, level),
                        Point::right(r0, c1, *ur, *lr, level),
                    )));
                    segments.push(None)
                }
                9 => {
                    if vertex_connect_high {
                        segments.push(Some((
                            Point::top(r0, c0, *ul, *ur, level),
                            Point::right(r0, c1, *ur, *lr, level),
                        )));
                        segments.push(Some((
                            Point::bottom(r1, c0, *ll, *lr, level),
                            Point::left(r0, c0, *ul, *ll, level),
                        )));
                    } else {
                        segments.push(Some((
                            Point::top(r0, c0, *ul, *ur, level),
                            Point::left(r0, c0, *ul, *ll, level),
                        )));
                        segments.push(Some((
                            Point::bottom(r1, c0, *ll, *lr, level),
                            Point::right(r0, c1, *ur, *lr, level),
                        )));
                    }
                }
                10 => {
                    segments.push(Some((
                        Point::bottom(r1, c0, *ll, *lr, level),
                        Point::top(r0, c0, *ul, *ur, level),
                    )));
                    segments.push(None)
                }
                11 => {
                    segments.push(Some((
                        Point::bottom(r1, c0, *ll, *lr, level),
                        Point::left(r0, c0, *ul, *ll, level),
                    )));
                    segments.push(None)
                }
                12 => {
                    segments.push(Some((
                        Point::left(r0, c0, *ul, *ll, level),
                        Point::right(r0, c1, *ur, *lr, level),
                    )));
                    segments.push(None)
                }
                13 => {
                    segments.push(Some((
                        Point::top(r0, c0, *ul, *ur, level),
                        Point::right(r0, c1, *ur, *lr, level),
                    )));
                    segments.push(None)
                }
                14 => {
                    segments.push(Some((
                        Point::left(r0, c0, *ul, *ll, level),
                        Point::top(r0, c0, *ul, *ur, level),
                    )));
                    segments.push(None)
                }
                0 | 15 => {
                    // No segments pass through the square
                    segments.push(None);
                    segments.push(None);
                }
                other_case => unreachable!("Unexpected case: {}", other_case),
            }
        }
        segments.push(None);
        segments.push(None);
    }
    for _ in 0..nb_cols {
        segments.push(None);
        segments.push(None);
    }
    return segments;
}

fn assemble_contours(
    segments: &mut Vec<Option<(Point, Point)>>,
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
) -> Vec<Vec<Point>> {
    let nb_rows = array.len();
    let nb_cols = match array.get(0) {
        Some(a) => a.len(),
        None => 0,
    };
    let mut segments = _get_contour_segments(&array, level, is_fully_connected, mask.as_ref());
    assert_eq!(segments.len(), 2 * nb_rows * nb_cols);
    return assemble_contours(&mut segments, nb_cols, tol);
}

/// A Python module implemented in Rust.
#[pymodule]
fn marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
