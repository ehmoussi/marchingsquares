use std::collections::VecDeque;

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
    fn top(r: usize, c: usize, ul: f64, ur: f64, level: f64) -> Point {
        Point {
            x: r as f64,
            y: c as f64 + get_fraction(ul, ur, level),
        }
    }

    #[staticmethod]
    fn bottom(r: usize, c: usize, ll: f64, lr: f64, level: f64) -> Point {
        Point {
            x: r as f64,
            y: c as f64 + get_fraction(ll, lr, level),
        }
    }

    #[staticmethod]
    fn left(r: usize, c: usize, ul: f64, ll: f64, level: f64) -> Point {
        Point {
            x: r as f64 + get_fraction(ul, ll, level),
            y: c as f64,
        }
    }

    #[staticmethod]
    fn right(r: usize, c: usize, ur: f64, lr: f64, level: f64) -> Point {
        Point {
            x: r as f64 + get_fraction(ur, lr, level),
            y: c as f64,
        }
    }
}

#[pyfunction]
#[pyo3(signature=(array, nb_cols, level, vertex_connect_high=false, mask=None))]
fn get_contour_segments(
    array: Vec<f64>,
    nb_cols: usize,
    level: f64,
    vertex_connect_high: bool,
    mask: Option<Vec<bool>>,
) -> Vec<(Point, Point)> {
    assert!(
        nb_cols > 1,
        "Can't find the contour segments of one column."
    );
    if let Some(m) = mask.as_ref() {
        assert!(
            array.len() == m.len(),
            "The array and the mask are not compatible"
        );
    }
    let nb_rows = array.len() / nb_cols;
    assert!(
        (nb_rows * nb_cols) == array.len(),
        "The array has invalid dimension"
    );
    let mut segments: Vec<(Point, Point)> = Vec::with_capacity(array.len());
    for r0 in 0..(nb_rows - 1) {
        for c0 in 0..(nb_cols - 1) {
            let (r1, c1) = (r0 + 1, c0 + 1);
            let index_ul = r0 * nb_cols + c0;
            let index_ur = r0 * nb_cols + c1;
            let index_ll = r1 * nb_cols + c0;
            let index_lr = r1 * nb_cols + c1;
            if let Some(m) = mask.as_ref() {
                if let (Some(&m_ul), Some(&m_ur), Some(&m_ll), Some(&m_lr)) = (
                    m.get(index_ul),
                    m.get(index_ur),
                    m.get(index_ll),
                    m.get(index_lr),
                ) {
                    if !(m_ul && m_ur && m_ll && m_lr) {
                        continue;
                    }
                }
            }
            if let (Some(&ul), Some(&ur), Some(&ll), Some(&lr)) = (
                array.get(index_ul),
                array.get(index_ur),
                array.get(index_ll),
                array.get(index_lr),
            ) {
                let square_case = 1 * (ul > level) as u8
                    | 2 * (ur > level) as u8
                    | 4 * (ll > level) as u8
                    | 8 * (lr > level) as u8;
                // Compute intersection points
                match square_case {
                    1 => segments.push((
                        Point::top(r0, c0, ul, ur, level),
                        Point::left(r0, c0, ul, ll, level),
                    )),
                    2 => segments.push((
                        Point::right(r0, c1, ur, lr, level),
                        Point::top(r0, c0, ul, ur, level),
                    )),
                    3 => segments.push((
                        Point::right(r0, c1, ur, lr, level),
                        Point::left(r0, c0, ul, ll, level),
                    )),
                    4 => segments.push((
                        Point::left(r0, c0, ul, ll, level),
                        Point::bottom(r1, c0, ll, lr, level),
                    )),
                    5 => segments.push((
                        Point::top(r0, c0, ul, ur, level),
                        Point::bottom(r1, c0, ll, lr, level),
                    )),
                    6 => {
                        if vertex_connect_high {
                            segments.push((
                                Point::left(r0, c0, ul, ll, level),
                                Point::top(r0, c0, ul, ur, level),
                            ));
                            segments.push((
                                Point::right(r0, c1, ur, lr, level),
                                Point::bottom(r1, c0, ll, lr, level),
                            ));
                        } else {
                            segments.push((
                                Point::right(r0, c1, ur, lr, level),
                                Point::top(r0, c0, ul, ur, level),
                            ));
                            segments.push((
                                Point::left(r0, c0, ul, ll, level),
                                Point::bottom(r1, c0, ll, lr, level),
                            ));
                        }
                    }
                    7 => segments.push((
                        Point::right(r0, c1, ur, lr, level),
                        Point::bottom(r1, c0, ll, lr, level),
                    )),
                    8 => segments.push((
                        Point::bottom(r1, c0, ll, lr, level),
                        Point::right(r0, c1, ur, lr, level),
                    )),
                    9 => {
                        if vertex_connect_high {
                            segments.push((
                                Point::top(r0, c0, ul, ur, level),
                                Point::right(r0, c1, ur, lr, level),
                            ));
                            segments.push((
                                Point::bottom(r1, c0, ll, lr, level),
                                Point::left(r0, c0, ul, ll, level),
                            ));
                        } else {
                            segments.push((
                                Point::top(r0, c0, ul, ur, level),
                                Point::left(r0, c0, ul, ll, level),
                            ));
                            segments.push((
                                Point::bottom(r1, c0, ll, lr, level),
                                Point::right(r0, c1, ur, lr, level),
                            ));
                        }
                    }
                    10 => segments.push((
                        Point::bottom(r1, c0, ll, lr, level),
                        Point::top(r0, c0, ul, ur, level),
                    )),
                    11 => segments.push((
                        Point::bottom(r1, c0, ll, lr, level),
                        Point::left(r0, c0, ul, ll, level),
                    )),
                    12 => segments.push((
                        Point::left(r0, c0, ul, ll, level),
                        Point::right(r0, c1, ur, lr, level),
                    )),
                    13 => segments.push((
                        Point::top(r0, c0, ul, ur, level),
                        Point::right(r0, c1, ur, lr, level),
                    )),
                    14 => segments.push((
                        Point::left(r0, c0, ul, ll, level),
                        Point::top(r0, c0, ul, ur, level),
                    )),
                    0 | 15 => continue, // No segments pass through the square
                    other_case => panic!("Unexpected case: {}", other_case),
                }
            }
        }
    }
    return segments;
}

fn assemble_contours(segments: &Vec<(Point, Point)>, tol: f64) -> Vec<Vec<Point>> {
    let mut contours = Vec::with_capacity(segments.len());
    let mut queue = VecDeque::with_capacity(segments.len());
    queue.extend(segments);
    let mut not_used = Vec::with_capacity(segments.len());
    while not_used.len() > 0 || queue.len() > 0 {
        queue.extend(not_used.iter());
        not_used.clear();
        let mut contour = VecDeque::with_capacity(queue.len());
        while queue.len() > 0 {
            let segment = queue.pop_front().unwrap();
            if contour.len() == 0 {
                contour.push_back(segment.0.clone());
                contour.push_back(segment.1.clone());
            } else if contour
                .iter()
                .last()
                .is_some_and(|p: &Point| p.close(&segment.0, tol))
            {
                contour.push_back(segment.1.clone());
                for &segment_not_used in not_used.iter().rev() {
                    queue.push_front(segment_not_used);
                }
                not_used.clear();
            } else if contour[0].close(&segment.1, tol) {
                contour.push_front(segment.0.clone());
                for &segment_not_used in not_used.iter().rev() {
                    queue.push_front(segment_not_used);
                }
                not_used.clear();
            } else {
                not_used.push(segment);
            }
        }
        contours.push(contour.into());
    }
    return contours;
}

#[pyfunction]
#[pyo3(signature=(array, nb_cols, level, is_fully_connected=false, mask=None, tol=1e-10))]
fn marching_squares(
    array: Vec<f64>,
    nb_cols: usize,
    level: f64,
    is_fully_connected: bool,
    mask: Option<Vec<bool>>,
    tol: f64,
) -> Vec<Vec<Point>> {
    let segments = get_contour_segments(array, nb_cols, level, is_fully_connected, mask);
    return assemble_contours(&segments, tol);
}

/// A Python module implemented in Rust.
#[pymodule]
fn marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    m.add_function(wrap_pyfunction!(marching_squares, m)?)?;
    Ok(())
}
