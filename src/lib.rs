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

#[pyclass]
#[derive(Debug, PartialEq, Clone)]
struct Segment {
    #[pyo3(get, set)]
    p1: Point,
    #[pyo3(get, set)]
    p2: Point,
}

#[pymethods]
impl Segment {
    #[staticmethod]
    fn new(p1: Point, p2: Point) -> Segment {
        Segment { p1, p2 }
    }

    fn close(&self, other: &Segment, tol: f64) -> bool {
        self.p1.close(&other.p1, tol) && self.p2.close(&other.p2, tol)
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn get_contour_segments(
    array: Vec<f64>,
    nb_cols: usize,
    level: f64,
    vertex_connect_high: bool,
    mask: Vec<bool>,
) -> Vec<Segment> {
    assert!(
        nb_cols > 1,
        "Can't find the contour segments of one column."
    );
    let use_mask: bool = mask.len() > 0;
    if use_mask {
        assert!(
            array.len() == mask.len(),
            "The array and the mask are not compatible"
        );
    }
    let nb_rows = array.len() / nb_cols;
    assert!(
        (nb_rows * nb_cols) == array.len(),
        "The array has invalid dimension"
    );
    let mut segments: Vec<Segment> = Vec::with_capacity(array.len());
    for r0 in 0..(nb_rows - 1) {
        for c0 in 0..(nb_cols - 1) {
            let (r1, c1) = (r0 + 1, c0 + 1);
            let index_ul = r0 * nb_cols + c0;
            let index_ur = r0 * nb_cols + c1;
            let index_ll = r1 * nb_cols + c0;
            let index_lr = r1 * nb_cols + c1;
            if use_mask && !(mask[index_ul] && mask[index_ur] && mask[index_ll] && mask[index_lr]) {
                continue;
            }
            let (ul, ur, ll, lr) = (
                array[index_ul],
                array[index_ur],
                array[index_ll],
                array[index_lr],
            );
            let square_case = 1 * (ul > level) as u8
                | 2 * (ur > level) as u8
                | 4 * (ll > level) as u8
                | 8 * (lr > level) as u8;
            // Compute intersection points
            match square_case {
                1 => segments.push(Segment {
                    p1: Point::top(r0, c0, ul, ur, level),
                    p2: Point::left(r0, c0, ul, ll, level),
                }),
                2 => segments.push(Segment {
                    p1: Point::right(r0, c1, ur, lr, level),
                    p2: Point::top(r0, c0, ul, ur, level),
                }),
                3 => segments.push(Segment {
                    p1: Point::right(r0, c1, ur, lr, level),
                    p2: Point::left(r0, c0, ul, ll, level),
                }),
                4 => segments.push(Segment {
                    p1: Point::left(r0, c0, ul, ll, level),
                    p2: Point::bottom(r1, c0, ll, lr, level),
                }),
                5 => segments.push(Segment {
                    p1: Point::top(r0, c0, ul, ur, level),
                    p2: Point::bottom(r1, c0, ll, lr, level),
                }),
                6 => {
                    if vertex_connect_high {
                        segments.push(Segment {
                            p1: Point::left(r0, c0, ul, ll, level),
                            p2: Point::top(r0, c0, ul, ur, level),
                        });
                        segments.push(Segment {
                            p1: Point::right(r0, c1, ur, lr, level),
                            p2: Point::bottom(r1, c0, ll, lr, level),
                        });
                    } else {
                        segments.push(Segment {
                            p1: Point::right(r0, c1, ur, lr, level),
                            p2: Point::top(r0, c0, ul, ur, level),
                        });
                        segments.push(Segment {
                            p1: Point::left(r0, c0, ul, ll, level),
                            p2: Point::bottom(r1, c0, ll, lr, level),
                        });
                    }
                }
                7 => segments.push(Segment {
                    p1: Point::right(r0, c1, ur, lr, level),
                    p2: Point::bottom(r1, c0, ll, lr, level),
                }),
                8 => segments.push(Segment {
                    p1: Point::bottom(r1, c0, ll, lr, level),
                    p2: Point::right(r0, c1, ur, lr, level),
                }),
                9 => {
                    if vertex_connect_high {
                        segments.push(Segment {
                            p1: Point::top(r0, c0, ul, ur, level),
                            p2: Point::right(r0, c1, ur, lr, level),
                        });
                        segments.push(Segment {
                            p1: Point::bottom(r1, c0, ll, lr, level),
                            p2: Point::left(r0, c0, ul, ll, level),
                        });
                    } else {
                        segments.push(Segment {
                            p1: Point::top(r0, c0, ul, ur, level),
                            p2: Point::left(r0, c0, ul, ll, level),
                        });
                        segments.push(Segment {
                            p1: Point::bottom(r1, c0, ll, lr, level),
                            p2: Point::right(r0, c1, ur, lr, level),
                        });
                    }
                }
                10 => segments.push(Segment {
                    p1: Point::bottom(r1, c0, ll, lr, level),
                    p2: Point::top(r0, c0, ul, ur, level),
                }),
                11 => segments.push(Segment {
                    p1: Point::bottom(r1, c0, ll, lr, level),
                    p2: Point::left(r0, c0, ul, ll, level),
                }),
                12 => segments.push(Segment {
                    p1: Point::left(r0, c0, ul, ll, level),
                    p2: Point::right(r0, c1, ur, lr, level),
                }),
                13 => segments.push(Segment {
                    p1: Point::top(r0, c0, ul, ur, level),
                    p2: Point::right(r0, c1, ur, lr, level),
                }),
                14 => segments.push(Segment {
                    p1: Point::left(r0, c0, ul, ll, level),
                    p2: Point::top(r0, c0, ul, ur, level),
                }),
                0 | 15 => continue, // No segments pass through the square
                _ => panic!("Unexpected value"),
            }
        }
    }
    return segments;
}

/// A Python module implemented in Rust.
#[pymodule]
fn marchingsquares(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_function(wrap_pyfunction!(get_contour_segments, m)?)?;
    Ok(())
}
