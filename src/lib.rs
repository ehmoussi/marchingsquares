mod contours;
mod neighbors;
mod segments;
mod utils;
use numpy::ndarray::{Array, ArrayD};
use numpy::{IntoPyArray, IxDyn, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature=(p1_x, p1_y, p2_x, p2_y, tol=1e-10))]
#[inline]
fn close(p1_x: f64, p1_y: f64, p2_x: f64, p2_y: f64, tol: f64) -> bool {
    utils::close(p1_x, p1_y, p2_x, p2_y, tol)
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
            segments::find_segments(&array, level, vertex_connect_high, &mask).0
        }
        None => {
            let mask = Array::from_shape_vec(array.shape(), vec![1u8; array.len()]).unwrap();
            let mask = mask.view().into_dyn();
            segments::find_segments(&array, level, vertex_connect_high, &mask).0
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
            segments::find_segments(&array, level, is_fully_connected, &mask)
        }
        None => {
            let mask = Array::from_shape_vec(array.shape(), vec![1u8; array.len()]).unwrap();
            let mask = mask.view().into_dyn();
            segments::find_segments(&array, level, is_fully_connected, &mask)
        }
    };
    let shape = array.shape();
    let (nb_rows, nb_cols) = (shape[0], shape[1]);
    let (head_neighbors, tail_neighbors) = neighbors::build_neighbors(
        &square_cases,
        &segments,
        nb_rows,
        nb_cols,
        is_fully_connected,
    );
    let contours = contours::assemble_contours(&segments, &head_neighbors, &tail_neighbors, tol);
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
