use crate::utils::close;
use numpy::{ndarray::ArrayD, IxDyn};

pub fn assemble_contours(
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
