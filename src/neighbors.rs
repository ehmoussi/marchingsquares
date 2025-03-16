pub fn build_neighbors(
    square_cases: &Vec<u8>,
    _segments: &Vec<f64>,
    _nb_rows: usize,
    nb_cols: usize,
    vertex_connect_high: bool,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut segment_positions = vec![0; square_cases.len() + 1];
    for (i, &square_case) in square_cases.iter().enumerate() {
        let nb_segments = match square_case {
            1 | 2 | 3 | 4 | 5 | 7 | 8 | 10 | 11 | 12 | 13 | 14 => 1,
            6 | 9 => 2,
            0 | 15 => 0,
            other_case => unreachable!("Unexpected square case: {}", other_case),
        };
        segment_positions[i + 1] = segment_positions[i] + nb_segments;
    }
    let mut head_neighbors = Vec::with_capacity(_segments.len() / 4);
    let mut tail_neighbors = Vec::with_capacity(_segments.len() / 4);
    let nb_cols_m1 = nb_cols - 1;
    for (square_index, square_case) in square_cases.iter().enumerate() {
        let r0 = square_index / nb_cols_m1;
        let c0 = square_index % nb_cols_m1;
        // let diff = head_neighbors.len();
        match square_case {
            0 | 15 => (),
            1 => {
                add_top_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_left_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    true,
                );
            }
            2 => {
                add_right_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    false,
                );
                add_top_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            3 => {
                add_right_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    false,
                );
                add_left_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    true,
                );
            }
            4 => {
                add_left_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    false,
                );
                add_bottom_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            5 => {
                add_top_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_bottom_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            6 => match vertex_connect_high {
                true => {
                    add_left_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        false,
                    );
                    add_top_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                    // seg 2
                    add_right_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        false,
                    );
                    add_bottom_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                }
                false => {
                    add_right_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        false,
                    );
                    add_top_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                    // seg 2
                    add_left_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        false,
                    );
                    add_bottom_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        true,
                    );
                }
            },
            7 => {
                add_right_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    false,
                );
                add_bottom_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            8 => {
                add_bottom_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_right_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    true,
                );
            }
            9 => match vertex_connect_high {
                true => {
                    add_top_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_right_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        true,
                    );
                    // seg 2
                    add_bottom_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_left_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        true,
                    );
                }
                false => {
                    add_top_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_left_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        true,
                    );
                    // seg 2
                    add_bottom_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        false,
                    );
                    add_right_neighbor(
                        r0,
                        c0,
                        nb_cols_m1,
                        &mut head_neighbors,
                        &mut tail_neighbors,
                        &square_cases,
                        &segment_positions,
                        vertex_connect_high,
                        true,
                    );
                }
            },
            10 => {
                add_bottom_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_top_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
                );
            }
            11 => {
                add_bottom_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_left_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    true,
                );
            }
            12 => {
                add_left_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    false,
                );
                add_right_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    true,
                );
            }
            13 => {
                add_top_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    false,
                );
                add_right_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    true,
                );
            }
            14 => {
                add_left_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    vertex_connect_high,
                    false,
                );
                add_top_neighbor(
                    r0,
                    c0,
                    nb_cols_m1,
                    &mut head_neighbors,
                    &mut tail_neighbors,
                    &square_cases,
                    &segment_positions,
                    true,
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
        // match square_case {
        //     0 | 15 => assert_eq!(head_neighbors.len(), diff, "{square_case}"),
        //     6 | 9 => assert_eq!(head_neighbors.len(), diff + 2, "{square_case}"),
        //     _ => assert_eq!(head_neighbors.len(), diff + 1, "{square_case}"),
        // }
    }
    // assert_eq!(tail_neighbors.len(), _segments.len() / 4);
    // assert_eq!(head_neighbors.len(), _segments.len() / 4);
    (head_neighbors, tail_neighbors)
}
#[inline(always)]
fn add_top_neighbor(
    r0: usize,
    c0: usize,
    nb_cols_m1: usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    is_head: bool,
) {
    if r0 == 0 {
        if is_head {
            head_neighbors.push(None);
        } else {
            tail_neighbors.push(None);
        }
    } else {
        let top_index = (r0 - 1) * nb_cols_m1 + c0;
        let square_case;
        unsafe {
            square_case = square_cases.get_unchecked(top_index);
        }
        match (square_case, is_head) {
            // head
            (8 | 10 | 11, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(top_index);
                }
                head_neighbors.push(Some(*index));
            }
            (9, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(top_index);
                }
                head_neighbors.push(Some(*index + 1));
            }
            (_, true) => {
                head_neighbors.push(None);
            }
            // tail
            (4 | 5 | 7, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(top_index);
                }
                tail_neighbors.push(Some(*index));
            }
            (6, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(top_index);
                }
                tail_neighbors.push(Some(*index + 1));
            }
            (_, false) => {
                tail_neighbors.push(None);
            }
        }
    }
}

#[inline(always)]
fn add_left_neighbor(
    r0: usize,
    c0: usize,
    nb_cols_m1: usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    vertex_connect_high: bool,
    is_head: bool,
) {
    if c0 == 0 {
        if is_head {
            head_neighbors.push(None);
        } else {
            tail_neighbors.push(None);
        }
    } else {
        let left_index = r0 * nb_cols_m1 + (c0 - 1);
        let square_case;
        unsafe {
            square_case = square_cases.get_unchecked(left_index);
        }
        match (square_case, vertex_connect_high, is_head) {
            // head
            (2 | 3 | 7, _, true) | (6, false, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(left_index);
                }
                head_neighbors.push(Some(*index));
            }
            (6, true, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(left_index);
                }
                head_neighbors.push(Some(*index + 1));
            }
            (_, _, true) => {
                head_neighbors.push(None);
            }
            // tail
            (8 | 12 | 13, _, false) | (9, true, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(left_index);
                }
                tail_neighbors.push(Some(*index));
            }
            (9, false, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(left_index);
                }
                tail_neighbors.push(Some(*index + 1));
            }
            (_, _, false) => {
                tail_neighbors.push(None);
            }
        }
    }
}

#[inline(always)]
fn add_right_neighbor(
    r0: usize,
    c0: usize,
    nb_cols_m1: usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    vertex_connect_high: bool,
    is_head: bool,
) {
    let right_index = r0 * nb_cols_m1 + (c0 + 1);
    if right_index >= square_cases.len() {
        if is_head {
            head_neighbors.push(None);
        } else {
            tail_neighbors.push(None);
        }
    } else {
        let square_case;
        unsafe {
            square_case = square_cases.get_unchecked(right_index);
        }
        match (square_case, vertex_connect_high, is_head) {
            // head
            (4 | 12 | 14, _, true) | (6, true, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(right_index);
                }
                head_neighbors.push(Some(*index));
            }
            (6, false, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(right_index);
                }
                head_neighbors.push(Some(*index + 1));
            }
            (_, _, true) => {
                head_neighbors.push(None);
            }
            // tail
            (1 | 3 | 11, _, false) | (9, false, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(right_index);
                }
                tail_neighbors.push(Some(*index));
            }
            (9, true, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(right_index);
                }
                tail_neighbors.push(Some(*index + 1));
            }
            (_, _, false) => {
                tail_neighbors.push(None);
            }
        }
    }
}

#[inline(always)]
fn add_bottom_neighbor(
    r0: usize,
    c0: usize,
    nb_cols_m1: usize,
    head_neighbors: &mut Vec<Option<usize>>,
    tail_neighbors: &mut Vec<Option<usize>>,
    square_cases: &Vec<u8>,
    positions: &Vec<usize>,
    is_head: bool,
) {
    let bottom_index = (r0 + 1) * nb_cols_m1 + c0;
    if bottom_index >= square_cases.len() {
        if is_head {
            head_neighbors.push(None);
        } else {
            tail_neighbors.push(None);
        }
    } else {
        let square_case;
        unsafe {
            square_case = square_cases.get_unchecked(bottom_index);
        }
        match (square_case, is_head) {
            // head
            (1 | 5 | 13, true) | (9, true) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(bottom_index);
                }
                head_neighbors.push(Some(*index));
            }
            (_, true) => {
                head_neighbors.push(None);
            }
            // tail
            (2 | 10 | 14, false) | (6, false) => {
                let index;
                unsafe {
                    index = positions.get_unchecked(bottom_index);
                }
                tail_neighbors.push(Some(*index));
            }
            (_, false) => {
                tail_neighbors.push(None);
            }
        }
    }
}
