use numpy::ndarray::ArrayViewD;

pub fn find_segments(
    array: &ArrayViewD<'_, f64>,
    level: f64,
    vertex_connect_high: bool,
    mask: &ArrayViewD<'_, u8>,
) -> (Vec<f64>, Vec<u8>) {
    let shape = array.shape();
    let nb_rows = shape[0];
    let nb_cols = shape[1];
    let mut segments = Vec::with_capacity(nb_rows * nb_cols);
    let mut square_cases: Vec<u8> = Vec::with_capacity((nb_rows - 1) * (nb_cols - 1));
    for r0 in 0..(nb_rows - 1) {
        let r1 = r0 + 1;
        for c0 in 0..(nb_cols - 1) {
            let c1 = c0 + 1;
            let (ul, ll, ur, lr): (f64, f64, f64, f64);
            unsafe {
                let is_masked = mask.uget([r0, c0])
                    * mask.uget([r1, c0])
                    * mask.uget([r0, c1])
                    * mask.uget([r1, c1]);
                if is_masked == 0 {
                    square_cases.push(0);
                    continue;
                }
                ul = *array.uget([r0, c0]);
                ll = *array.uget([r1, c0]);
                ur = *array.uget([r0, c1]);
                lr = *array.uget([r1, c1]);
            }
            let square_case = 1 * u8::from(ul > level) as u8
                | 2 * (ur > level) as u8
                | 4 * (ll > level) as u8
                | 8 * (lr > level) as u8;
            square_cases.push(square_case);
            let square_segments = match square_case {
                1 => vec![
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ],
                2 => vec![
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ],
                3 => vec![
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ],
                4 => vec![
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ],
                5 => vec![
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ],
                6 => match vertex_connect_high {
                    true => vec![
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        // seg 2
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                    ],
                    false => vec![
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        // seg 2
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                    ],
                },
                7 => vec![
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                ],
                8 => vec![
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ],
                9 => match vertex_connect_high {
                    true => vec![
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                        // seg 2
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                    ],
                    false => vec![
                        top_x(r0, c0, ul, ur, level),
                        top_y(r0, c0, ul, ur, level),
                        left_x(r0, c0, ul, ll, level),
                        left_y(r0, c0, ul, ll, level),
                        // seg 2
                        bottom_x(r1, c0, ll, lr, level),
                        bottom_y(r1, c0, ll, lr, level),
                        right_x(r0, c1, ur, lr, level),
                        right_y(r0, c1, ur, lr, level),
                    ],
                },
                10 => vec![
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ],
                11 => vec![
                    bottom_x(r1, c0, ll, lr, level),
                    bottom_y(r1, c0, ll, lr, level),
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                ],
                12 => vec![
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ],
                13 => vec![
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                    right_x(r0, c1, ur, lr, level),
                    right_y(r0, c1, ur, lr, level),
                ],
                14 => vec![
                    left_x(r0, c0, ul, ll, level),
                    left_y(r0, c0, ul, ll, level),
                    top_x(r0, c0, ul, ur, level),
                    top_y(r0, c0, ul, ur, level),
                ],
                0 | 15 => vec![], // No segments pass through the square
                other_case => unreachable!("Unexpected square case: {}", other_case),
            };
            segments.extend(square_segments);
        }
    }
    (segments, square_cases)
}

#[inline(always)]
fn top_x(r: usize, _: usize, _: f64, _: f64, _: f64) -> f64 {
    r as f64
}

#[inline(always)]
fn top_y(_: usize, c: usize, ul: f64, ur: f64, level: f64) -> f64 {
    c as f64 + get_fraction(ul, ur, level)
}

#[inline(always)]
fn bottom_x(r: usize, _: usize, _: f64, _: f64, _: f64) -> f64 {
    r as f64
}

#[inline(always)]
fn bottom_y(_: usize, c: usize, ll: f64, lr: f64, level: f64) -> f64 {
    c as f64 + get_fraction(ll, lr, level)
}

#[inline(always)]
fn left_x(r: usize, _: usize, ul: f64, ll: f64, level: f64) -> f64 {
    r as f64 + get_fraction(ul, ll, level)
}

#[inline(always)]
fn left_y(_: usize, c: usize, _: f64, _: f64, _: f64) -> f64 {
    c as f64
}

#[inline(always)]
fn right_x(r: usize, _: usize, ur: f64, lr: f64, level: f64) -> f64 {
    r as f64 + get_fraction(ur, lr, level)
}

#[inline(always)]
fn right_y(_: usize, c: usize, _: f64, _: f64, _: f64) -> f64 {
    c as f64
}

#[inline(always)]
fn get_fraction(from_value: f64, to_value: f64, level: f64) -> f64 {
    if to_value == from_value {
        return 0.0;
    }
    return (level - from_value) / (to_value - from_value);
}
