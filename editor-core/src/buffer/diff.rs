use std::{
    borrow::Cow,
    ops::Range,
    sync::{
        atomic::{self, AtomicBool, AtomicU64},
        Arc,
    },
};

use lapce_xi_rope::Rope;

const MAX_DIFF_MATRIX_CELLS: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffResult<T> {
    Left(T),
    Both(T, T),
    Right(T),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffBothInfo {
    pub left: Range<usize>,
    pub right: Range<usize>,
    pub skip: Option<Range<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffLines {
    Left(Range<usize>),
    Both(DiffBothInfo),
    Right(Range<usize>),
}

pub enum DiffExpand {
    Up(usize),
    Down(usize),
    All,
}

pub fn expand_diff_lines(
    diff_lines: &mut [DiffLines],
    line: usize,
    expand: DiffExpand,
    is_right: bool,
) {
    for diff_line in diff_lines.iter_mut() {
        if let DiffLines::Both(info) = diff_line {
            if (is_right && info.right.start == line) || (!is_right && info.left.start == line) {
                match expand {
                    DiffExpand::All => {
                        info.skip = None;
                    }
                    DiffExpand::Up(n) => {
                        if let Some(skip) = &mut info.skip {
                            if n >= skip.len() {
                                info.skip = None;
                            } else {
                                skip.start += n;
                            }
                        }
                    }
                    DiffExpand::Down(n) => {
                        if let Some(skip) = &mut info.skip {
                            if n >= skip.len() {
                                info.skip = None;
                            } else {
                                skip.end -= n;
                            }
                        }
                    }
                }
                break;
            }
        }
    }
}

pub fn rope_diff(
    left_rope: Rope,
    right_rope: Rope,
    rev: u64,
    atomic_rev: Arc<AtomicU64>,
    context_lines: Option<usize>,
) -> Option<Vec<DiffLines>> {
    rope_diff_cancellable(
        left_rope,
        right_rope,
        rev,
        atomic_rev,
        Arc::new(AtomicBool::new(false)),
        context_lines,
    )
}

pub fn rope_diff_cancellable(
    left_rope: Rope,
    right_rope: Rope,
    rev: u64,
    atomic_rev: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
    context_lines: Option<usize>,
) -> Option<Vec<DiffLines>> {
    let is_cancelled = || {
        cancelled.load(atomic::Ordering::Acquire)
            || atomic_rev.load(atomic::Ordering::Acquire) != rev
    };
    if is_cancelled() {
        return None;
    }
    let left_lines = left_rope.lines(..).collect::<Vec<Cow<str>>>();
    let right_lines = right_rope.lines(..).collect::<Vec<Cow<str>>>();

    let left_count = left_lines.len();
    let right_count = right_lines.len();
    let min_count = std::cmp::min(left_count, right_count);

    let leading_equals = left_lines
        .iter()
        .zip(right_lines.iter())
        .take_while(|p| p.0 == p.1)
        .count();
    let trailing_equals = left_lines
        .iter()
        .rev()
        .zip(right_lines.iter().rev())
        .take(min_count - leading_equals)
        .take_while(|p| p.0 == p.1)
        .count();

    let left_diff_size = left_count - leading_equals - trailing_equals;
    let right_diff_size = right_count - leading_equals - trailing_equals;

    let cells = (left_diff_size + 1).checked_mul(right_diff_size + 1);
    if cells.is_none_or(|cells| cells > MAX_DIFF_MATRIX_CELLS) {
        let mut changes = Vec::with_capacity(4);
        if leading_equals > 0 {
            changes.push(DiffLines::Both(DiffBothInfo {
                left: 0..leading_equals,
                right: 0..leading_equals,
                skip: None,
            }));
        }
        if left_diff_size > 0 {
            changes.push(DiffLines::Left(
                leading_equals..leading_equals + left_diff_size,
            ));
        }
        if right_diff_size > 0 {
            changes.push(DiffLines::Right(
                leading_equals..leading_equals + right_diff_size,
            ));
        }
        if trailing_equals > 0 {
            changes.push(DiffLines::Both(DiffBothInfo {
                left: left_count - trailing_equals..left_count,
                right: right_count - trailing_equals..right_count,
                skip: None,
            }));
        }
        if is_cancelled() {
            return None;
        }
        return Some(changes);
    }

    let table: Vec<Vec<u32>> = {
        let mut table = vec![vec![0; right_diff_size + 1]; left_diff_size + 1];
        let left_skip = left_lines.iter().skip(leading_equals).take(left_diff_size);
        let right_skip = right_lines
            .iter()
            .skip(leading_equals)
            .take(right_diff_size);

        for (i, l) in left_skip.enumerate() {
            for (j, r) in right_skip.clone().enumerate() {
                if is_cancelled() {
                    return None;
                }
                table[i + 1][j + 1] = if l == r {
                    table[i][j] + 1
                } else {
                    std::cmp::max(table[i][j + 1], table[i + 1][j])
                };
            }
        }

        table
    };

    let diff = {
        let mut diff = Vec::with_capacity(left_diff_size + right_diff_size);
        let mut i = left_diff_size;
        let mut j = right_diff_size;
        let mut li = left_lines.iter().rev().skip(trailing_equals);
        let mut ri = right_lines.iter().skip(trailing_equals);

        loop {
            if is_cancelled() {
                return None;
            }
            if j > 0 && (i == 0 || table[i][j] == table[i][j - 1]) {
                j -= 1;
                diff.push(DiffResult::Right(ri.next().unwrap()));
            } else if i > 0 && (j == 0 || table[i][j] == table[i - 1][j]) {
                i -= 1;
                diff.push(DiffResult::Left(li.next().unwrap()));
            } else if i > 0 && j > 0 {
                i -= 1;
                j -= 1;
                diff.push(DiffResult::Both(li.next().unwrap(), ri.next().unwrap()));
            } else {
                break;
            }
        }

        diff
    };

    let mut changes = Vec::new();
    let mut left_line = 0;
    let mut right_line = 0;
    if leading_equals > 0 {
        changes.push(DiffLines::Both(DiffBothInfo {
            left: 0..leading_equals,
            right: 0..leading_equals,
            skip: None,
        }))
    }
    left_line += leading_equals;
    right_line += leading_equals;

    for diff in diff.iter().rev() {
        if is_cancelled() {
            return None;
        }
        match diff {
            DiffResult::Left(_) => {
                match changes.last_mut() {
                    Some(DiffLines::Left(r)) => r.end = left_line + 1,
                    _ => changes.push(DiffLines::Left(left_line..left_line + 1)),
                }
                left_line += 1;
            }
            DiffResult::Both(_, _) => {
                match changes.last_mut() {
                    Some(DiffLines::Both(info)) => {
                        info.left.end = left_line + 1;
                        info.right.end = right_line + 1;
                    }
                    _ => changes.push(DiffLines::Both(DiffBothInfo {
                        left: left_line..left_line + 1,
                        right: right_line..right_line + 1,
                        skip: None,
                    })),
                }
                left_line += 1;
                right_line += 1;
            }
            DiffResult::Right(_) => {
                match changes.last_mut() {
                    Some(DiffLines::Right(r)) => r.end = right_line + 1,
                    _ => changes.push(DiffLines::Right(right_line..right_line + 1)),
                }
                right_line += 1;
            }
        }
    }

    if trailing_equals > 0 {
        changes.push(DiffLines::Both(DiffBothInfo {
            left: left_count - trailing_equals..left_count,
            right: right_count - trailing_equals..right_count,
            skip: None,
        }));
    }
    if let Some(context_lines) = context_lines {
        if !changes.is_empty() {
            let changes_last = changes.len() - 1;
            for (i, change) in changes.iter_mut().enumerate() {
                if is_cancelled() {
                    return None;
                }
                if let DiffLines::Both(info) = change {
                    if i == 0 || i == changes_last {
                        if info.right.len() > context_lines {
                            if i == 0 {
                                info.skip = Some(0..info.right.len() - context_lines);
                            } else {
                                info.skip = Some(context_lines..info.right.len());
                            }
                        }
                    } else if info.right.len() > context_lines * 2 {
                        info.skip = Some(context_lines..info.right.len() - context_lines);
                    }
                }
            }
        }
    }

    Some(changes)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64};

    use super::{rope_diff_cancellable, DiffLines};
    use lapce_xi_rope::Rope;

    #[test]
    fn oversized_diff_uses_coarse_ranges_and_preserves_context() {
        let left = format!("header\n{}footer", "left\n".repeat(2048));
        let right = format!("header\n{}footer", "right\n".repeat(2048));
        let revision = std::sync::Arc::new(AtomicU64::new(7));
        let changes = rope_diff_cancellable(
            Rope::from(left),
            Rope::from(right),
            7,
            revision,
            std::sync::Arc::new(AtomicBool::new(false)),
            None,
        )
        .unwrap();

        assert!(matches!(changes.first(), Some(DiffLines::Both(_))));
        assert!(matches!(changes.get(1), Some(DiffLines::Left(_))));
        assert!(matches!(changes.get(2), Some(DiffLines::Right(_))));
        assert!(matches!(changes.last(), Some(DiffLines::Both(_))));
    }

    #[test]
    fn cancelled_diff_returns_none() {
        let cancelled = std::sync::Arc::new(AtomicBool::new(true));
        let changes = rope_diff_cancellable(
            Rope::from("left"),
            Rope::from("right"),
            1,
            std::sync::Arc::new(AtomicU64::new(1)),
            cancelled,
            None,
        );
        assert!(changes.is_none());
    }
}
