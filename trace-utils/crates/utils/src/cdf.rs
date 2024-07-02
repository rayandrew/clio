use num::{FromPrimitive, Num, ToPrimitive};

// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=1a94ff90ffeea07a25074a15ed974d54
// from https://users.rust-lang.org/t/observed-cdf-of-a-vector/77566/4
pub fn calc_cdf<T: Num + FromPrimitive + ToPrimitive + Clone + PartialOrd + Copy>(
    x: &[T],
) -> Vec<(T, f64)> {
    let ln = x.len() as f64;
    let mut x_ord = x.to_vec();
    x_ord.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if let Some(mut previous) = x_ord.get(0).map(|&f| f) {
        let mut cdf = Vec::new();
        for (i, f) in x_ord.into_iter().enumerate() {
            if f != previous {
                cdf.push((previous, i as f64 / ln));
                previous = f;
            }
        }

        cdf.push((previous, 1.0));
        cdf
    } else {
        Vec::new()
    }
}
