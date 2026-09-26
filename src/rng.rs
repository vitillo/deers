//! Seeded global random number generator.
//!
//! Every stochastic op in deers draws from a single global generator:
//! [`Tensor::rand`](crate::Tensor::rand),
//! [`Tensor::randn`](crate::Tensor::randn), dropout, batch sampling, and the
//! [`permutation`](crate::dataset::permutation) /
//! [`shuffle`](crate::dataset::shuffle) helpers. Call [`manual_seed`] once up
//! front to make a run reproducible; without it the generator is seeded from
//! OS entropy, so runs differ by default and are reproducible only when seeded.

use std::sync::{Mutex, OnceLock};

use rand::SeedableRng;
use rand::rngs::StdRng;

/// Global generator behind every stochastic op; see the module docs.
static GLOBAL_RNG: OnceLock<Mutex<StdRng>> = OnceLock::new();

/// Returns the global generator, lazily seeded from OS entropy.
fn global_rng() -> &'static Mutex<StdRng> {
    GLOBAL_RNG.get_or_init(|| Mutex::new(StdRng::from_rng(&mut rand::rng())))
}

/// Seeds deers' global random number generator.
///
/// After `manual_seed(seed)`, `rand`, `randn`, dropout, batch sampling, and
/// shuffling all produce the same sequence on every run, so experiments and
/// training runs become reproducible. Call it once at startup, before building
/// models or sampling batches.
///
/// Without a call the generator is seeded from OS entropy, so runs differ by
/// default and are reproducible only when seeded.
pub fn manual_seed(seed: u64) {
    *global_rng().lock().expect("deers global RNG lock poisoned") = StdRng::seed_from_u64(seed);
}

/// Runs `f` with exclusive access to the global generator.
pub(crate) fn with_rng<R>(f: impl FnOnce(&mut StdRng) -> R) -> R {
    f(&mut global_rng().lock().expect("deers global RNG lock poisoned"))
}
