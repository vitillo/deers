use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use crate::storage;
use crate::tensor::Tensor;

thread_local! {
    static ACTIVE_PROFILER: RefCell<Option<Rc<ProfilerSession>>> = const { RefCell::new(None) };
    static ACTIVE_SCOPE_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Configuration for a profiling session.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProfilerConfig {
    /// Whether to capture input tensor shapes per operation.
    pub record_shapes: bool,
    /// Whether to accumulate allocation sizes per operation.
    pub profile_memory: bool,
}

impl ProfilerConfig {
    /// Sets whether input shapes are recorded for each operation.
    pub fn record_shapes(mut self, record_shapes: bool) -> Self {
        self.record_shapes = record_shapes;
        self
    }

    /// Sets whether per-operation allocation sizes are tracked.
    pub fn profile_memory(mut self, profile_memory: bool) -> Self {
        self.profile_memory = profile_memory;
        self
    }
}

/// Aggregated stats for a single operation recorded during a profiling session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileRow {
    /// Operation name (e.g. `"add"`, `"matmul"`).
    pub name: String,
    /// Input shapes, present only when `ProfilerConfig::record_shapes` is set.
    pub input_shapes: Option<Vec<Vec<usize>>>,
    /// Number of times this operation was called.
    pub calls: u64,
    /// Total CPU-side time in nanoseconds across all calls.
    pub cpu_time_ns: u64,
    /// Total device execution time in nanoseconds across all calls; zero when backend timings are unavailable.
    pub device_time_ns: u64,
    /// Total bytes allocated across all calls; zero unless `ProfilerConfig::profile_memory` is set.
    pub alloc_bytes: u64,
}

/// The result of a completed profiling session.
#[derive(Clone, Debug)]
pub struct Profile {
    rows: Vec<ProfileRow>,
    profile_memory: bool,
    record_shapes: bool,
}

impl Profile {
    /// Returns per-operation rows sorted by total time (device first, then CPU).
    pub fn rows(&self) -> &[ProfileRow] {
        &self.rows
    }

    /// Renders a human-readable summary table as a string.
    pub fn table(&self) -> String {
        let has_device = self.rows.iter().any(|row| row.device_time_ns > 0);
        let has_shapes =
            self.record_shapes && self.rows.iter().any(|row| row.input_shapes.is_some());
        let has_memory = self.profile_memory;

        let mut lines = Vec::new();
        let mut header = format!(
            "{:<24}  {:>12}  {:>12}  {:>12}  {:>12}  {:>7}",
            "Name", "CPU total", "CPU avg", "Device total", "Device avg", "Calls"
        );
        if !has_device {
            header =
                format!("{:<24}  {:>12}  {:>12}  {:>7}", "Name", "CPU total", "CPU avg", "Calls");
        }
        if has_memory {
            header.push_str(&format!("  {:>10}", "Alloc"));
        }
        if has_shapes {
            header.push_str("  Input Shapes");
        }
        lines.push(header.clone());
        lines.push("-".repeat(header.len()));

        for row in &self.rows {
            let mut line = if has_device {
                format!(
                    "{:<24}  {:>12}  {:>12}  {:>12}  {:>12}  {:>7}",
                    row.name,
                    format_duration(row.cpu_time_ns),
                    format_duration(avg_time(row.cpu_time_ns, row.calls)),
                    format_duration(row.device_time_ns),
                    format_duration(avg_time(row.device_time_ns, row.calls)),
                    row.calls
                )
            } else {
                format!(
                    "{:<24}  {:>12}  {:>12}  {:>7}",
                    row.name,
                    format_duration(row.cpu_time_ns),
                    format_duration(avg_time(row.cpu_time_ns, row.calls)),
                    row.calls
                )
            };

            if has_memory {
                line.push_str(&format!("  {:>10}", format_bytes(row.alloc_bytes)));
            }
            if has_shapes {
                let shapes = row
                    .input_shapes
                    .as_ref()
                    .map(|shapes| format!("{shapes:?}"))
                    .unwrap_or_else(|| "[]".to_string());
                line.push_str("  ");
                line.push_str(&shapes);
            }

            lines.push(line);
        }

        lines.push("-".repeat(header.len()));
        lines.push(format!(
            "CPU total: {}",
            format_duration(self.rows.iter().map(|row| row.cpu_time_ns).sum())
        ));
        if has_device {
            lines.push(format!(
                "Device total: {}",
                format_duration(self.rows.iter().map(|row| row.device_time_ns).sum())
            ));
        }
        if has_memory {
            lines.push(format!(
                "Allocated: {}",
                format_bytes(self.rows.iter().map(|row| row.alloc_bytes).sum())
            ));
        }

        lines.join("\n")
    }
}

/// A handle to an active profiling session.
///
/// Call [`Profiler::start`] to begin recording, then [`Profiler::finish`] to
/// stop and collect results. Dropping a `Profiler` without calling `finish`
/// silently cancels the session.
pub struct Profiler {
    session: Rc<ProfilerSession>,
    active: bool,
}

impl Profiler {
    /// Synchronizes all backends and starts a new profiling session.
    ///
    /// Panics if a profiling session is already active on this thread.
    pub fn start(config: ProfilerConfig) -> Self {
        storage::synchronize_all();
        let session = Rc::new(ProfilerSession::new(config));
        ACTIVE_PROFILER.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "nested profiler sessions are not supported");
            *slot = Some(session.clone());
        });
        Self { session, active: true }
    }

    /// Synchronizes all backends, stops recording, and returns the collected [`Profile`].
    ///
    /// Panics if this profiler is no longer the active session on this thread.
    pub fn finish(mut self) -> Profile {
        self.active = false;
        storage::synchronize_all();
        ACTIVE_SCOPE_STACK.with(|stack| {
            assert!(stack.borrow().is_empty(), "profiler scope stack was left unbalanced");
        });
        ACTIVE_PROFILER.with(|slot| {
            let current = slot.borrow_mut().take();
            assert!(
                current.as_ref().is_some_and(|active| Rc::ptr_eq(active, &self.session)),
                "finishing a profiler that is not active"
            );
        });
        self.session.snapshot()
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        ACTIVE_SCOPE_STACK.with(|stack| stack.borrow_mut().clear());
        ACTIVE_PROFILER.with(|slot| {
            let _ = slot.borrow_mut().take();
        });
    }
}

/// Convenience wrapper: runs `f` under a profiling session and returns both the result and the [`Profile`].
pub fn profile<R>(config: ProfilerConfig, f: impl FnOnce() -> R) -> (R, Profile) {
    let profiler = Profiler::start(config);
    let result = f();
    let profile = profiler.finish();
    (result, profile)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EventKey {
    name: String,
    input_shapes: Option<Vec<Vec<usize>>>,
}

#[derive(Clone, Debug, Default)]
struct EventStat {
    calls: u64,
    cpu_time_ns: u64,
    device_time_ns: u64,
    alloc_bytes: u64,
}

#[derive(Default)]
struct ProfilerInner {
    stats: Vec<EventStat>,
    keys: Vec<EventKey>,
    index_by_key: HashMap<EventKey, usize>,
}

struct ProfilerSession {
    config: ProfilerConfig,
    inner: RefCell<ProfilerInner>,
}

impl ProfilerSession {
    fn new(config: ProfilerConfig) -> Self {
        Self { config, inner: RefCell::new(ProfilerInner::default()) }
    }

    fn event_id(&self, name: &'static str, inputs: &[&Tensor]) -> usize {
        let key = EventKey {
            name: name.to_string(),
            input_shapes: self.config.record_shapes.then(|| {
                inputs
                    .iter()
                    .map(|tensor| tensor.layout().shape().iter().copied().collect())
                    .collect()
            }),
        };
        let mut inner = self.inner.borrow_mut();
        if let Some(&id) = inner.index_by_key.get(&key) {
            return id;
        }
        let id = inner.stats.len();
        inner.stats.push(EventStat::default());
        inner.keys.push(key.clone());
        inner.index_by_key.insert(key, id);
        id
    }

    fn record_cpu_time(&self, event_id: usize, elapsed_ns: u64, alloc_bytes: u64) {
        let mut inner = self.inner.borrow_mut();
        let stat = &mut inner.stats[event_id];
        stat.calls += 1;
        stat.cpu_time_ns += elapsed_ns;
        stat.alloc_bytes += alloc_bytes;
    }

    #[allow(dead_code)]
    fn record_device_time(&self, event_id: usize, elapsed_ns: u64) {
        self.inner.borrow_mut().stats[event_id].device_time_ns += elapsed_ns;
    }

    fn snapshot(&self) -> Profile {
        let inner = self.inner.borrow();
        let mut rows: Vec<_> = inner
            .keys
            .iter()
            .zip(inner.stats.iter())
            .map(|(key, stat)| ProfileRow {
                name: key.name.clone(),
                input_shapes: key.input_shapes.clone(),
                calls: stat.calls,
                cpu_time_ns: stat.cpu_time_ns,
                device_time_ns: stat.device_time_ns,
                alloc_bytes: stat.alloc_bytes,
            })
            .collect();
        rows.sort_by(|a, b| {
            b.device_time_ns
                .cmp(&a.device_time_ns)
                .then_with(|| b.cpu_time_ns.cmp(&a.cpu_time_ns))
                .then_with(|| a.name.cmp(&b.name))
        });
        Profile {
            rows,
            profile_memory: self.config.profile_memory,
            record_shapes: self.config.record_shapes,
        }
    }
}

pub(crate) struct ProfileScope {
    session: Rc<ProfilerSession>,
    event_id: usize,
    alloc_bytes: u64,
    start: Instant,
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;
        ACTIVE_SCOPE_STACK.with(|stack| {
            let popped = stack.borrow_mut().pop();
            assert_eq!(popped, Some(self.event_id), "profiler scope stack is out of sync");
        });
        self.session.record_cpu_time(self.event_id, elapsed_ns, self.alloc_bytes);
    }
}

pub(crate) fn scope(
    name: &'static str,
    inputs: &[&Tensor],
    alloc_bytes: usize,
) -> Option<ProfileScope> {
    let session = ACTIVE_PROFILER.with(|slot| slot.borrow().clone())?;
    let event_id = session.event_id(name, inputs);
    ACTIVE_SCOPE_STACK.with(|stack| stack.borrow_mut().push(event_id));
    let alloc_bytes = if session.config.profile_memory { alloc_bytes as u64 } else { 0 };
    Some(ProfileScope { session, event_id, alloc_bytes, start: Instant::now() })
}

#[allow(dead_code)]
pub(crate) fn current_scope_id() -> Option<usize> {
    ACTIVE_SCOPE_STACK.with(|stack| stack.borrow().last().copied())
}

#[allow(dead_code)]
pub(crate) fn is_active() -> bool {
    ACTIVE_PROFILER.with(|slot| slot.borrow().is_some())
}

#[allow(dead_code)]
pub(crate) fn record_device_time(event_id: usize, elapsed_ns: u64) {
    ACTIVE_PROFILER.with(|slot| {
        if let Some(session) = slot.borrow().as_ref() {
            session.record_device_time(event_id, elapsed_ns);
        }
    });
}

fn avg_time(total_ns: u64, calls: u64) -> u64 {
    if calls == 0 { 0 } else { total_ns / calls }
}

fn format_duration(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        return format!("{:.3}s", ns as f64 / 1_000_000_000.0);
    }
    if ns >= 1_000_000 {
        return format!("{:.3}ms", ns as f64 / 1_000_000.0);
    }
    if ns >= 1_000 {
        return format!("{:.3}us", ns as f64 / 1_000.0);
    }
    format!("{ns}ns")
}

fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;

    if bytes as f64 >= MB {
        return format!("{:.2} MB", bytes as f64 / MB);
    }
    if bytes as f64 >= KB {
        return format!("{:.2} KB", bytes as f64 / KB);
    }
    format!("{bytes} B")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device, Tensor};

    #[test]
    fn aggregates_cpu_time_shapes_and_allocations() {
        // Arrange
        let config = ProfilerConfig::default().record_shapes(true).profile_memory(true);

        // Act
        let (_, profile) = profile(config, || {
            let lhs = Tensor::ones((2, 3), DType::F32, Device::Cpu);
            let rhs = Tensor::ones((2, 3), DType::F32, Device::Cpu);
            let _ = &lhs + &rhs;
            let _ = &lhs + &rhs;
        });

        // Assert
        let row = profile.rows().iter().find(|row| row.name == "add").unwrap();
        assert_eq!(row.calls, 2);
        assert_eq!(row.alloc_bytes, (2 * 3 * std::mem::size_of::<f32>() * 2) as u64);
        assert_eq!(row.input_shapes, Some(vec![vec![2, 3], vec![2, 3]]));
        assert!(row.cpu_time_ns > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn flushes_pending_mps_work_on_finish() {
        let Some(_device) = metal::Device::system_default() else {
            return;
        };

        // Arrange
        let config = ProfilerConfig::default();

        // Act
        let (tensor, profile) = profile(config, || {
            let lhs = Tensor::ones((256, 256), crate::DType::F32, Device::Mps);
            let rhs = Tensor::ones((256, 256), crate::DType::F32, Device::Mps);
            lhs.matmul(&rhs)
        });

        // Assert
        let values = tensor.to_vec::<f32>().unwrap();
        assert_eq!(values[0], 256.0);

        let row = profile.rows().iter().find(|row| row.name == "matmul").unwrap();
        assert_eq!(row.calls, 1);
    }
}
