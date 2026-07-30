//! Linux-only, bounded shared-memory ingress for the inference daemon.
//!
//! ## ABI and ordering
//! The C++ daemon owns the System V segment.  The gateway only attaches and
//! calls `shmdt` on drop: it never calls `IPC_RMID`.  All fields below are
//! little-endian.  The `AtomicU64`/`AtomicU32` fields are naturally aligned;
//! this is a Linux x86_64/aarch64 contract, where these atomics are lock-free.
//! Do not use this ABI on a platform that emulates process-shared atomics.
//!
//! A slot starts with `sequence == i`.  A producer claims position `p` only
//! after observing `sequence == p`, publishes with `sequence = p + 1`
//! (release), and increments `ready_count`.  The daemon writes the response
//! and stores `p + 2` (release).  The gateway reads the response after an
//! acquire load and returns ownership with `sequence = p + capacity`.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("fraud-shm-api-gateway is Linux-only: it uses System V shared memory");
#[cfg(not(target_pointer_width = "64"))]
compile_error!("fraud-shm-api-gateway requires naturally aligned 64-bit atomics");
#[cfg(not(target_endian = "little"))]
compile_error!("fraud-shm-api-gateway shared-memory ABI is little-endian only");

#[allow(clippy::all, dead_code)]
mod transaction_generated {
    include!("generated/transaction_generated.rs");
}
use transaction_generated::fraud::ipc as transaction_fb;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use flatbuffers::{Allocator, FlatBufferBuilder};
use serde::Deserialize;
use serde_json::json;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;

pub const MAGIC: u32 = 0x4644_4950;
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 320;
pub const SLOT_PREFIX_BYTES: usize = 64;
const MAX_BODY_BYTES: usize = 4096;
const MAX_STRING_BYTES: usize = 128;
const MIN_TOKEN_BYTES: usize = 16;
const MAX_TOKEN_BYTES: usize = 256;
/// $9 trillion represented in micro-units, below i64::MAX.
const MAX_ABS_AMOUNT_MICROS: u64 = 9_000_000_000_000_000_000;
const MAX_WAIT: Duration = Duration::from_millis(50);
const PARK_SLICE: Duration = Duration::from_micros(50);
const RECLAIMER_TICK: Duration = Duration::from_micros(100);
const DAEMON_STATUS_INCOMPLETE: u32 = u32::MAX;
const DAEMON_STATUS_CANCELLED: u32 = u32::MAX - 1;

const MAGIC_OFFSET: usize = 0;
const VERSION_OFFSET: usize = 4;
const HEADER_BYTES_OFFSET: usize = 8;
const SLOT_COUNT_OFFSET: usize = 12;
const SLOT_BYTES_OFFSET: usize = 16;
const ENQUEUE_OFFSET: usize = 64;
const DEQUEUE_OFFSET: usize = 128;
const READY_COUNT_OFFSET: usize = 192;
const SHUTDOWN_OFFSET: usize = 256;

const SLOT_SEQUENCE_OFFSET: usize = 0;
const SLOT_PAYLOAD_OFFSET: usize = 8;
const SLOT_PAYLOAD_SIZE: usize = 12;
const SLOT_RESPONSE_STATUS: usize = 16;
const SLOT_DECISION: usize = 20;
const SLOT_SCORE: usize = 24;
const SLOT_REQUEST_ID: usize = 32;
const SLOT_ENQUEUE_NS: usize = 40;
const SLOT_COMPLETE_NS: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentLayout {
    pub slot_count: u32,
    pub slot_bytes: u32,
    pub segment_bytes: usize,
}

#[derive(Debug)]
pub enum GatewayError {
    Attach(std::io::Error),
    Abi(&'static str),
    Auth,
    Invalid(&'static str),
    Full,
    Timeout,
    DaemonStopped,
    Internal(&'static str),
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attach(error) => write!(f, "shared-memory attach failed: {error}"),
            Self::Abi(message) | Self::Invalid(message) | Self::Internal(message) => {
                f.write_str(message)
            }
            Self::Auth => f.write_str("unauthorized"),
            Self::Full => f.write_str("inference queue is full"),
            Self::Timeout => f.write_str("inference deadline exceeded"),
            Self::DaemonStopped => f.write_str("inference daemon is stopping"),
        }
    }
}

impl std::error::Error for GatewayError {}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Auth => StatusCode::UNAUTHORIZED,
            Self::Invalid(_) => StatusCode::BAD_REQUEST,
            Self::Full | Self::Timeout | Self::DaemonStopped => StatusCode::SERVICE_UNAVAILABLE,
            Self::Attach(_) | Self::Abi(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({"error": self.to_string()}))).into_response()
    }
}

fn read_u32(base: *const u8, offset: usize) -> u32 {
    // ABI scalars are byte-addressed and little endian, so no alignment is assumed.
    unsafe { u32::from_le_bytes(std::ptr::read_unaligned(base.add(offset).cast::<[u8; 4]>())) }
}

fn write_u32(base: *mut u8, offset: usize, value: u32) {
    unsafe { std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), base.add(offset), 4) }
}

fn write_u64(base: *mut u8, offset: usize, value: u64) {
    unsafe { std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), base.add(offset), 8) }
}

fn read_f32(base: *const u8, offset: usize) -> f32 {
    f32::from_bits(read_u32(base, offset))
}

fn atomic_u64(base: *mut u8, offset: usize) -> Result<&'static AtomicU64, GatewayError> {
    let pointer = unsafe { base.add(offset).cast::<AtomicU64>() };
    if (pointer as usize) % std::mem::align_of::<AtomicU64>() != 0 {
        return Err(GatewayError::Abi("unaligned shared AtomicU64"));
    }
    // The segment outlives all Gateway references; C++ uses the same atomic ABI.
    Ok(unsafe { &*pointer })
}

fn atomic_u32(base: *mut u8, offset: usize) -> Result<&'static AtomicU32, GatewayError> {
    let pointer = unsafe { base.add(offset).cast::<AtomicU32>() };
    if (pointer as usize) % std::mem::align_of::<AtomicU32>() != 0 {
        return Err(GatewayError::Abi("unaligned shared AtomicU32"));
    }
    Ok(unsafe { &*pointer })
}

fn validate_layout(base: *mut u8, bytes: usize) -> Result<SegmentLayout, GatewayError> {
    if base.is_null() || bytes < HEADER_BYTES {
        return Err(GatewayError::Abi(
            "shared segment is smaller than its header",
        ));
    }
    if read_u32(base, MAGIC_OFFSET) != MAGIC || read_u32(base, VERSION_OFFSET) != VERSION {
        return Err(GatewayError::Abi("shared segment magic/version mismatch"));
    }
    if read_u32(base, HEADER_BYTES_OFFSET) as usize != HEADER_BYTES {
        return Err(GatewayError::Abi("shared segment header size mismatch"));
    }
    let slot_count = read_u32(base, SLOT_COUNT_OFFSET);
    let slot_bytes = read_u32(base, SLOT_BYTES_OFFSET);
    // The p+2 completion and p+capacity reuse protocol requires at least four
    // positions. Smaller rings can alias a completed sequence with reuse.
    if slot_count < 4 || !slot_count.is_power_of_two() || slot_bytes as usize <= SLOT_PREFIX_BYTES {
        return Err(GatewayError::Abi("invalid slot dimensions"));
    }
    let segment_bytes = HEADER_BYTES
        .checked_add(
            (slot_count as usize)
                .checked_mul(slot_bytes as usize)
                .ok_or(GatewayError::Abi("slot dimensions overflow"))?,
        )
        .ok_or(GatewayError::Abi("segment dimensions overflow"))?;
    if segment_bytes > bytes {
        return Err(GatewayError::Abi("shared segment is truncated"));
    }
    for offset in [ENQUEUE_OFFSET, DEQUEUE_OFFSET, READY_COUNT_OFFSET] {
        let _ = atomic_u64(base, offset)?;
    }
    let _ = atomic_u32(base, SHUTDOWN_OFFSET)?;
    for index in 0..slot_count as usize {
        let slot = unsafe { base.add(HEADER_BYTES + index * slot_bytes as usize) };
        let _ = atomic_u64(slot, SLOT_SEQUENCE_OFFSET)?;
    }
    Ok(SegmentLayout {
        slot_count,
        slot_bytes,
        segment_bytes,
    })
}

/// An attach-only handle.  `Drop` detaches but never marks the daemon-owned
/// segment for deletion.
pub struct SharedSegment {
    base: NonNull<u8>,
    shmid: libc::c_int,
    layout: SegmentLayout,
}

unsafe impl Send for SharedSegment {}
unsafe impl Sync for SharedSegment {}

impl SharedSegment {
    pub fn attach(key: libc::key_t) -> Result<Self, GatewayError> {
        let shmid = unsafe { libc::shmget(key, 0, 0) };
        if shmid < 0 {
            return Err(GatewayError::Attach(std::io::Error::last_os_error()));
        }
        let address = unsafe { libc::shmat(shmid, std::ptr::null(), 0) };
        if address == (-1isize) as *mut libc::c_void {
            return Err(GatewayError::Attach(std::io::Error::last_os_error()));
        }
        let mut info: libc::shmid_ds = unsafe { std::mem::zeroed() };
        if unsafe { libc::shmctl(shmid, libc::IPC_STAT, &mut info) } != 0 {
            let error = std::io::Error::last_os_error();
            unsafe { libc::shmdt(address) };
            return Err(GatewayError::Attach(error));
        }
        let base =
            NonNull::new(address.cast::<u8>()).ok_or(GatewayError::Abi("null shmat result"))?;
        let layout = match validate_layout(base.as_ptr(), info.shm_segsz) {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { libc::shmdt(address) };
                return Err(error);
            }
        };
        Ok(Self {
            base,
            shmid,
            layout,
        })
    }

    pub fn layout(&self) -> SegmentLayout {
        self.layout
    }
    pub fn shmid(&self) -> libc::c_int {
        self.shmid
    }

    fn header_u64(&self, offset: usize) -> &AtomicU64 {
        atomic_u64(self.base.as_ptr(), offset).expect("validated ABI")
    }
    fn header_u32(&self, offset: usize) -> &AtomicU32 {
        atomic_u32(self.base.as_ptr(), offset).expect("validated ABI")
    }
    fn slot(&self, index: usize) -> *mut u8 {
        debug_assert!(index < self.layout.slot_count as usize);
        unsafe {
            self.base
                .as_ptr()
                .add(HEADER_BYTES + index * self.layout.slot_bytes as usize)
        }
    }
}

impl Drop for SharedSegment {
    fn drop(&mut self) {
        // shmdt only removes this process's mapping.  IPC_RMID is forbidden here.
        unsafe { libc::shmdt(self.base.as_ptr().cast()) };
    }
}

#[derive(Debug)]
pub struct FixedBufferError;
impl fmt::Display for FixedBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FlatBuffer exceeds shared slot")
    }
}

/// Fixed, non-growing backing storage for FlatBuffers.  It writes directly to
/// a slot's payload area; `grow_downwards` returns an error rather than ever
/// allocating or copying to a heap buffer.
pub struct SharedSlotAllocator {
    start: NonNull<u8>,
    len: usize,
}

impl SharedSlotAllocator {
    /// # Safety
    /// `start..start+len` must stay exclusively owned until the builder drops.
    pub unsafe fn new(start: *mut u8, len: usize) -> Result<Self, GatewayError> {
        Ok(Self {
            start: NonNull::new(start).ok_or(GatewayError::Internal("null slot payload"))?,
            len,
        })
    }
}

impl Deref for SharedSlotAllocator {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.start.as_ptr(), self.len) }
    }
}
impl DerefMut for SharedSlotAllocator {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.start.as_ptr(), self.len) }
    }
}
unsafe impl Allocator for SharedSlotAllocator {
    type Error = FixedBufferError;
    fn grow_downwards(&mut self) -> Result<(), Self::Error> {
        Err(FixedBufferError)
    }
    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IncomingTransaction<'a> {
    transaction_id: &'a str,
    account_id: &'a str,
    amount_micros: i64,
    currency: &'a str,
    occurred_at_ns: u64,
    #[serde(default)]
    merchant_category: Option<&'a str>,
}

fn safe_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_STRING_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}
fn safe_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_STRING_BYTES && !value.bytes().any(|byte| byte < 0x20)
}

impl<'a> IncomingTransaction<'a> {
    fn validate(&self) -> Result<i64, GatewayError> {
        if !safe_identifier(self.transaction_id)
            || !safe_identifier(self.account_id)
            || !safe_identifier(self.currency)
        {
            return Err(GatewayError::Invalid("invalid transaction identifier"));
        }
        if let Some(category) = self.merchant_category {
            if !safe_text(category) {
                return Err(GatewayError::Invalid("invalid merchant category"));
            }
        }
        if self.amount_micros.unsigned_abs() > MAX_ABS_AMOUNT_MICROS {
            return Err(GatewayError::Invalid("amount is out of range"));
        }
        Ok(self.amount_micros)
    }
    fn worst_case_bytes(&self) -> usize {
        // Four zero-terminated strings + table/vtable/root/id + conservative alignment.
        128 + self.transaction_id.len()
            + self.account_id.len()
            + self.currency.len()
            + self.merchant_category.map_or(0, str::len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DaemonResponseStatus {
    Scored = 200,
    InvalidInput = 422,
    Unavailable = 503,
}

impl DaemonResponseStatus {
    fn http_status(self) -> StatusCode {
        match self {
            Self::Scored => StatusCode::OK,
            Self::InvalidInput => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    fn as_u32(self) -> u32 {
        self as u32
    }
}

impl TryFrom<u32> for DaemonResponseStatus {
    type Error = GatewayError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            200 => Ok(Self::Scored),
            422 => Ok(Self::InvalidInput),
            503 => Ok(Self::Unavailable),
            DAEMON_STATUS_INCOMPLETE => Err(GatewayError::Internal(
                "daemon completed slot without response",
            )),
            _ => Err(GatewayError::Internal(
                "daemon returned unknown response status",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InferenceResponse {
    pub request_id: u64,
    pub status: DaemonResponseStatus,
    pub decision: u32,
    pub score: f32,
    pub completed_ns: u64,
}

#[derive(Clone, Copy)]
struct Reclaim {
    position: u64,
}

struct ReclaimQueue {
    pending: Mutex<Vec<Reclaim>>,
    capacity: usize,
}

impl ReclaimQueue {
    fn push(&self, reclaim: Reclaim) {
        loop {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if pending.iter().any(|item| item.position == reclaim.position) {
                return;
            }
            if pending.len() < self.capacity {
                // Both vectors are allocated to slot_count at construction and
                // retain capacity while being swapped by the reaper, so this
                // cannot grow the queue on the hot request path.
                pending.push(reclaim);
                return;
            }
            drop(pending);
            // The reaper swaps its intake list every 100 us even when an older
            // position is permanently stuck. This bounded retry cannot be
            // blocked behind an earlier request as the old channel design was.
            std::thread::yield_now();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationState {
    Held,
    Published,
    HandedOff,
    Released,
}

/// Owns a claimed ring position until it has either been returned or handed to
/// the bounded reclaimer. Its Drop implementation turns every pre-publication
/// error or unwind into a daemon-consumable cancellation record.
struct SlotReservation {
    segment: Arc<SharedSegment>,
    reclaims: Arc<ReclaimQueue>,
    position: u64,
    request_id: u64,
    enqueue_ns: u64,
    state: ReservationState,
}

impl SlotReservation {
    fn slot(&self) -> *mut u8 {
        self.segment
            .slot((self.position as usize) & (self.segment.layout.slot_count as usize - 1))
    }

    fn publish_success(&mut self, range: PayloadRange) {
        let slot = self.slot();
        write_u32(slot, SLOT_PAYLOAD_OFFSET, range.offset as u32);
        write_u32(slot, SLOT_PAYLOAD_SIZE, range.size as u32);
        write_u32(slot, SLOT_RESPONSE_STATUS, DAEMON_STATUS_INCOMPLETE);
        write_u32(slot, SLOT_DECISION, 0);
        write_u32(slot, SLOT_SCORE, 0);
        write_u64(slot, SLOT_REQUEST_ID, self.request_id);
        write_u64(slot, SLOT_ENQUEUE_NS, self.enqueue_ns);
        write_u64(slot, SLOT_COMPLETE_NS, 0);
        atomic_u64(slot, SLOT_SEQUENCE_OFFSET)
            .expect("validated slot atomic")
            .store(self.position + 1, Ordering::Release);
        self.segment
            .header_u64(READY_COUNT_OFFSET)
            .fetch_add(1, Ordering::Release);
        self.state = ReservationState::Published;
    }

    fn publish_cancel(&mut self) {
        if self.state != ReservationState::Held {
            return;
        }
        let slot = self.slot();
        // A zero-size payload is a cancellation record. The C++ daemon must
        // skip model execution, decrement ready_count, and complete p+2.
        write_u32(slot, SLOT_PAYLOAD_OFFSET, SLOT_PREFIX_BYTES as u32);
        write_u32(slot, SLOT_PAYLOAD_SIZE, 0);
        write_u32(slot, SLOT_RESPONSE_STATUS, DAEMON_STATUS_CANCELLED);
        write_u32(slot, SLOT_DECISION, 0);
        write_u32(slot, SLOT_SCORE, 0);
        write_u64(slot, SLOT_REQUEST_ID, self.request_id);
        write_u64(slot, SLOT_ENQUEUE_NS, self.enqueue_ns);
        write_u64(slot, SLOT_COMPLETE_NS, 0);
        atomic_u64(slot, SLOT_SEQUENCE_OFFSET)
            .expect("validated slot atomic")
            .store(self.position + 1, Ordering::Release);
        self.segment
            .header_u64(READY_COUNT_OFFSET)
            .fetch_add(1, Ordering::Release);
        self.state = ReservationState::Published;
    }

    fn release(&mut self) {
        debug_assert_eq!(self.state, ReservationState::Published);
        atomic_u64(self.slot(), SLOT_SEQUENCE_OFFSET)
            .expect("validated slot atomic")
            .store(
                self.position + self.segment.layout.slot_count as u64,
                Ordering::Release,
            );
        self.state = ReservationState::Released;
    }

    fn handoff(&mut self) {
        if self.state != ReservationState::Published {
            return;
        }
        let sequence =
            atomic_u64(self.slot(), SLOT_SEQUENCE_OFFSET).expect("validated slot atomic");
        if sequence.load(Ordering::Acquire) == self.position + 2 {
            self.release();
        } else {
            self.reclaims.push(Reclaim {
                position: self.position,
            });
            self.state = ReservationState::HandedOff;
        }
    }
}

impl Drop for SlotReservation {
    fn drop(&mut self) {
        match self.state {
            ReservationState::Held => {
                self.publish_cancel();
                self.handoff();
            }
            ReservationState::Published => self.handoff(),
            ReservationState::HandedOff | ReservationState::Released => {}
        }
    }
}

#[derive(Clone, Copy)]
struct PayloadRange {
    offset: usize,
    size: usize,
}

fn serialize_transaction(
    slot: *mut u8,
    payload_capacity: usize,
    request_id: u64,
    input: &IncomingTransaction<'_>,
    amount_micros: i64,
) -> Result<PayloadRange, GatewayError> {
    let allocator =
        unsafe { SharedSlotAllocator::new(slot.add(SLOT_PREFIX_BYTES), payload_capacity) }?;
    let mut builder = FlatBufferBuilder::new_in(allocator);
    let transaction_id = builder.create_string(input.transaction_id);
    let account_id = builder.create_string(input.account_id);
    let currency = builder.create_string(input.currency);
    let merchant_category = input
        .merchant_category
        .map(|value| builder.create_string(value));
    let root = transaction_fb::Transaction::create(
        &mut builder,
        &transaction_fb::TransactionArgs {
            request_id,
            transaction_id: Some(transaction_id),
            account_id: Some(account_id),
            amount_micros,
            currency: Some(currency),
            occurred_at_ns: input.occurred_at_ns,
            merchant_category,
        },
    );
    transaction_fb::finish_transaction_buffer(&mut builder, root);
    let size = builder.finished_data().len();
    let (_, start) = builder.mut_finished_buffer();
    let end = start
        .checked_add(size)
        .filter(|end| *end <= payload_capacity)
        .ok_or(GatewayError::Internal("invalid FlatBuffer slot bounds"))?;
    let _ = end;
    Ok(PayloadRange {
        offset: SLOT_PREFIX_BYTES + start,
        size,
    })
}

pub struct Gateway {
    segment: Arc<SharedSegment>,
    bearer_token: [u8; MAX_TOKEN_BYTES],
    bearer_token_len: usize,
    deadline: Duration,
    reclaims: Arc<ReclaimQueue>,
}

impl Gateway {
    pub fn new(
        segment: SharedSegment,
        bearer_token: impl Into<Vec<u8>>,
        deadline: Duration,
    ) -> Result<Arc<Self>, GatewayError> {
        let token = bearer_token.into();
        if !(MIN_TOKEN_BYTES..=MAX_TOKEN_BYTES).contains(&token.len()) {
            return Err(GatewayError::Invalid(
                "FRAUD_GATEWAY_TOKEN must contain 16..=256 bytes",
            ));
        }
        let mut padded_token = [0_u8; MAX_TOKEN_BYTES];
        padded_token[..token.len()].copy_from_slice(&token);
        let segment = Arc::new(segment);
        let reclaims = Arc::new(ReclaimQueue {
            pending: Mutex::new(Vec::with_capacity(segment.layout.slot_count as usize)),
            capacity: segment.layout.slot_count as usize,
        });
        let reaper_segment = Arc::clone(&segment);
        let reaper_queue = Arc::clone(&reclaims);
        std::thread::Builder::new()
            .name("fraud-shm-reclaimer".into())
            .spawn(move || run_reclaimer(reaper_segment, reaper_queue))
            .map_err(|_| GatewayError::Internal("could not start shared-memory reclaimer"))?;
        Ok(Arc::new(Self {
            segment,
            bearer_token: padded_token,
            bearer_token_len: token.len(),
            deadline: deadline.min(MAX_WAIT),
            reclaims,
        }))
    }

    pub fn is_ready(&self) -> bool {
        self.segment
            .header_u32(SHUTDOWN_OFFSET)
            .load(Ordering::Acquire)
            == 0
    }

    pub fn authorize(&self, headers: &HeaderMap) -> Result<(), GatewayError> {
        let supplied = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .ok_or(GatewayError::Auth)?;
        if token_matches(
            &self.bearer_token,
            self.bearer_token_len,
            supplied.as_bytes(),
        ) {
            Ok(())
        } else {
            Err(GatewayError::Auth)
        }
    }

    pub fn submit(&self, body: &[u8]) -> Result<InferenceResponse, GatewayError> {
        let deadline = Instant::now() + self.deadline;
        if body.is_empty() || body.len() > MAX_BODY_BYTES {
            return Err(GatewayError::Invalid("request body must be 1..=4096 bytes"));
        }
        let incoming: IncomingTransaction<'_> = serde_json::from_slice(body)
            .map_err(|_| GatewayError::Invalid("invalid transaction JSON"))?;
        let amount_micros = incoming.validate()?;
        if !self.is_ready() {
            return Err(GatewayError::DaemonStopped);
        }
        let payload_capacity = self.segment.layout.slot_bytes as usize - SLOT_PREFIX_BYTES;
        if incoming.worst_case_bytes() > payload_capacity {
            return Err(GatewayError::Invalid(
                "transaction exceeds fixed shared-memory slot",
            ));
        }
        if Instant::now() >= deadline {
            return Err(GatewayError::Timeout);
        }
        let request_id = random_request_id()?;
        let enqueue_ns = monotonic_ns()?;
        let position = self.reserve_slot(deadline)?;
        let mut reservation = SlotReservation {
            segment: Arc::clone(&self.segment),
            reclaims: Arc::clone(&self.reclaims),
            position,
            request_id,
            enqueue_ns,
            state: ReservationState::Held,
        };
        // The current FlatBuffers Rust API panics on allocator growth rather
        // than exposing try_* builders. The fixed-size allocator plus this
        // unwind boundary and RAII reservation guard prevents a lost slot.
        let serialization = catch_unwind(AssertUnwindSafe(|| {
            serialize_transaction(
                reservation.slot(),
                payload_capacity,
                request_id,
                &incoming,
                amount_micros,
            )
        }));
        let range = match serialization {
            Ok(Ok(range)) => range,
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(GatewayError::Internal("FlatBuffer serialization panicked")),
        };
        if Instant::now() >= deadline {
            return Err(GatewayError::Timeout);
        }
        reservation.publish_success(range);
        self.wait_or_reclaim(&mut reservation, deadline)
    }

    fn reserve_slot(&self, deadline: Instant) -> Result<u64, GatewayError> {
        loop {
            if !self.is_ready() {
                return Err(GatewayError::DaemonStopped);
            }
            let position = self
                .segment
                .header_u64(ENQUEUE_OFFSET)
                .load(Ordering::Relaxed);
            let slot = self
                .segment
                .slot((position as usize) & (self.segment.layout.slot_count as usize - 1));
            let sequence = atomic_u64(slot, SLOT_SEQUENCE_OFFSET)?.load(Ordering::Acquire);
            let difference = sequence.wrapping_sub(position) as i64;
            match difference.cmp(&0) {
                std::cmp::Ordering::Equal => {
                    if self
                        .segment
                        .header_u64(ENQUEUE_OFFSET)
                        .compare_exchange_weak(
                            position,
                            position + 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        return Ok(position);
                    }
                }
                std::cmp::Ordering::Less => return Err(GatewayError::Full),
                std::cmp::Ordering::Greater => {}
            }
            if Instant::now() >= deadline {
                return Err(GatewayError::Full);
            }
            std::hint::spin_loop();
        }
    }

    fn wait_or_reclaim(
        &self,
        reservation: &mut SlotReservation,
        deadline: Instant,
    ) -> Result<InferenceResponse, GatewayError> {
        let sequence =
            atomic_u64(reservation.slot(), SLOT_SEQUENCE_OFFSET).expect("validated slot atomic");
        while sequence.load(Ordering::Acquire) != reservation.position + 2 {
            if Instant::now() >= deadline {
                reservation.handoff();
                return Err(GatewayError::Timeout);
            }
            std::thread::sleep(PARK_SLICE);
        }
        // Release even an invalid daemon response; otherwise a corrupted
        // response permanently consumes capacity and becomes a denial of service.
        let response = read_response(reservation.slot(), reservation.request_id);
        reservation.release();
        response
    }
}

fn token_matches(expected: &[u8; MAX_TOKEN_BYTES], expected_len: usize, supplied: &[u8]) -> bool {
    if supplied.len() > MAX_TOKEN_BYTES {
        return false;
    }
    let mut padded_supplied = [0_u8; MAX_TOKEN_BYTES];
    padded_supplied[..supplied.len()].copy_from_slice(supplied);
    let matches =
        padded_supplied.ct_eq(expected) & (supplied.len() as u64).ct_eq(&(expected_len as u64));
    matches.into()
}

fn run_reclaimer(segment: Arc<SharedSegment>, queue: Arc<ReclaimQueue>) {
    let mut outstanding = Vec::<Reclaim>::with_capacity(queue.capacity);
    loop {
        {
            let mut pending = queue
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::swap(&mut outstanding, &mut *pending);
        }
        let mut index = 0;
        while index < outstanding.len() {
            let position = outstanding[index].position;
            let slot = segment.slot((position as usize) & (segment.layout.slot_count as usize - 1));
            let sequence = atomic_u64(slot, SLOT_SEQUENCE_OFFSET).expect("validated slot atomic");
            if sequence.load(Ordering::Acquire) == position + 2 {
                sequence.store(
                    position + segment.layout.slot_count as u64,
                    Ordering::Release,
                );
                outstanding.swap_remove(index);
            } else {
                index += 1;
            }
        }
        // Scan every outstanding position every tick. A permanently stuck early
        // request therefore cannot prevent a later completed request from reuse.
        std::thread::sleep(RECLAIMER_TICK);
    }
}

fn read_response(
    slot: *mut u8,
    expected_request_id: u64,
) -> Result<InferenceResponse, GatewayError> {
    let request_id = unsafe {
        u64::from_le_bytes(std::ptr::read_unaligned(
            slot.add(SLOT_REQUEST_ID).cast::<[u8; 8]>(),
        ))
    };
    if request_id != expected_request_id {
        return Err(GatewayError::Internal(
            "daemon response request_id mismatch",
        ));
    }
    let status = DaemonResponseStatus::try_from(read_u32(slot, SLOT_RESPONSE_STATUS))?;
    let score = read_f32(slot, SLOT_SCORE);
    if !score.is_finite() {
        return Err(GatewayError::Internal("daemon returned non-finite score"));
    }
    Ok(InferenceResponse {
        request_id,
        status,
        decision: read_u32(slot, SLOT_DECISION),
        score,
        completed_ns: unsafe {
            u64::from_le_bytes(std::ptr::read_unaligned(
                slot.add(SLOT_COMPLETE_NS).cast::<[u8; 8]>(),
            ))
        },
    })
}

fn random_request_id() -> Result<u64, GatewayError> {
    let mut value = 0u64;
    let result = unsafe {
        libc::getrandom(
            (&mut value as *mut u64).cast(),
            std::mem::size_of::<u64>(),
            0,
        )
    };
    if result == std::mem::size_of::<u64>() as isize && value != 0 {
        Ok(value)
    } else {
        Err(GatewayError::Internal("getrandom failed"))
    }
}

fn monotonic_ns() -> Result<u64, GatewayError> {
    let mut timestamp: libc::timespec = unsafe { std::mem::zeroed() };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) } != 0 {
        return Err(GatewayError::Internal("CLOCK_MONOTONIC failed"));
    }
    if timestamp.tv_sec < 0 || timestamp.tv_nsec < 0 || timestamp.tv_nsec >= 1_000_000_000 {
        return Err(GatewayError::Internal(
            "CLOCK_MONOTONIC returned invalid timestamp",
        ));
    }
    (timestamp.tv_sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|seconds| seconds.checked_add(timestamp.tv_nsec as u64))
        .ok_or(GatewayError::Internal("CLOCK_MONOTONIC overflow"))
}

#[derive(Clone)]
pub struct AppState {
    gateway: Arc<Gateway>,
}
pub fn app(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/health", axum::routing::get(health_handler))
        .route("/v1/transactions", axum::routing::post(submit_handler))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(AppState { gateway })
}

async fn health_handler(State(state): State<AppState>) -> Response {
    if state.gateway.is_ready() {
        (StatusCode::OK, Json(json!({"status":"ready"}))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"daemon_stopping"})),
        )
            .into_response()
    }
}

async fn submit_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GatewayError> {
    state.gateway.authorize(&headers)?;
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if !content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
    {
        return Err(GatewayError::Invalid(
            "content-type must be application/json",
        ));
    }
    let response = tokio::task::spawn_blocking(move || state.gateway.submit(&body))
        .await
        .map_err(|_| GatewayError::Internal("gateway worker panicked"))??;
    Ok((
        response.status.http_status(),
        Json(json!({"request_id":response.request_id, "status":response.status.as_u32(), "decision":response.decision, "score":response.score, "completed_ns":response.completed_ns})),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialized_segment(slots: u32, slot_bytes: u32) -> Vec<u8> {
        let mut bytes = vec![0; HEADER_BYTES + slots as usize * slot_bytes as usize];
        let base = bytes.as_mut_ptr();
        write_u32(base, MAGIC_OFFSET, MAGIC);
        write_u32(base, VERSION_OFFSET, VERSION);
        write_u32(base, HEADER_BYTES_OFFSET, HEADER_BYTES as u32);
        write_u32(base, SLOT_COUNT_OFFSET, slots);
        write_u32(base, SLOT_BYTES_OFFSET, slot_bytes);
        for offset in [ENQUEUE_OFFSET, DEQUEUE_OFFSET, READY_COUNT_OFFSET] {
            unsafe {
                std::ptr::write(base.add(offset).cast::<AtomicU64>(), AtomicU64::new(0));
            }
        }
        unsafe {
            std::ptr::write(
                base.add(SHUTDOWN_OFFSET).cast::<AtomicU32>(),
                AtomicU32::new(0),
            );
        }
        for index in 0..slots as usize {
            unsafe {
                std::ptr::write(
                    base.add(HEADER_BYTES + index * slot_bytes as usize)
                        .cast::<AtomicU64>(),
                    AtomicU64::new(index as u64),
                );
            }
        }
        bytes
    }

    #[test]
    fn abi_layout_has_exact_offsets_and_validates() {
        let mut bytes = initialized_segment(8, 512);
        let layout = validate_layout(bytes.as_mut_ptr(), bytes.len()).unwrap();
        assert_eq!(layout.slot_count, 8);
        assert_eq!(layout.slot_bytes, 512);
        assert_eq!(HEADER_BYTES, 320);
        assert_eq!(ENQUEUE_OFFSET, 64);
        assert_eq!(DEQUEUE_OFFSET, 128);
        assert_eq!(READY_COUNT_OFFSET, 192);
        assert_eq!(SHUTDOWN_OFFSET, 256);
        assert_eq!(SLOT_REQUEST_ID, 32);
        assert_eq!(SLOT_ENQUEUE_NS, 40);
        assert_eq!(SLOT_COMPLETE_NS, 48);
    }

    #[test]
    fn abi_rejects_ring_smaller_than_four_slots() {
        let mut bytes = initialized_segment(2, 512);
        assert!(matches!(
            validate_layout(bytes.as_mut_ptr(), bytes.len()),
            Err(GatewayError::Abi("invalid slot dimensions"))
        ));
    }

    #[test]
    fn fixed_allocator_serializes_without_a_vec() {
        let mut backing = vec![0u8; 512];
        let input = IncomingTransaction {
            transaction_id: "tx-1",
            account_id: "acct-1",
            amount_micros: 12_340_000,
            currency: "USD",
            occurred_at_ns: 5,
            merchant_category: Some("retail"),
        };
        let range = serialize_transaction(
            backing.as_mut_ptr(),
            backing.len() - SLOT_PREFIX_BYTES,
            9,
            &input,
            input.validate().unwrap(),
        )
        .unwrap();
        let data = &backing[range.offset..range.offset + range.size];
        assert!(data.len() < backing.len());
        assert!(transaction_fb::transaction_buffer_has_identifier(data));
        let decoded = transaction_fb::root_as_transaction(data).unwrap();
        assert_eq!(decoded.request_id(), 9);
        assert_eq!(decoded.transaction_id(), "tx-1");
        assert_eq!(decoded.amount_micros(), 12_340_000);
    }

    #[test]
    fn borrowed_json_and_validation_reject_unsafe_identifiers() {
        let input: IncomingTransaction<'_> = serde_json::from_slice(br#"{"transaction_id":"t1","account_id":"a1","amount_micros":1250000,"currency":"USD","occurred_at_ns":1}"#).unwrap();
        assert_eq!(input.validate().unwrap(), 1_250_000);
        let bad: IncomingTransaction<'_> = serde_json::from_slice(br#"{"transaction_id":"bad space","account_id":"a1","amount_micros":1,"currency":"USD","occurred_at_ns":1}"#).unwrap();
        assert!(matches!(bad.validate(), Err(GatewayError::Invalid(_))));
        let out_of_range: IncomingTransaction<'_> = serde_json::from_slice(br#"{"transaction_id":"t1","account_id":"a1","amount_micros":-9223372036854775808,"currency":"USD","occurred_at_ns":1}"#).unwrap();
        assert!(matches!(
            out_of_range.validate(),
            Err(GatewayError::Invalid(_))
        ));
    }

    #[test]
    fn response_status_is_closed_and_monotonic_clock_advances() {
        assert_eq!(
            DaemonResponseStatus::try_from(200).unwrap(),
            DaemonResponseStatus::Scored
        );
        assert_eq!(
            DaemonResponseStatus::try_from(503).unwrap(),
            DaemonResponseStatus::Unavailable
        );
        assert!(DaemonResponseStatus::try_from(42).is_err());
        let before = monotonic_ns().unwrap();
        std::thread::sleep(Duration::from_micros(1));
        assert!(monotonic_ns().unwrap() >= before);
    }

    #[test]
    fn bearer_token_comparison_checks_content_and_length() {
        let token = b"0123456789abcdef";
        let mut padded = [0_u8; MAX_TOKEN_BYTES];
        padded[..token.len()].copy_from_slice(token);
        assert!(token_matches(&padded, token.len(), token));
        assert!(!token_matches(&padded, token.len(), b"0123456789abcdeg"));
        assert!(!token_matches(&padded, token.len(), b"0123456789abcde"));
        assert!(!token_matches(
            &padded,
            token.len(),
            &[b'x'; MAX_TOKEN_BYTES + 1]
        ));
    }

    #[test]
    #[ignore = "requires a running daemon-owned System V segment"]
    fn attaches_to_live_daemon_segment() {
        let key = std::env::var("FRAUD_SHM_KEY").expect("set FRAUD_SHM_KEY for live IPC test");
        let segment = SharedSegment::attach(key.parse::<i64>().unwrap() as libc::key_t).unwrap();
        assert!(segment.layout().slot_count.is_power_of_two());
    }
}
