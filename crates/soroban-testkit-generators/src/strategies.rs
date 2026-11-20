use core::sync::atomic::{AtomicU64, Ordering};

pub use crate::cost_model::{
    contract_cost_model, contract_cost_model_with_config, contract_cost_model_with_multipliers,
    ContractCostComponent, ContractCostEntry, ContractCostModel, ContractCostModelConfig,
    CostDimension, BASELINE_COST_ENTRIES,
};
pub use crate::crypto::{
    ed25519_signature_pair, malleated_signature, secp256k1_signature_pair, Ed25519SignaturePair,
    Secp256k1SignaturePair,
};
pub use crate::executables::{contract_executable, contract_executable_wasm_hash, deployer_salt};
pub use crate::storage_tier::{
    mixed_tier_allocations, mixed_tier_allocations_with_ttl, mixed_tier_key_pairs, storage_tier,
    storage_tier_with_default_ttl, storage_tier_with_ttl, MixedTierAllocation,
    MixedTierTtlAllocation, StorageTierWithTtl,
};

use proptest::collection::SizeRange;
use proptest::prelude::*;
use soroban_sdk::xdr::{
    AccountId, ContractId, Hash, Limits, MuxedAccount, MuxedAccountMed25519, MuxedEd25519Account,
    PublicKey, ScAddress, ScMap, ScMapEntry, ScVal, ScVec, Uint256, VecM, WriteXdr as _,
};
use soroban_sdk::{testutils::Address as _, Address, Env, IntoVal, Symbol, TryFromVal as _, Val};

/// Generate arbitrary Soroban invocation arguments, including values whose
/// object handles are owned by `env`. `typed` contains valid arguments for a
/// particular contract method; the strategy inserts zero or more rogue values
/// at arbitrary positions so the same client call exercises type validation.
///
/// The primitive branches shrink to zero, `false`, and the empty symbol. Host
/// objects and errors are included as type mismatches and disappear as the
/// list shrinks.
pub fn fuzzed_invocation_args(
    env: &Env,
    typed: Vec<Val>,
    max_rogue: u32,
) -> impl Strategy<Value = Vec<Val>> {
    let env = env.clone();
    let int_env = env.clone();
    let bool_env = env.clone();
    let uint_env = env.clone();
    let address_env = env.clone();
    let rogue = prop_oneof![
        4 => any::<i32>().prop_map(move |value| value.into_val(&int_env)),
        4 => any::<bool>().prop_map(move |value| value.into_val(&bool_env)),
        2 => any::<u32>().prop_map(move |value| value.into_val(&uint_env)),
        2 => Just(Symbol::new(&env, "rogue_symbol").into_val(&env)),
        1 => Just(soroban_sdk::Error::from_contract_error(1).into_val(&env)),
        1 => any::<u8>().prop_map(move |_| Address::generate(&address_env).into_val(&address_env)),
    ];
    (
        prop::collection::vec(rogue, 0..=max_rogue as usize),
        any::<Vec<usize>>(),
    )
        .prop_map(move |(mut rogue, order)| {
            let mut args = typed.clone();
            // Interleave rogue values at deterministic positions derived from
            // the order vector. This keeps the strategy shrinkable.
            for (index, value) in rogue.drain(..).enumerate() {
                let position = order.get(index).copied().unwrap_or(index).min(args.len());
                args.insert(position, value);
            }
            args
        })
}

/// The two address formats Soroban accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressType {
    /// An ed25519 account, the `G...` form.
    Account,
    /// A contract id, the `C...` form.
    Contract,
}

/// A generated Soroban address: 32 raw bytes tagged with the format they
/// belong to.
///
/// Both formats are opaque 32-byte values on the wire, so the tag is what
/// separates an ed25519 public key from a contract id. Turn it into a host
/// [`Address`] with [`AddressBytes::to_address`], or into its XDR form with
/// `ScAddress::from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AddressBytes {
    address_type: AddressType,
    bytes: [u8; 32],
}

impl AddressBytes {
    /// Build an address from its format tag and bytes.
    ///
    /// Crate-internal: the public API generates addresses, and a hand-built one
    /// is only useful to a test that needs a predictable value.
    #[cfg(test)]
    pub(crate) fn from_parts(address_type: AddressType, bytes: [u8; 32]) -> Self {
        Self {
            address_type,
            bytes,
        }
    }

    /// The format this address was generated as.
    pub fn address_type(&self) -> AddressType {
        self.address_type
    }

    /// The raw 32 bytes, as an ed25519 public key or a contract id.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// The raw 32 bytes, consuming the address.
    pub fn into_bytes(self) -> [u8; 32] {
        self.bytes
    }

    /// Rebuild the address as a host value owned by `env`.
    pub fn to_address(self, env: &Env) -> Address {
        Address::try_from_val(env, &ScAddress::from(self))
            .expect("a 32-byte address is a valid ScAddress")
    }

    /// The address as its wire form: an `ScAddress`, without an environment.
    pub fn to_sc_address(self) -> ScAddress {
        ScAddress::from(self)
    }

    /// The address as the strkey a test recognises: `G...` for an account,
    /// `C...` for a contract id.
    pub fn to_strkey(self) -> String {
        match self.address_type {
            AddressType::Account => {
                format!("{}", stellar_strkey::ed25519::PublicKey(self.bytes))
            }
            AddressType::Contract => format!("{}", stellar_strkey::Contract(self.bytes)),
        }
    }

    /// Parse a `G...` or `C...` strkey back into its bytes and format tag.
    ///
    /// Muxed (`M...`) strkeys carry a 64-bit id that an `ScAddress` cannot
    /// hold, so they are refused here; use [`MuxedAddress::from_strkey`] for
    /// those. Anything that is not a valid, checksummed address strkey is
    /// refused as well.
    pub fn from_strkey(strkey: &str) -> Option<Self> {
        if let Ok(contract) = stellar_strkey::Contract::from_string(strkey) {
            return Some(Self {
                address_type: AddressType::Contract,
                bytes: contract.0,
            });
        }
        if let Ok(public_key) = stellar_strkey::ed25519::PublicKey::from_string(strkey) {
            return Some(Self {
                address_type: AddressType::Account,
                bytes: public_key.0,
            });
        }
        None
    }

    /// Recover the bytes and format of an account or contract `ScAddress`.
    ///
    /// The other XDR variants — a muxed account, a claimable balance, a
    /// liquidity pool — are not a 32-byte account or contract id, so they have
    /// no `AddressBytes` equivalent and yield `None`.
    pub fn from_sc_address(sc_address: &ScAddress) -> Option<Self> {
        match sc_address {
            ScAddress::Account(AccountId(PublicKey::PublicKeyTypeEd25519(Uint256(bytes)))) => {
                Some(Self {
                    address_type: AddressType::Account,
                    bytes: *bytes,
                })
            }
            ScAddress::Contract(ContractId(Hash(bytes))) => Some(Self {
                address_type: AddressType::Contract,
                bytes: *bytes,
            }),
            _ => None,
        }
    }
}

impl From<AddressBytes> for ScAddress {
    fn from(address: AddressBytes) -> Self {
        match address.address_type {
            AddressType::Account => ScAddress::Account(AccountId(PublicKey::PublicKeyTypeEd25519(
                Uint256(address.bytes),
            ))),
            AddressType::Contract => ScAddress::Contract(ContractId(Hash(address.bytes))),
        }
    }
}

/// Hands every generated address a distinct suffix, so two cases in one run
/// cannot collide even if their random prefixes do.
static NEXT_ADDRESS_NONCE: AtomicU64 = AtomicU64::new(1);

/// Generates either an account or a contract address.
pub fn address() -> impl Strategy<Value = AddressBytes> {
    prop_oneof![account_address(), contract_address()]
}

/// Generates an ed25519 account address.
pub fn account_address() -> impl Strategy<Value = AddressBytes> {
    address_of_type(AddressType::Account)
}

/// Generates a contract address.
pub fn contract_address() -> impl Strategy<Value = AddressBytes> {
    address_of_type(AddressType::Contract)
}

/// Generates an address of the requested format.
pub fn address_of_type(address_type: AddressType) -> impl Strategy<Value = AddressBytes> {
    any::<[u8; 32]>().prop_map(move |mut bytes| {
        let nonce = NEXT_ADDRESS_NONCE.fetch_add(1, Ordering::Relaxed);
        bytes[24..].copy_from_slice(&nonce.to_be_bytes());
        AddressBytes {
            address_type,
            bytes,
        }
    })
}

/// Generates a raw XDR [`ScAddress`]: an ed25519 account or a contract id.
///
/// The existing address strategies return an [`AddressBytes`], which is what a
/// contract call takes. Low-level XDR work — decoding a ledger entry, comparing
/// a value against `ScAddress::Account` and `ScAddress::Contract`, writing a
/// parser for the two arms — needs the `ScAddress` itself, so this strategy
/// yields the encoded form. Both arms shrink towards the zeroed 32-byte payload,
/// so a failing account case minimises to the all-zero ed25519 key and a failing
/// contract case to the zeroed hash.
pub fn sc_address() -> impl Strategy<Value = ScAddress> {
    prop_oneof![sc_account_address(), sc_contract_address()]
}

/// Generates an [`ScAddress::Account`] with an ed25519 public key.
///
/// The payload shrinks towards the all-zero key.
pub fn sc_account_address() -> impl Strategy<Value = ScAddress> {
    any::<[u8; 32]>().prop_map(|bytes| {
        ScAddress::Account(AccountId(PublicKey::PublicKeyTypeEd25519(Uint256(bytes))))
    })
}

/// Generates an [`ScAddress::Contract`] with a 32-byte contract hash.
///
/// The hash shrinks towards zero, which is what makes a failing contract
/// address report the smallest id that still reproduces it.
pub fn sc_contract_address() -> impl Strategy<Value = ScAddress> {
    any::<[u8; 32]>().prop_map(|bytes| ScAddress::Contract(ContractId(Hash(bytes))))
}

/// Generates an [`ScAddress::MuxedAccount`]: an ed25519 key and a 64-bit id.
///
/// The key and id shrink towards zero, so a failing muxed case reports the
/// smallest key and id that still reproduce it.
pub fn sc_muxed_account_address() -> impl Strategy<Value = ScAddress> {
    (any::<[u8; 32]>(), any::<u64>()).prop_map(|(bytes, id)| {
        ScAddress::MuxedAccount(MuxedEd25519Account {
            id,
            ed25519: Uint256(bytes),
        })
    })
}

/// Generates a vector of mixed [`ScAddress`] variants — account, contract and
/// muxed account — with a length drawn from `len`.
///
/// ```text
/// proptest! {
///     #[test]
///     fn a_decoder_handles_every_address_kind(addresses in sc_address_collection(0..=8)) {
///         // addresses: Vec<ScAddress>, possibly empty
///     }
/// }
/// ```
pub fn sc_address_collection(len: impl Into<SizeRange>) -> impl Strategy<Value = Vec<ScAddress>> {
    prop::collection::vec(
        prop_oneof![
            sc_account_address(),
            sc_contract_address(),
            sc_muxed_account_address()
        ],
        len,
    )
}

/// Generates a muxed ed25519 account: an account address plus a 64-bit id.
///
/// Muxed addresses are a classic Stellar account concept (`M...` strkeys), not
/// a Soroban `ScAddress` variant, so the id is not representable as a contract
/// address. The strategy yields both parts together so a test can check that
/// the id survives the strkey round trip and is dropped — or threaded through —
/// when the value becomes an `ScAddress`. The key and the id both shrink
/// towards zero.
pub fn muxed_address() -> impl Strategy<Value = MuxedAddress> {
    (any::<[u8; 32]>(), any::<u64>()).prop_map(|(ed25519, id)| MuxedAddress::new(ed25519, id))
}

/// A muxed ed25519 account: an ed25519 public key and a 64-bit multiplex id.
///
/// `to_strkey` renders the `M...` form, `account_strkey` the `G...` form of the
/// same key, and `to_sc_address` the [`ScAddress::Account`] a Soroban contract
/// sees. Shrinking is by component: the key towards zero, the id towards zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MuxedAddress {
    ed25519: [u8; 32],
    id: u64,
}

impl MuxedAddress {
    /// Pair a public key with a muxed id.
    pub fn new(ed25519: [u8; 32], id: u64) -> Self {
        Self { ed25519, id }
    }

    /// The ed25519 public key behind the muxed address.
    pub fn ed25519(&self) -> &[u8; 32] {
        &self.ed25519
    }

    /// The 64-bit multiplex id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The underlying account, without the multiplex id.
    pub fn account(&self) -> AddressBytes {
        AddressBytes {
            address_type: AddressType::Account,
            bytes: self.ed25519,
        }
    }

    /// The underlying account as an XDR [`AccountId`].
    pub fn account_id(&self) -> AccountId {
        AccountId(PublicKey::PublicKeyTypeEd25519(Uint256(self.ed25519)))
    }

    /// The XDR address that keeps both parts: [`ScAddress::MuxedAccount`].
    pub fn to_sc_address(&self) -> ScAddress {
        ScAddress::MuxedAccount(MuxedEd25519Account {
            id: self.id,
            ed25519: Uint256(self.ed25519),
        })
    }

    /// The plain [`ScAddress::Account`] form, which drops the muxed id.
    pub fn to_account_sc_address(&self) -> ScAddress {
        ScAddress::Account(self.account_id())
    }

    /// Rebuild the underlying account as a host value owned by `env`.
    pub fn to_address(&self, env: &Env) -> Address {
        self.account().to_address(env)
    }

    /// The XDR muxed-account form, which does carry the id.
    pub fn to_muxed_account(&self) -> MuxedAccount {
        MuxedAccount::MuxedEd25519(MuxedAccountMed25519 {
            id: self.id,
            ed25519: Uint256(self.ed25519),
        })
    }

    /// The `M...` strkey for this account and id.
    pub fn to_strkey(&self) -> String {
        format!(
            "{}",
            stellar_strkey::ed25519::MuxedAccount {
                ed25519: self.ed25519,
                id: self.id,
            }
        )
    }

    /// The `G...` strkey of the underlying account, without the id.
    pub fn account_strkey(&self) -> String {
        self.account().to_strkey()
    }

    /// Parse an `M...` strkey back into its key and id.
    pub fn from_strkey(strkey: &str) -> Option<Self> {
        let muxed = stellar_strkey::ed25519::MuxedAccount::from_string(strkey).ok()?;
        Some(Self {
            ed25519: muxed.ed25519,
            id: muxed.id,
        })
    }
}

impl From<MuxedAddress> for ScAddress {
    fn from(muxed: MuxedAddress) -> Self {
        muxed.to_sc_address()
    }
}

impl From<MuxedAddress> for MuxedAccount {
    fn from(muxed: MuxedAddress) -> Self {
        muxed.to_muxed_account()
    }
}

pub fn token_amount() -> impl Strategy<Value = i128> {
    0i128..=1_000_000_000_000_000i128
}

pub fn ledger_sequence() -> impl Strategy<Value = u32> {
    1u32..=10_000_000u32
}

pub fn timestamp() -> impl Strategy<Value = u64> {
    1_600_000_000u64..=2_000_000_000u64
}

/// The entry budget [`fuzzed_scval_map`] and [`fuzzed_scval_vec`] use when the
/// caller has no opinion: large enough to cross the host's element limits in a
/// handful of cases, small enough that printed payloads stay readable.
pub const DEFAULT_FUZZ_ENTRIES: u32 = 64;

/// A scalar a fuzzed map key or vector element can be built from.
///
/// Only the small, cheap variants are generated: a fuzz corpus is judged by
/// what the parser does with the *shape* of an object, and a nested tree three
/// objects deep says nothing a one-level object does not.
fn fuzz_scalar() -> impl Strategy<Value = ScVal> {
    prop_oneof![
        Just(ScVal::Void),
        any::<bool>().prop_map(ScVal::Bool),
        any::<u32>().prop_map(ScVal::U32),
        any::<i32>().prop_map(ScVal::I32),
        any::<u64>().prop_map(ScVal::U64),
        any::<i64>().prop_map(ScVal::I64),
    ]
}

/// A value a fuzzed map can carry, including the two null object forms.
///
/// `ScVal::Map(None)` and `ScVal::Vec(None)` are the XDR spellings of a null
/// object. A parser that unwraps `Option<ScMap>` without checking it is a real
/// deserialization bug, so the corpus includes them.
fn fuzz_value() -> impl Strategy<Value = ScVal> {
    prop_oneof![
        fuzz_scalar(),
        Just(ScVal::Vec(None)),
        Just(ScVal::Map(None)),
    ]
}

/// Unordered `(key, value)` pairs, bounded by `max_entries`.
fn fuzz_pairs(max_entries: u32) -> impl Strategy<Value = Vec<(ScVal, ScVal)>> {
    prop::collection::vec((fuzz_scalar(), fuzz_value()), 0..=max_entries as usize)
}

/// Builds an `ScMap`, ordering and de-duplicating the keys when `sorted`.
///
/// The unsorted branch keeps generation order and therefore keeps duplicate
/// keys too: both are shapes a contract parser has to reject rather than
/// assume away, and the host validates a `ScMap` before a contract ever sees
/// one.
fn sc_map(pairs: Vec<(ScVal, ScVal)>, sorted: bool) -> ScMap {
    let mut pairs = pairs;
    if sorted {
        pairs.sort_by(|left, right| left.0.cmp(&right.0));
        pairs.dedup_by(|left, right| left.0 == right.0);
    }
    let entries: Vec<ScMapEntry> = pairs
        .into_iter()
        .map(|(key, val)| ScMapEntry { key, val })
        .collect();
    ScMap(VecM::try_from(entries).expect("a generated map fits the XDR length bound"))
}

/// Generates an `ScVal::Map` with at most `max_entries` entries.
///
/// Half the cases are sorted and key-unique, which is the shape the host
/// accepts; the other half keep generation order, which is what a parser sees
/// when a caller hands it an XDR map that was never canonicalised. Shrinking
/// removes entries and simplifies keys, so a failing call shrinks towards the
/// smallest map that still fails.
///
/// ```text
/// proptest! {
///     #[test]
///     fn a_parser_never_panics_on_a_map(map in fuzzed_scval_map(16)) {
///         // map: ScVal::Map(Some(..)) bounded by 16 entries
///     }
/// }
/// ```
pub fn fuzzed_scval_map(max_entries: u32) -> impl Strategy<Value = ScVal> {
    (fuzz_pairs(max_entries), any::<bool>())
        .prop_map(|(pairs, sorted)| ScVal::Map(Some(sc_map(pairs, sorted))))
}

/// Generates an `ScVal::Vec` with at most `max_entries` elements.
///
/// Pass the host's element limit to stay inside it, or a multiple of it to
/// generate the oversized vectors a parser has to refuse instead of allocating
/// for. Elements include the null object forms, so a parser that unwraps
/// `Option<ScVec>` is exercised as well.
pub fn fuzzed_scval_vec(max_entries: u32) -> impl Strategy<Value = ScVal> {
    prop::collection::vec(fuzz_value(), 0..=max_entries as usize).prop_map(|items| {
        let vec = VecM::try_from(items).expect("a generated vector fits the XDR length bound");
        ScVal::Vec(Some(ScVec(vec)))
    })
}

/// Generates truncated XDR: the bytes of a real map or vector, cut at an
/// arbitrary offset.
///
/// A cut inside the four-byte type discriminant leaves a truncated tag; a cut
/// inside an entry leaves a length that promises more bytes than follow. Both
/// are what a contract sees when a caller sends a short payload, and neither
/// may panic a parser that returns an error instead.
///
/// The strategy is built to shrink: proptest shortens the cut towards zero, so
/// a failing deserialization reports the smallest prefix that still fails
/// rather than the whole object.
pub fn malformed_object_payload() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![fuzzed_scval_map(4), fuzzed_scval_vec(4)]
        .prop_flat_map(|object| {
            let bytes = object
                .to_xdr(Limits::none())
                .expect("a generated ScVal encodes as XDR");
            let length = bytes.len();
            (Just(bytes), 0..length)
        })
        .prop_map(|(mut bytes, keep)| {
            bytes.truncate(keep);
            bytes
        })
}

#[cfg(test)]
mod tests {
    use super::{
        account_address, address, address_of_type, contract_address, ledger_sequence, timestamp,
        token_amount, AddressType,
    };
    use proptest::prelude::*;
    use proptest::test_runner::TestRunner;
    use soroban_sdk::xdr::ScAddress;
    use soroban_sdk::Env;
    use std::collections::HashSet;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn token_amounts_stay_non_negative(amount in token_amount()) {
            prop_assert!(amount >= 0);
        }

        #[test]
        fn token_amounts_stay_below_the_supply_cap(amount in token_amount()) {
            prop_assert!(amount <= 1_000_000_000_000_000);
        }

        #[test]
        fn ledger_sequences_are_never_zero(seq in ledger_sequence()) {
            prop_assert!(seq >= 1);
            prop_assert!(seq <= 10_000_000);
        }

        #[test]
        fn timestamps_fall_inside_the_stellar_era(ts in timestamp()) {
            prop_assert!(ts >= 1_600_000_000);
            prop_assert!(ts <= 2_000_000_000);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn account_addresses_are_32_byte_ed25519_keys(address in account_address()) {
            prop_assert_eq!(address.address_type(), AddressType::Account);
            prop_assert_eq!(address.as_bytes().len(), 32);
            prop_assert!(matches!(ScAddress::from(address), ScAddress::Account(_)));
        }

        #[test]
        fn contract_addresses_are_32_byte_contract_ids(address in contract_address()) {
            prop_assert_eq!(address.address_type(), AddressType::Contract);
            prop_assert_eq!(address.as_bytes().len(), 32);
            prop_assert!(matches!(ScAddress::from(address), ScAddress::Contract(_)));
        }

        #[test]
        fn the_type_filter_only_yields_the_requested_format(address in address_of_type(AddressType::Contract)) {
            prop_assert_eq!(address.address_type(), AddressType::Contract);
        }

        #[test]
        fn the_default_strategy_yields_one_of_the_two_formats(address in address()) {
            prop_assert!(matches!(
                address.address_type(),
                AddressType::Account | AddressType::Contract
            ));
        }
    }

    #[test]
    fn generated_addresses_convert_to_host_addresses() {
        let env = Env::default();
        let mut runner = TestRunner::new(ProptestConfig::with_cases(32));
        runner
            .run(
                &(account_address(), contract_address()),
                |(account, contract)| {
                    let account = ScAddress::from(account.to_address(&env));
                    let contract = ScAddress::from(contract.to_address(&env));
                    prop_assert!(matches!(account, ScAddress::Account(_)));
                    prop_assert!(matches!(contract, ScAddress::Contract(_)));
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn a_single_run_never_generates_a_duplicate_address() {
        let mixed = prop_oneof![account_address(), contract_address()];
        let mut runner = TestRunner::new(ProptestConfig::with_cases(32));
        runner
            .run(&proptest::collection::vec(mixed, 64), |addresses| {
                let unique: HashSet<_> =
                    addresses.iter().map(|address| address.as_bytes()).collect();
                prop_assert_eq!(unique.len(), addresses.len());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn strategies_compose_into_tuples() {
        let mut runner = TestRunner::default();
        runner
            .run(&(token_amount(), ledger_sequence()), |(amount, seq)| {
                prop_assert!(amount >= 0);
                prop_assert!(seq >= 1);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn filtered_strategy_only_yields_sufficient_balances() {
        let strategy = (token_amount(), token_amount())
            .prop_filter("sender must have sufficient balance", |(a, b)| a >= b);

        let mut runner = TestRunner::default();
        runner
            .run(&strategy, |(sender_balance, amount)| {
                prop_assert!(sender_balance >= amount);
                Ok(())
            })
            .unwrap();
    }
}

#[cfg(test)]
mod object_fuzz_tests {
    use super::{fuzzed_scval_map, fuzzed_scval_vec, malformed_object_payload};
    use proptest::prelude::*;
    use proptest::test_runner::{TestError, TestRunner};
    use soroban_sdk::xdr::{Limits, ReadXdr as _, ScMap, ScVal, WriteXdr as _};

    /// The keys of a map in the order the ledger would hand them over.
    fn keys(map: &ScMap) -> Vec<&ScVal> {
        map.0.iter().map(|entry| &entry.key).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn a_fuzzed_map_stays_within_its_entry_budget(map in fuzzed_scval_map(6)) {
            let ScVal::Map(Some(map)) = &map else {
                return Err(TestCaseError::fail("fuzzed_scval_map did not yield a map"));
            };

            prop_assert!(map.0.len() <= 6);

            // Whatever the ordering, the object is real XDR: the encoder and
            // decoder round-trip the same entries, so a parser is being handed
            // a payload it could receive on chain rather than random bytes.
            let wrapped = ScVal::Map(Some(map.clone()));
            let bytes = wrapped.to_xdr(Limits::none()).expect("a map encodes");
            prop_assert_eq!(
                ScVal::from_xdr(bytes.as_slice(), Limits::none()).expect("a map decodes"),
                wrapped
            );
        }

        #[test]
        fn a_fuzzed_vector_stays_within_its_entry_budget(vector in fuzzed_scval_vec(6)) {
            let ScVal::Vec(Some(vector)) = &vector else {
                return Err(TestCaseError::fail("fuzzed_scval_vec did not yield a vector"));
            };

            prop_assert!(vector.0.len() <= 6);

            let wrapped = ScVal::Vec(Some(vector.clone()));
            let bytes = wrapped.to_xdr(Limits::none()).expect("a vector encodes");
            prop_assert_eq!(
                ScVal::from_xdr(bytes.as_slice(), Limits::none()).expect("a vector decodes"),
                wrapped
            );
        }

        #[test]
        fn an_oversized_budget_generates_larger_vectors(vector in fuzzed_scval_vec(4_096)) {
            let ScVal::Vec(Some(vector)) = &vector else {
                return Err(TestCaseError::fail("fuzzed_scval_vec did not yield a vector"));
            };

            // The caller names the limit, so a test that wants the size a host
            // refuses asks for more entries than the host accepts.
            prop_assert!(vector.0.len() <= 4_096);
        }

        #[test]
        fn a_malformed_payload_never_decodes_as_a_complete_value(
            payload in malformed_object_payload()
        ) {
            prop_assert!(
                ScVal::from_xdr(payload.as_slice(), Limits::none()).is_err(),
                "a {} byte payload decoded even though it was truncated",
                payload.len()
            );
        }
    }

    #[test]
    fn the_map_strategy_produces_both_canonical_and_unsorted_maps() {
        let mut runner = TestRunner::new(ProptestConfig::with_cases(64));
        // `run` takes `Fn`, so the tallies live outside the closure.
        let sorted = std::cell::Cell::new(0usize);
        let unsorted = std::cell::Cell::new(0usize);

        runner
            .run(&fuzzed_scval_map(5), |value| {
                let ScVal::Map(Some(map)) = &value else {
                    return Ok(());
                };
                let keys = keys(map);
                if keys.windows(2).all(|pair| pair[0] < pair[1]) {
                    sorted.set(sorted.get() + 1);
                } else {
                    unsorted.set(unsorted.get() + 1);
                }
                Ok(())
            })
            .expect("the map strategy runs");

        // Both halves are reachable: a suite that only ever saw sorted maps
        // would never prove anything about a host that refuses the others.
        assert!(
            sorted.get() > 0,
            "no canonical map was generated in 64 cases"
        );
        assert!(
            unsorted.get() > 0,
            "no unsorted map was generated in 64 cases"
        );
    }

    #[test]
    fn an_unsorted_map_breaks_a_parser_that_assumes_ledger_order() {
        // This is the bug the strategy exists to find: a parser that assumes the
        // ledger handed it sorted keys and indexes into them. The corpus below
        // has to produce a map that violates the assumption, and proptest
        // reports the smallest one it found.
        // The failure is the point of this test, so there is nothing worth
        // persisting as a regression seed.
        let config = ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::with_cases(64)
        };
        let mut runner = TestRunner::new(config);
        let outcome = runner.run(&fuzzed_scval_map(5), |value| {
            let ScVal::Map(Some(map)) = &value else {
                return Ok(());
            };
            let keys = keys(map);
            prop_assert!(
                keys.windows(2).all(|pair| pair[0] < pair[1]),
                "keys arrived out of ledger order"
            );
            Ok(())
        });

        assert!(
            matches!(outcome, Err(TestError::Fail(..))),
            "the corpus never produced an unsorted map, so the property never failed"
        );
    }

    #[test]
    fn a_failing_deserialization_shrinks_towards_a_minimal_payload() {
        // A parser that insists on a complete value fails on every truncated
        // payload, and the reported case is the smallest prefix rather than the
        // object it was cut from.
        let config = ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::with_cases(64)
        };
        let mut runner = TestRunner::new(config);
        let outcome = runner.run(&malformed_object_payload(), |payload| {
            prop_assert!(
                ScVal::from_xdr(payload.as_slice(), Limits::none()).is_ok(),
                "{} byte payload was rejected",
                payload.len()
            );
            Ok(())
        });

        match outcome {
            Err(TestError::Fail(_, payload)) => assert!(
                payload.len() < 4,
                "the smallest failing payload should be a truncated tag, got {} bytes",
                payload.len()
            ),
            other => panic!("expected a shrunk counterexample, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod sc_address_tests {
    use super::{
        muxed_address, sc_account_address, sc_address, sc_address_collection, sc_contract_address,
        sc_muxed_account_address, AddressBytes, AddressType, MuxedAddress,
    };
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use soroban_sdk::xdr::{
        AccountId, ContractId, Hash, MuxedAccount, PublicKey, ScAddress, Uint256,
    };
    use soroban_sdk::{Address, Env, TryFromVal as _};
    use std::cell::Cell;

    /// Drive proptest's shrinker by hand: keep simplifying while `fails` holds.
    fn minimise<T: Strategy>(strategy: T, fails: impl Fn(&T::Value) -> bool) -> T::Value
    where
        T::Value: Clone,
    {
        let mut runner = TestRunner::deterministic();
        let mut tree = loop {
            let tree = strategy.new_tree(&mut runner).unwrap();
            if fails(&tree.current()) {
                break tree;
            }
        };
        let mut smallest = tree.current();
        loop {
            if fails(&tree.current()) {
                smallest = tree.current();
                if !tree.simplify() {
                    break;
                }
            } else if !tree.complicate() {
                break;
            }
        }
        smallest
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn a_generated_sc_address_is_a_well_formed_variant(address in sc_address()) {
            match address {
                ScAddress::Account(AccountId(PublicKey::PublicKeyTypeEd25519(Uint256(bytes)))) => {
                    prop_assert_eq!(bytes.len(), 32);
                }
                ScAddress::Contract(ContractId(Hash(bytes))) => {
                    prop_assert_eq!(bytes.len(), 32);
                }
                other => prop_assert!(false, "unexpected ScAddress variant: {other:?}"),
            }
        }

        #[test]
        fn a_muxed_sc_address_carries_its_key_and_id(
            address in sc_muxed_account_address(),
        ) {
            let ScAddress::MuxedAccount(muxed) = address else {
                return Err(TestCaseError::fail("expected a muxed account address"));
            };
            prop_assert_eq!(muxed.ed25519.0.len(), 32);
        }

        #[test]
        fn the_two_arms_only_yield_their_own_variant(
            account in sc_account_address(),
            contract in sc_contract_address(),
        ) {
            prop_assert!(matches!(account, ScAddress::Account(_)));
            prop_assert!(matches!(contract, ScAddress::Contract(_)));
        }
    }

    #[test]
    fn an_sc_address_round_trips_through_the_host_address_and_its_strkey() {
        let env = Env::default();
        let mut runner = TestRunner::new(ProptestConfig::with_cases(64));
        runner
            .run(&sc_address(), |sc_address| {
                let host = Address::try_from_val(&env, &sc_address)
                    .expect("a generated ScAddress is a host address");
                let bytes = AddressBytes::from_sc_address(&sc_address)
                    .expect("a generated address has a 32-byte payload");
                let strkey = bytes.to_strkey();
                let parsed =
                    AddressBytes::from_strkey(&strkey).expect("a generated address strkey parses");

                prop_assert_eq!(parsed, bytes);
                prop_assert_eq!(ScAddress::from(host), sc_address);
                Ok(())
            })
            .expect("every generated address round trips");
    }

    #[test]
    fn an_account_strkey_and_a_contract_strkey_are_told_apart() {
        let mut runner = TestRunner::new(ProptestConfig::with_cases(64));
        runner
            .run(
                &(
                    super::account_address(),
                    super::contract_address(),
                    super::address_of_type(AddressType::Account),
                ),
                |(account, contract, filtered)| {
                    prop_assert!(account.to_strkey().starts_with('G'));
                    prop_assert!(contract.to_strkey().starts_with('C'));

                    let parsed = AddressBytes::from_strkey(&account.to_strkey())
                        .expect("an account strkey parses");
                    prop_assert_eq!(parsed.address_type(), AddressType::Account);
                    prop_assert_eq!(parsed.as_bytes(), account.as_bytes());

                    prop_assert_eq!(
                        AddressBytes::from_sc_address(&filtered.to_sc_address())
                            .map(|bytes| bytes.address_type()),
                        Some(AddressType::Account)
                    );
                    prop_assert!(AddressBytes::from_strkey("not a strkey").is_none());
                    Ok(())
                },
            )
            .expect("every generated address keeps its format");
    }

    #[test]
    fn a_contract_sc_address_shrinks_to_the_zeroed_hash() {
        let smallest = minimise(sc_contract_address(), |address| match address {
            ScAddress::Contract(ContractId(Hash(hash))) => hash[0] >= 0x10,
            _ => false,
        });

        let ScAddress::Contract(ContractId(Hash(hash))) = smallest else {
            panic!("the failing case must stay a contract address");
        };
        assert_eq!(hash[0], 0x10);
        assert!(hash[1..].iter().all(|byte| *byte == 0), "{hash:?}");
    }

    #[test]
    fn an_account_sc_address_shrinks_to_the_zeroed_key() {
        let smallest = minimise(sc_account_address(), |address| match address {
            ScAddress::Account(AccountId(PublicKey::PublicKeyTypeEd25519(Uint256(bytes)))) => {
                bytes[0] >= 0x10
            }
            _ => false,
        });

        let ScAddress::Account(AccountId(PublicKey::PublicKeyTypeEd25519(Uint256(bytes)))) =
            smallest
        else {
            panic!("the failing case must stay an account address");
        };
        assert_eq!(bytes[0], 0x10);
        assert!(bytes[1..].iter().all(|byte| *byte == 0), "{bytes:?}");
    }

    #[test]
    fn a_muxed_address_round_trips_through_its_strkey() {
        let mut runner = TestRunner::new(ProptestConfig::with_cases(64));
        runner
            .run(&muxed_address(), |muxed| {
                let strkey = muxed.to_strkey();
                prop_assert!(strkey.starts_with('M'));
                prop_assert_eq!(MuxedAddress::from_strkey(&strkey), Some(muxed));

                // The id is not part of the account form, so the M and G
                // strkeys describe the same key but not the same value.
                prop_assert!(muxed.account_strkey().starts_with('G'));
                prop_assert!(MuxedAddress::from_strkey(&muxed.account_strkey()).is_none());
                prop_assert_eq!(muxed.account().to_strkey(), muxed.account_strkey());

                // The muxed form keeps the id; the plain account form drops it.
                let ScAddress::MuxedAccount(encoded) = muxed.to_sc_address() else {
                    prop_assert!(false, "a muxed address encodes as a muxed ScAddress");
                    return Ok(());
                };
                prop_assert_eq!(encoded.id, muxed.id());
                prop_assert_eq!(encoded.ed25519.0, *muxed.ed25519());
                prop_assert_eq!(
                    muxed.to_account_sc_address(),
                    ScAddress::Account(muxed.account_id())
                );

                let MuxedAccount::MuxedEd25519(inner) = muxed.to_muxed_account() else {
                    prop_assert!(false, "a generated muxed address must encode as muxed");
                    return Ok(());
                };

                prop_assert_eq!(inner.id, muxed.id());
                prop_assert_eq!(inner.ed25519.0, *muxed.ed25519());
                Ok(())
            })
            .expect("every generated muxed address round trips");
    }

    #[test]
    fn a_muxed_address_shrinks_its_key_and_id_towards_zero() {
        let smallest = minimise(muxed_address(), |muxed| muxed.id() >= 7);

        assert_eq!(smallest.id(), 7);
        assert_eq!(*smallest.ed25519(), [0u8; 32]);
        assert_eq!(smallest.account().address_type(), AddressType::Account);
    }

    #[test]
    fn a_mixed_collection_holds_both_variants_and_respects_its_bounds() {
        let accounts = Cell::new(0usize);
        let contracts = Cell::new(0usize);

        let mut runner = TestRunner::new(ProptestConfig::with_cases(64));
        runner
            .run(&sc_address_collection(1..=6), |addresses| {
                prop_assert!((1..=6).contains(&addresses.len()));
                for address in &addresses {
                    match address {
                        ScAddress::Account(_) => accounts.set(accounts.get() + 1),
                        ScAddress::Contract(_) => contracts.set(contracts.get() + 1),
                        // Muxed accounts are in the mix too, but the two plain
                        // kinds are what the collection has to cover.
                        _ => {}
                    }
                }
                Ok(())
            })
            .expect("every generated collection is well formed");

        assert!(accounts.get() > 0, "no account address was generated");
        assert!(contracts.get() > 0, "no contract address was generated");
    }
}
