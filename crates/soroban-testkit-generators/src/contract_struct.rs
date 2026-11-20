//! Declarative strategies for the `#[contracttype]` structs and enums a
//! contract defines.
//!
//! A Soroban contract rarely takes scalars only: an invoice arrives as a
//! `#[contracttype]` struct of line items, each of those holding a memo, a
//! vector of child structs and maybe an enum tag. Writing the proptest strategy
//! for that tree by hand is the same seven lines every time — a tuple of field
//! strategies and a `prop_map` into the struct — and getting it wrong is easy
//! when a child struct gains a field.
//!
//! [`prop_contract_struct!`] and [`prop_contract_enum!`] generate those
//! strategies from the field list, so the struct stays the single source of
//! truth: add a field and the macro stops compiling until the strategy names a
//! strategy for it.
//!
//! # Structs
//!
//! ```rust
//! use proptest::prelude::*;
//! use soroban_sdk::Env;
//! use soroban_testkit_generators::{prop_contract_struct, strategies::token_amount};
//!
//! #[derive(Clone, Debug, PartialEq)]
//! struct LineItem {
//!     quantity: u32,
//!     unit_price: i128,
//! }
//!
//! prop_contract_struct! {
//!     LineItem as line_item(env) {
//!         quantity: 0u32..=1_000,
//!         unit_price: token_amount(),
//!     }
//! }
//!
//! // `line_item(env)` is a `Strategy<Value = LineItem>` that can be nested in
//! // any other strategy, including another `prop_contract_struct!` field.
//! fn cart(env: &Env) -> impl Strategy<Value = Vec<LineItem>> {
//!     proptest::collection::vec(line_item(env), 0..=8)
//! }
//! ```
//!
//! The generated function takes `&Env` because most Soroban field types — a
//! `Symbol`, a `Vec`, a `Map`, an `Address` — do, and because that is how a
//! child struct's strategy is nested: `lines: prop::collection::vec(line_item(env), 0..=8)`.
//! The identifier in parentheses names that parameter, so a field strategy can
//! refer to the environment by the name it is given there; name it whatever
//! reads best in the invocation, as long as the field strategies agree.
//! Field strategies are ordinary proptest strategies, so any composition the
//! field needs (`prop::option::of` for an `Option<T>` field, `prop_oneof!` for
//! an enum field, `any::<[u8; 32]>()` for a hash) works unchanged.
//!
//! # Enums
//!
//! An enum variant with a payload is generated from that payload's strategy,
//! and a unit variant is generated with a single copy of itself:
//!
//! ```rust
//! use proptest::prelude::*;
//! use soroban_sdk::Env;
//! use soroban_testkit_generators::prop_contract_enum;
//!
//! #[derive(Clone, Debug, PartialEq)]
//! enum Payment {
//!     Card(u32),
//!     Cash,
//! }
//!
//! prop_contract_enum! {
//!     Payment as payment(env) {
//!         Card: any::<u32>(),
//!         Cash,
//!     }
//! }
//!
//! // `payment(env)` yields either `Payment::Card(..)` or `Payment::Cash`.
//! ```
//!
//! # Shrinking
//!
//! The generated strategy is a tuple of field strategies followed by a
//! `prop_map`, so proptest shrinks it exactly as it shrinks that tuple: one
//! field at a time, each towards whatever its own strategy shrinks towards.
//! Integers and byte arrays go to zero, `prop::collection::vec` drops elements
//! and then simplifies the ones that remain, `prop::option::of` prefers `None`,
//! and a nested contract struct shrinks recursively through its own fields. A
//! failing `Invoice` therefore minimises to the smallest tree — fewest line
//! items, zeroed amounts, `None` optionals — that still reproduces the bug,
//! rather than to the random tree proptest first drew.
//!
//! Each macro accepts at most twelve fields, proptest's own tuple limit.

pub use proptest::strategy::Just;
pub use proptest::strategy::Strategy;
pub use soroban_sdk::Env;

/// Derive a proptest strategy for an existing struct from its field
/// strategies.
///
/// ```rust
/// use proptest::prelude::*;
/// use soroban_sdk::Env;
/// use soroban_testkit_generators::prop_contract_struct;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// prop_contract_struct! {
///     /// The strategy is documented like any other function.
///     pub Point as point(env) {
///         x: any::<i32>(),
///         y: any::<i32>(),
///     }
/// }
///
/// fn inside_the_unit_square(env: &Env) -> impl Strategy<Value = Point> {
///     point(env).prop_filter("inside the unit square", |p| p.x.abs() <= 1 && p.y.abs() <= 1)
/// }
/// ```
///
/// The syntax is `Type as strategy_fn { field: strategy, .. }`, where every
/// field of the struct must appear exactly once — a missing or misspelled
/// field is a compile error, which is what keeps the strategy and the struct in
/// sync. Visibility and doc comments on the strategy function come from the
/// optional attributes and visibility before the type name.
///
/// The generated function is
/// `fn strategy_fn(env: &Env) -> impl Strategy<Value = Type>`; the `env` is the
/// one the field strategies close over, so the generated value belongs to the
/// same environment as the contract under test.
#[macro_export]
macro_rules! prop_contract_struct {
    // One field: a bare strategy maps straight into the struct, without the
    // one-element tuple that would otherwise be involved.
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident as $strategy:ident ( $env:ident ) {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident : $field_strategy:expr
            ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis fn $strategy(
            $env: &$crate::contract_struct::Env,
        ) -> impl $crate::contract_struct::Strategy<Value = $name> {
            let _ = $env;
            $crate::__field_strategy!(
                $name,
                ( $($field: $field_strategy),+ )
            )
        }
    };
}

/// Derive a proptest strategy for an existing enum from its variants.
///
/// A variant written `Name: strategy` is constructed from that strategy's
/// value; a variant written just `Name` is a unit variant and yields itself.
/// The variants are combined with [`proptest::prop_oneof!`], so the strategy
/// shrinks towards the first variant that still reproduces the failure.
///
/// ```rust
/// use proptest::prelude::*;
/// use soroban_sdk::Env;
/// use soroban_testkit_generators::prop_contract_enum;
///
/// #[derive(Clone, Debug, PartialEq)]
/// enum Memo {
///     None,
///     Number(u64),
///     Text(u64),
/// }
///
/// prop_contract_enum! {
///     pub Memo as memo(env) {
///         None,
///         Number: any::<u64>(),
///         Text: 0u64..1_000,
///     }
/// }
/// ```
///
/// Soroban's `#[contracttype]` enums carry at most one payload value per
/// variant, so a variant names at most one strategy.
#[macro_export]
macro_rules! prop_contract_enum {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident as $strategy:ident ( $env:ident ) {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident $( : $variant_strategy:expr )?
            ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis fn $strategy(
            $env: &$crate::contract_struct::Env,
        ) -> impl $crate::contract_struct::Strategy<Value = $name> {
            let _ = $env;
            ::proptest::prop_oneof![
                $(
                    $crate::__variant_strategy!(
                        $name,
                        $variant
                        $(, $variant_strategy)?
                    )
                ),+
            ]
        }
    };
}

/// Builds a struct strategy from a `field: strategy` list.
///
/// Hidden implementation detail of [`prop_contract_struct!`]: proptest's
/// `Strategy` is implemented for tuples, and the tuple is mapped into the
/// struct with a trailing-comma pattern, which works for any field count.
#[doc(hidden)]
#[macro_export]
macro_rules! __field_strategy {
    ($name:ident, ( $($field:ident : $field_strategy:expr),+ )) => {{
        use $crate::contract_struct::Strategy as _;
        (
            $( $field_strategy, )+
        ).prop_map(|( $( $field, )+ )| $name { $( $field ),+ })
    }};
}

/// Builds one enum variant arm for [`prop_contract_enum!`].
///
/// Hidden implementation detail: the two arms are a payload variant and a unit
/// variant.
#[doc(hidden)]
#[macro_export]
macro_rules! __variant_strategy {
    ($name:ident, $variant:ident, $variant_strategy:expr) => {{
        use $crate::contract_struct::Strategy as _;
        ($variant_strategy).prop_map(|value| $name::$variant(value))
    }};
    ($name:ident, $variant:ident) => {
        $crate::contract_struct::Just($name::$variant)
    };
}

#[cfg(test)]
mod tests {
    use crate::strategies::{account_address, token_amount};
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use soroban_sdk::{contracttype, Address, Env, IntoVal, Map, Symbol, TryFromVal, Val, Vec};

    /// A strategy for a short, valid Soroban symbol.
    fn symbols(env: &Env) -> impl Strategy<Value = Symbol> {
        let env = env.clone();
        "[a-z]{1,8}".prop_map(move |name| Symbol::new(&env, &name))
    }

    /// A strategy for an account address owned by `env`.
    fn addresses(env: &Env) -> impl Strategy<Value = Address> {
        let env = env.clone();
        account_address().prop_map(move |address| address.to_address(&env))
    }

    /// A three-level tree: `Invoice` -> `Vec<LineItem>` -> `Option<Dimensions>`.
    #[contracttype]
    #[derive(Clone, Debug, PartialEq)]
    pub struct Dimensions {
        pub width: u32,
        pub height: u32,
    }

    #[contracttype]
    #[derive(Clone, Debug, PartialEq)]
    pub struct Invoice {
        pub id: u64,
        pub memo: Option<Symbol>,
        pub line: LineItem,
        pub tags: Vec<Address>,
        pub totals: Map<Address, i128>,
        pub surcharge: Surcharge,
    }

    #[contracttype]
    #[derive(Clone, Debug, PartialEq)]
    pub struct LineItem {
        pub sku: Symbol,
        pub quantity: u32,
        pub unit_price: i128,
        pub dimensions: Dimensions,
    }

    prop_contract_struct! {
        /// Uniformly sized boxes, the innermost level.
        pub Dimensions as dimensions(env) {
            width: 0u32..=4_096u32,
            height: 0u32..=4_096u32,
        }
    }

    prop_contract_struct! {
        /// One invoice line, with an optional child struct.
        pub LineItem as line_item(env) {
            sku: symbols(env),
            quantity: 0u32..=1_000u32,
            unit_price: token_amount(),
            dimensions: dimensions(env),
        }
    }

    prop_contract_struct! {
        /// The whole invoice: two more levels of struct, an SDK vector, an SDK
        /// map, an optional field and an enum field, in one strategy.
        pub Invoice as invoice(env) {
            id: any::<u64>(),
            memo: proptest::option::of(symbols(env)),
            line: line_item(env),
            tags: proptest::collection::vec(addresses(env), 0..=4).prop_map({
                let env = env.clone();
                move |tags| {
                    let mut vec = Vec::new(&env);
                    for tag in tags {
                        vec.push_back(tag);
                    }
                    vec
                }
            }),
            totals: proptest::collection::vec((addresses(env), token_amount()), 0..=4).prop_map({
                let env = env.clone();
                move |totals| {
                    let mut map = Map::new(&env);
                    for (symbol, amount) in totals {
                        map.set(symbol, amount);
                    }
                    map
                }
            }),
            surcharge: surcharge(env),
        }
    }

    /// A two-variant enum, one payload and one unit.
    #[contracttype]
    #[derive(Clone, Debug, PartialEq)]
    pub enum Surcharge {
        Percent(u32),
        None,
    }

    prop_contract_enum! {
        /// Either a percentage surcharge or none at all.
        pub Surcharge as surcharge(env) {
            Percent: 0u32..=10_000u32,
            None,
        }
    }

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

    #[test]
    fn a_three_level_nested_struct_is_a_valid_contract_value() {
        let env = Env::default();

        let mut runner = TestRunner::new(ProptestConfig::with_cases(32));
        runner
            .run(&invoice(&env), |invoice| {
                // The generated tree has to encode as a contract value: a
                // struct the host cannot represent is not a usable input.
                let encoded: Val = invoice.clone().into_val(&env);
                let decoded = <Invoice as TryFromVal<Env, Val>>::try_from_val(&env, &encoded)
                    .expect("a generated invoice decodes as a contract value");
                prop_assert_eq!(decoded, invoice);
                Ok(())
            })
            .expect("every generated invoice is a valid contract value");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn a_generated_invoice_respects_its_collection_bounds(
            invoice in invoice(&Env::default()),
        ) {
            prop_assert!(invoice.tags.len() <= 4);
            prop_assert!(invoice.totals.len() <= 4);
        }

        #[test]
        fn a_generated_enum_of_variants_stays_within_its_payload_range(
            surcharge in surcharge(&Env::default()),
        ) {
            match surcharge {
                Surcharge::Percent(percent) => prop_assert!(percent <= 10_000),
                Surcharge::None => {}
            }
        }
    }

    #[test]
    fn a_failing_nested_struct_shrinks_every_level_towards_zero() {
        // Only the outermost id matters; the nested children should simplify
        // to their cheapest value: no lines, no totals, no memo.
        let smallest = minimise(invoice(&Env::default()), |invoice| invoice.id >= 500);

        assert_eq!(smallest.id, 500);
        assert_eq!(smallest.tags.len(), 0);
        assert_eq!(smallest.totals.len(), 0);
        assert!(smallest.memo.is_none());
        assert_eq!(smallest.line.quantity, 0);
        assert_eq!(smallest.line.unit_price, 0);
        assert_eq!(smallest.line.dimensions.width, 0);
    }

    #[test]
    fn a_failing_child_struct_shrinks_its_own_fields() {
        let smallest = minimise(line_item(&Env::default()), |item| item.quantity >= 500);

        assert_eq!(smallest.quantity, 500);
        assert_eq!(smallest.unit_price, 0);
        assert_eq!(smallest.dimensions.width, 0);
        assert_eq!(smallest.dimensions.height, 0);
    }

    #[test]
    fn a_failing_grandchild_struct_shrinks_through_its_parent() {
        let smallest = minimise(line_item(&Env::default()), |item| {
            item.dimensions.width >= 1_000
        });

        assert_eq!(smallest.dimensions.width, 1_000);
        assert_eq!(smallest.dimensions.height, 0);
        assert_eq!(smallest.quantity, 0);
        assert_eq!(smallest.unit_price, 0);
    }
}
