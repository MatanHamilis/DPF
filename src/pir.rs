//! # Private Information Retrieval module
//!
//! DPFs imply a very simple 2-Server Private information retrieval scheme with running time linear in the DB size for each server for each query.
//! This module implements such a basic scheme for simple DBs made of bits.
//!
//! ## Batching
//!
//! Current implementation supports *errorless batching*, meaning that multiple queries can be evaluated in a batch with a perfect success probability of responding to all of the queries correctly.
//! Since answering each query requires iterating through the whole database, errorless batching achieves greater throughput by exploiting the same iteration over the DB to answer multiple queries at once, better utilizing the memory bandwidth.
//! Check out [`answer_query_batched`] to use this feature.
//!
//! ## PIR Variants
//!
//! ### Classic DPF-based PIR
//!
//! In this scheme the client uploads a DPF key to each server who then [evaluate the whole DPF](`DpfKey<T>::eval_all()`) ,computes the dot-product of the expanded vector with the DB and returns the result (a single bit) to the client.
//! This variant is also the most computationally-expensive as each server has to expand the DPF fully.
//!
//! ### Information Theoretic PIR
//!
//! This module also supports an information theoretic variant of PIR rather than a DPF-based one.
//! By letting the client generate two additive shares of a unit vector and uploading `sqrt(N)` elements to each server and downloading `sqrt(N)` elements from each server, an information theoretic PIR can be implemented.

use std::simd::u64x8;

const BITS_IN_BYTE: usize = 8;
const LOG_BITS_IN_BYTE: usize = 3;

trait PirScheme<QueryType> {
    fn gen_query(index: usize) -> (QueryType, QueryType);
    fn answer_query(db: &[u64x8], query: &QueryType);
    fn answer_query_batched(db: &[u64x8], query: &QueryType);
}

pub mod information_theoretic {
    use std::simd::u64x8;

    /// Given a random vector, generate the queries to the information theoretic two-server PIR scheme.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use std::simd::u64x8;
    /// use dpf::pir::information_theoretic::gen_query;
    /// let random_vector = vec![u64x8::splat(5)];
    /// let index = 0;
    /// let query = gen_query(index, &random_vector[..]);
    /// assert_eq!(random_vector[0][0]^query[0][0], 1u64);
    /// for i in 1..u64x8::LANES {
    ///     assert_eq!(random_vector[0][i]^query[0][i], 0);
    /// }
    /// ```
    pub fn gen_query(index: usize, random_vector: &[u64x8]) -> Vec<u64x8> {
        let entry = index >> 9;
        let lane = (index & ((1 << 9) - 1)) / 64;
        let bit_index = index & 63;
        assert!(entry < random_vector.len());
        let mut output = Vec::from(random_vector);
        output[entry][lane] ^= 1 << bit_index;
        output
    }

    /// Given a database and a query, answer the query by multiplying the query by the columns of database.
    pub fn answer_query(db: &[u64x8], query: &[u64x8]) -> Vec<u64x8> {
        let answer_size = db.len() / query.len();

        // ensure db size is multiple of query size.
        assert_eq!(answer_size * query.len(), db.len());

        let queries_transmuted = unsafe { std::mem::transmute(query) };
        let output = answer_query_batched::<1>(db, queries_transmuted);
        unsafe { std::mem::transmute(output) }
    }

    /// A non-allocating variant of [`answer_query()`]
    pub fn answer_query_into(db: &[u64x8], query: &[u64x8], output: &mut [u64x8]) {
        assert_eq!(output.len() * query.len(), db.len());
        let output_transmuted = unsafe { std::mem::transmute(output) };
        let query_trnasmuted = unsafe { std::mem::transmute(query) };
        answer_query_batched_into::<1>(db, query_trnasmuted, output_transmuted);
    }

    /// A batched variant of [`answer_query()`]
    pub fn answer_query_batched<const BATCH: usize>(
        db: &[u64x8],
        queries: &[[u64x8; BATCH]],
    ) -> Vec<[u64x8; BATCH]> {
        let output_size = db.len() / queries.len();
        assert_eq!(queries.len() * output_size, db.len());
        let mut output = vec![[u64x8::default(); BATCH]; output_size];
        answer_query_batched_into(db, queries, &mut output[..]);
        output
    }

    /// A non-allocating variant batched of [`answer_query_batched()`]
    pub fn answer_query_batched_into<const BATCH: usize>(
        db: &[u64x8],
        queries: &[[u64x8; BATCH]],
        output: &mut [[u64x8; BATCH]],
    ) {
        assert_eq!(output.len() * queries.len(), db.len());
        output
            .iter_mut()
            .zip(db.chunks(queries.len()))
            .for_each(|(output_item, db_chunk)| {
                *output_item = [u64x8::default(); BATCH];
                queries
                    .iter()
                    .zip(db_chunk.iter())
                    .for_each(|(queries_item, db_item)| {
                        queries_item.iter().zip(output_item.iter_mut()).for_each(
                            |(query_inner_item, output_inner_item)| {
                                *output_inner_item ^= query_inner_item & db_item;
                            },
                        )
                    })
            })
    }
}

pub mod dpf_based {
    use super::information_theoretic;
    use std::mem::size_of;
    use std::simd::u64x8;

    use crate::pir::{BITS_IN_BYTE, LOG_BITS_IN_BYTE};
    use crate::DpfKey;
    use crate::DPF_KEY_SIZE;

    /// Since DPF-based queries require some intermediate state, to avoid from allocating large vectors for each query a scratchpad struct is used as part of the API. Generate a default scratchpad using the [`Default`] trait.
    pub struct ResponseScratchpad<const BATCH: usize, const DEPTH: usize> {
        pub batched_query: Vec<[u64x8; BATCH]>,
        single_dpf_output: Vec<u64x8>,
        toggle_bits: Vec<bool>,
    }

    impl<const BATCH: usize, const DEPTH: usize> Default for ResponseScratchpad<BATCH, DEPTH> {
        fn default() -> Self {
            Self {
                batched_query: vec![
                    [u64x8::default(); BATCH];
                    (DPF_KEY_SIZE << DEPTH) / size_of::<u64x8>()
                ],
                single_dpf_output: vec![
                    u64x8::default();
                    (DPF_KEY_SIZE << DEPTH) / size_of::<u64x8>()
                ],
                toggle_bits: vec![bool::default(); 1 << DEPTH],
            }
        }
    }

    /// Generate the appropriate DPF keys to query the given index.
    pub fn gen_query<const DEPTH: usize>(
        index: usize,
        dpf_root_0: [u8; DPF_KEY_SIZE],
        dpf_root_1: [u8; DPF_KEY_SIZE],
    ) -> (DpfKey<DEPTH>, DpfKey<DEPTH>) {
        let arr_index = (index & ((DPF_KEY_SIZE * BITS_IN_BYTE) - 1)) >> LOG_BITS_IN_BYTE;
        let cell_index = index & (BITS_IN_BYTE - 1);
        let mut hiding_value = [0u8; DPF_KEY_SIZE];
        hiding_value[arr_index] = 1 << cell_index;
        DpfKey::gen(
            index / (DPF_KEY_SIZE * BITS_IN_BYTE),
            &hiding_value,
            dpf_root_0,
            dpf_root_1,
        )
    }

    fn batch_dpf_queries_into<const BATCH: usize, const DEPTH: usize>(
        query: &[DpfKey<DEPTH>; BATCH],
        scratch: &mut ResponseScratchpad<BATCH, DEPTH>,
    ) {
        assert_eq!(
            scratch.batched_query.len() * size_of::<u64x8>(),
            DPF_KEY_SIZE << DEPTH
        );
        for (key_idx, key) in query.iter().enumerate() {
            let scratch_output: &mut [[u8; DPF_KEY_SIZE]] = unsafe {
                std::slice::from_raw_parts_mut(
                    scratch.single_dpf_output.as_mut_ptr().cast(),
                    scratch.single_dpf_output.len()
                        * (size_of::<u64x8>() / size_of::<[u8; DPF_KEY_SIZE]>()),
                )
            };
            key.eval_all_into(scratch_output, &mut scratch.toggle_bits);
            for (output_item, scratch_item) in scratch
                .batched_query
                .iter_mut()
                .zip(scratch.single_dpf_output.iter())
            {
                output_item[key_idx] = *scratch_item;
            }
        }
    }

    /// Answer a single query,
    pub fn answer_query<const DEPTH: usize>(
        db: &[u64x8],
        query: &DpfKey<DEPTH>,
        scratch: &mut ResponseScratchpad<1, DEPTH>,
    ) -> Vec<u64x8> {
        let mut output = vec![u64x8::default(); db.len() / scratch.batched_query.len()];
        answer_query_into(db, query, &mut output[..], scratch);
        output
    }

    /// A non allocating variant of [`answer_query`].
    pub fn answer_query_into<const DEPTH: usize>(
        db: &[u64x8],
        query: &DpfKey<DEPTH>,
        output: &mut [u64x8],
        scratch: &mut ResponseScratchpad<1, DEPTH>,
    ) {
        assert_eq!(output.len() * scratch.batched_query.len(), db.len());
        let output_transmuted = unsafe { std::mem::transmute(output) };
        answer_query_batched_into::<1, DEPTH>(db, &[*query], output_transmuted, scratch);
    }

    /// A batched variant of [`answer_query`].
    pub fn answer_query_batched<const BATCH: usize, const DEPTH: usize>(
        db: &[u64x8],
        queries: &[DpfKey<DEPTH>; BATCH],
        scratch: &mut ResponseScratchpad<BATCH, DEPTH>,
    ) -> Vec<[u64x8; BATCH]> {
        let output_size = db.len() / scratch.batched_query.len();
        assert_eq!(scratch.batched_query.len() * output_size, db.len());
        let mut output = vec![[u64x8::default(); BATCH]; output_size];
        answer_query_batched_into(db, queries, &mut output[..], scratch);
        output
    }

    /// A non-allocating variant of [`answer_query_batched`].
    pub fn answer_query_batched_into<const BATCH: usize, const DEPTH: usize>(
        db: &[u64x8],
        queries: &[DpfKey<DEPTH>; BATCH],
        output: &mut [[u64x8; BATCH]],
        scratch: &mut ResponseScratchpad<BATCH, DEPTH>,
    ) {
        assert_eq!(output.len() * scratch.batched_query.len(), db.len());
        batch_dpf_queries_into(queries, scratch);
        information_theoretic::answer_query_batched_into(db, &scratch.batched_query, output);
    }
}

#[cfg(test)]
mod tests {
    use crate::pir::dpf_based::{answer_query, gen_query, ResponseScratchpad};
    use crate::DPF_KEY_SIZE;
    use std::simd::u64x8;

    #[test]
    pub fn test_pir() {
        const LOG_DB_SZ: usize = 33;
        const DB_SZ: usize = 1 << LOG_DB_SZ;
        const DPF_DEPTH: usize = 12;
        const QUERY_INDEX: usize = 512;
        let entry_size = 8 * std::mem::size_of::<u64x8>();
        let lane_size = 8 * std::mem::size_of::<u64>();
        let array_index = QUERY_INDEX / entry_size;
        let lane_index = (QUERY_INDEX % entry_size) / (lane_size);
        let bit_index = QUERY_INDEX % lane_size;
        let db: Vec<_> = (0..(DB_SZ / entry_size))
            .map(|i| u64x8::splat(i as u64))
            .collect();
        let dpf_root_0 = [1u8; DPF_KEY_SIZE];
        let dpf_root_1 = [2u8; DPF_KEY_SIZE];
        let (k_0, k_1) = gen_query::<DPF_DEPTH>(QUERY_INDEX, dpf_root_0, dpf_root_1);
        let mut scratch = ResponseScratchpad::default();
        let output_0 = answer_query(&db, &k_0, &mut scratch);
        let output_1 = answer_query(&db, &k_1, &mut scratch);

        assert_eq!(
            ((output_0[array_index][lane_index] ^ output_1[array_index][lane_index]) >> bit_index)
                & 1,
            (db[array_index][lane_index] >> bit_index) & 1
        );
    }
}
