//! # Private Information Retrieval module
//!
//! DPFs imply a very simple 2-Server Private information retrieval scheme with running time linear in the DB size for each server for each query.
//! This module implements such a basic scheme for simple DBs made of bits.
//!
//!
//! ## PIR Variants
//!
//! ### Classic DPF-based PIR
//!
//! In this scheme the client uploads a DPF key to each server who then [evaluate the whole DPF](`crate::DpfKey<T>::eval_all`) ,computes the dot-product of the expanded vector with the DB and returns the result (a single bit) to the client.
//! This variant is also the most computationally-expensive as each server has to expand the DPF fully.
//!
//! ### Information Theoretic PIR
//!
//! This module also supports an information theoretic variant of PIR rather than a DPF-based one.
//! By letting the client generate two additive shares of a unit vector and uploading `sqrt(N)` elements to each server and downloading `sqrt(N)` elements from each server, an information theoretic PIR can be implemented.

const LOG_BITS_IN_BYTE: usize = 3;

pub mod information_theoretic {
    use super::LOG_BITS_IN_BYTE;
    use std::ops::Deref;
    use std::ops::DerefMut;
    use std::simd::u64x8;

    pub struct QueryIterator<'a> {
        q: &'a Query,
        index: usize,
    }
    impl<'a> QueryIterator<'a> {
        fn new(q: &'a Query) -> Self {
            QueryIterator { q, index: 0 }
        }
    }

    impl<'a> Iterator for QueryIterator<'a> {
        type Item = bool;
        fn next(&mut self) -> Option<Self::Item> {
            // if self.index == self.q.0.len() << LOG_BITS_IN_BYTE {
            //     return None;
            // }
            let (index, bit) = (self.index >> LOG_BITS_IN_BYTE, self.index & 7);
            let output = (self.q.0[index] >> bit) & 1 == 1;
            self.index += 1;
            Some(output)
        }
    }

    pub struct OwnedQuery(Vec<u8>);
    impl AsRef<Query> for OwnedQuery {
        fn as_ref(&self) -> &Query {
            Query::new(&self.0[..])
        }
    }

    impl Deref for OwnedQuery {
        type Target = Query;
        fn deref(&self) -> &Self::Target {
            Query::new(&self.0[..])
        }
    }

    impl DerefMut for OwnedQuery {
        fn deref_mut(&mut self) -> &mut Self::Target {
            Query::new_mut(&mut self.0[..])
        }
    }

    impl From<Vec<u8>> for OwnedQuery {
        fn from(v: Vec<u8>) -> Self {
            Self(v)
        }
    }

    #[repr(transparent)]
    pub struct Query([u8]);
    impl Query {
        pub fn new(q: &[u8]) -> &Self {
            // SAFETY: Query is a transparent wrapper around a byte slice
            unsafe { &*(q as *const [u8] as *const Self) }
        }
        pub fn new_mut(q: &mut [u8]) -> &mut Self {
            // SAFETY: Query is a transparent wrapper around a byte slice
            unsafe { &mut *(q as *mut [u8] as *mut Self) }
        }
        pub fn len_bits(&self) -> usize {
            self.0.len() << LOG_BITS_IN_BYTE
        }
        pub fn get_bit(&self, i: usize) -> bool {
            let (entry, bit_index) = Self::translate_to_coordinate(i);
            (self.0[entry] >> bit_index) & 1 != 0
        }

        fn translate_to_coordinate(i: usize) -> (usize, usize) {
            let entry = i >> LOG_BITS_IN_BYTE;
            let bit_index = i & ((1 << LOG_BITS_IN_BYTE) - 1);
            (entry, bit_index)
        }

        pub fn flip_bit(&mut self, i: usize) {
            let (entry, bit_index) = Self::translate_to_coordinate(i);
            self.0[entry] ^= 1 << bit_index;
        }

        pub fn set_bit(&mut self, i: usize, v: bool) {
            let (entry, bit_index) = Self::translate_to_coordinate(i);
            let cur_lane = &mut self.0[entry];
            *cur_lane ^= *cur_lane & (1 << bit_index) ^ (u8::from(v) << bit_index)
        }
        pub fn iter(&self) -> QueryIterator {
            QueryIterator::new(self)
        }
    }

    impl AsRef<[u8]> for Query {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }
    impl Deref for Query {
        type Target = [u8];
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    /// Given a random vector, generate the queries to the information theoretic two-server PIR scheme.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use std::simd::u64x8;
    /// use dpf::pir::information_theoretic::gen_query;
    /// use dpf::pir::information_theoretic::OwnedQuery;
    /// let random_query = OwnedQuery::from(vec![5u8;16]);
    /// let index = 0;
    /// let query = gen_query(index, &random_query, 512);
    /// assert_eq!(query[0]^random_query[0], 1u8);
    /// for i in 1..query.len() {
    ///     assert_eq!(query[i], random_query[i]);
    /// }
    /// ```
    pub fn gen_query(index: usize, random_query: &'_ Query, db_size: usize) -> OwnedQuery {
        let column_size = db_size / random_query.len_bits();
        let column_index = index / column_size;
        let mut output: OwnedQuery = OwnedQuery::from(random_query.to_vec());
        output.flip_bit(column_index);
        output
    }

    /// Given a database and a query, answer the query by XORing all the columns indicated by the query.
    pub fn answer_query(db: &[u64x8], query: &Query) -> Vec<u64x8> {
        let row_size = query.len_bits();
        let column_size = db.len() / row_size;

        // ensure db size is multiple of query size.
        assert_eq!(column_size * row_size, db.len());
        let mut output = vec![u64x8::default(); column_size];
        answer_query_into(db, query, &mut output);
        output
    }

    /// A non-allocating variant of [`answer_query()`]
    pub fn answer_query_into(db: &[u64x8], query: &Query, output: &mut [u64x8]) {
        let row_size = query.len_bits();
        let column_size = db.len() / row_size;

        // ensure db size is multiple of query size.
        assert_eq!(column_size * row_size, db.len());
        assert_eq!(output.len(), column_size);

        db.chunks(column_size)
            .zip(query.iter())
            .fold(output, |acc, (chunk, bit)| {
                if bit {
                    acc.iter_mut()
                        .zip(chunk.iter())
                        .for_each(|(acc, chunk)| *acc ^= chunk);
                }
                acc
            });
    }
}

pub mod dpf_based {
    use super::information_theoretic;
    use std::simd::u64x8;

    use crate::pir::information_theoretic::Query;
    use crate::pir::LOG_BITS_IN_BYTE;
    use crate::xor_slices;
    use crate::DpfKey;
    use crate::DPF_KEY_SIZE;

    /// Since DPF-based queries require some intermediate state, to avoid from allocating large vectors for each query a scratchpad struct is used as part of the API. Generate a default scratchpad using the [`Default`] trait.
    pub struct ResponseScratchpad<const DEPTH: usize> {
        single_dpf_output: Vec<[u8; DPF_KEY_SIZE]>,
        toggle_bits: Vec<bool>,
    }

    impl<const DEPTH: usize> Default for ResponseScratchpad<DEPTH> {
        fn default() -> Self {
            Self {
                single_dpf_output: vec![[0u8; DPF_KEY_SIZE]; 1 << DEPTH],
                toggle_bits: vec![bool::default(); 1 << DEPTH],
            }
        }
    }

    fn index_to_coordinates(index: usize) -> (usize, usize, usize) {
        let arr_idx = index / (DPF_KEY_SIZE << LOG_BITS_IN_BYTE);
        let lane_idx = index & ((DPF_KEY_SIZE << LOG_BITS_IN_BYTE) - 1) >> LOG_BITS_IN_BYTE;
        let bit_idx = index & ((1 << LOG_BITS_IN_BYTE) - 1);
        (arr_idx, lane_idx, bit_idx)
    }

    /// Generate the appropriate DPF keys to query the given index.
    pub fn gen_query<const DEPTH: usize>(
        index: usize,
        dpf_root_0: [u8; DPF_KEY_SIZE],
        dpf_root_1: [u8; DPF_KEY_SIZE],
        db_size_in_items: usize,
    ) -> (DpfKey<DEPTH>, DpfKey<DEPTH>) {
        let column_size = db_size_in_items / (DPF_KEY_SIZE << (DEPTH + LOG_BITS_IN_BYTE));
        let column_index = index / column_size;
        let (arr_idx, lane_idx, bit_idx) = index_to_coordinates(column_index);
        let mut hiding_value = [0u8; DPF_KEY_SIZE];
        hiding_value[lane_idx] = 1 << bit_idx;
        DpfKey::gen(arr_idx, &hiding_value, dpf_root_0, dpf_root_1)
    }

    /// Answer a single query with a scratchpad.
    pub fn answer_query_with_scratchpad<const DEPTH: usize>(
        db: &[u64x8],
        query: &DpfKey<DEPTH>,
        mut scratch: ResponseScratchpad<DEPTH>,
    ) -> (Vec<u64x8>, ResponseScratchpad<DEPTH>) {
        let columns_num = DPF_KEY_SIZE << (DEPTH + LOG_BITS_IN_BYTE);
        let mut output = vec![u64x8::default(); db.len() / columns_num];
        answer_query_into_with_scratchpad(db, query, &mut output[..], &mut scratch);
        (output, scratch)
    }

    /// A non allocating variant of [`answer_query_with_scratchpad`].
    pub fn answer_query_into_with_scratchpad<const DEPTH: usize>(
        db: &[u64x8],
        query: &DpfKey<DEPTH>,
        output: &mut [u64x8],
        scratch: &mut ResponseScratchpad<DEPTH>,
    ) {
        let columns_num = DPF_KEY_SIZE << (DEPTH + LOG_BITS_IN_BYTE);
        assert_eq!(output.len() * columns_num, db.len());
        let output_dpf = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.single_dpf_output.as_mut_ptr().cast(),
                1 << DEPTH,
            )
        };
        query.eval_all_into(output_dpf, &mut scratch.toggle_bits);
        let scratch_slice = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.single_dpf_output.as_mut_ptr().cast(),
                DPF_KEY_SIZE << DEPTH,
            )
        };
        let q = Query::new(scratch_slice);
        information_theoretic::answer_query_into(db, q, output);
    }

    /// Answer a single query.
    pub fn answer_query<const DEPTH: usize>(db: &[u64x8], query: &DpfKey<DEPTH>) -> Vec<u64x8> {
        // Each bit in the output of the DPF refers to a column in the DB.
        let columns_num = DPF_KEY_SIZE << (DEPTH + LOG_BITS_IN_BYTE);
        let mut output = vec![u64x8::default(); db.len() / columns_num];
        answer_query_into(db, query, &mut output[..]);
        output
    }
    /// A non allocating variant of [`answer_query`].
    pub fn answer_query_into<const DEPTH: usize>(
        db: &[u64x8],
        query: &DpfKey<DEPTH>,
        output: &mut [u64x8],
    ) {
        let columns_num = DPF_KEY_SIZE << (DEPTH + LOG_BITS_IN_BYTE);
        assert_eq!(output.len() * columns_num, db.len());
        for (selector_bit, column) in query.bit_iter().zip(db.chunks(output.len())) {
            if selector_bit {
                xor_slices(output, column)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LOG_BITS_IN_BYTE;
    use crate::pir::dpf_based::{
        answer_query, answer_query_with_scratchpad, gen_query, ResponseScratchpad,
    };
    use crate::DPF_KEY_SIZE;
    use std::simd::u64x8;

    #[test]
    pub fn test_pir() {
        const LOG_BITS_IN_U64: usize = 6;
        const LOG_DB_SZ: usize = 25;
        const DB_SZ: usize = 1 << LOG_DB_SZ;
        const DPF_DEPTH: usize = 6;
        const QUERY_INDEX: usize = 512;
        let entry_size = std::mem::size_of::<u64x8>() << LOG_BITS_IN_BYTE;
        let db: Vec<_> = (0..(DB_SZ / entry_size))
            .map(|i| u64x8::splat(i as u64))
            .collect();
        let number_of_columns = DPF_KEY_SIZE << (LOG_BITS_IN_BYTE + DPF_DEPTH);
        let column_size_in_bits = DB_SZ / number_of_columns;
        let item_index = QUERY_INDEX / entry_size;
        let item_index_in_column = item_index % column_size_in_bits;
        let lane_index = (QUERY_INDEX % entry_size) >> LOG_BITS_IN_U64;
        let bit_index = QUERY_INDEX & ((1 << LOG_BITS_IN_U64) - 1);

        let dpf_root_0 = [1u8; DPF_KEY_SIZE];
        let dpf_root_1 = [2u8; DPF_KEY_SIZE];
        let (k_0, k_1) = gen_query::<DPF_DEPTH>(item_index, dpf_root_0, dpf_root_1, db.len());
        let scratch = ResponseScratchpad::default();
        let (output_0, scratch) = answer_query_with_scratchpad(&db, &k_0, scratch);
        let (output_1, _) = answer_query_with_scratchpad(&db, &k_1, scratch);
        let output_0_dfs = answer_query(&db, &k_0);
        assert_eq!(output_0, output_0_dfs);

        assert_eq!(
            ((output_0[item_index_in_column][lane_index]
                ^ output_1[item_index_in_column][lane_index])
                >> bit_index)
                & 1,
            (db[item_index][lane_index] >> bit_index) & 1
        );
    }
}
