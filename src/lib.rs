#![feature(portable_simd)]
#![doc = include_str!("../README.md")]
use std::{ops::BitXorAssign, u8};

use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128, Block,
};
use once_cell::sync::Lazy;

pub mod pir;
pub const DPF_KEY_SIZE: usize = 16;

const DPF_AES_KEY: [u8; DPF_KEY_SIZE] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2];
static AES: Lazy<Aes128> = Lazy::new(|| Aes128::new_from_slice(&DPF_AES_KEY).unwrap());

fn double_prg(input: &[u8; DPF_KEY_SIZE]) -> ([u8; DPF_KEY_SIZE], [u8; DPF_KEY_SIZE]) {
    let mut blocks = [Block::from(*input); 2];
    blocks[1][0] = !blocks[1][0];
    AES.encrypt_blocks(&mut blocks);
    xor_arrays(&mut blocks[0].into(), input);
    xor_arrays(&mut blocks[1].into(), input);
    blocks[1][0] = !blocks[1][0];
    unsafe {
        (
            std::mem::transmute(blocks[0]),
            std::mem::transmute(blocks[1]),
        )
    }
}

#[derive(Default, Clone, Copy)]
struct CorrectionWord {
    string: [u8; DPF_KEY_SIZE],
    bit_0: bool,
    bit_1: bool,
}

#[derive(Clone, Copy)]
pub struct DpfKey<const DEPTH: usize> {
    first_word: [u8; DPF_KEY_SIZE],
    first_toggle_bit: bool,
    corrections: [CorrectionWord; DEPTH],
    last_correction: [u8; DPF_KEY_SIZE],
}

fn expand_seed(seed: &[u8; DPF_KEY_SIZE]) -> ([u8; DPF_KEY_SIZE], bool, [u8; DPF_KEY_SIZE], bool) {
    let (mut scw_0, mut scw_1) = double_prg(seed);
    let toggle_bit_0 = scw_0[0] & 1;
    let toggle_bit_1 = scw_1[0] & 1;
    scw_0[0] &= u8::MAX - 1;
    scw_1[0] &= u8::MAX - 1;
    (scw_0, toggle_bit_0 == 0, scw_1, toggle_bit_1 == 0)
}
fn into_block(s: &mut [u8; DPF_KEY_SIZE]) {
    AES.encrypt_block(s.into());
}

fn into_blocks(s: &mut [[u8; DPF_KEY_SIZE]]) {
    let s = unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast(), s.len()) };
    AES.encrypt_blocks(s);
}
impl<const DEPTH: usize> DpfKey<DEPTH> {
    /// Generates a new a new pair of DPF keys to share a vector of size $2^`DEPTH`$.
    ///
    /// # Arguments
    ///
    /// * `hiding_point`: To index in the shared unit vector to be set to a non-zero value.
    /// * `point_val`: The non-zero value to be shared at index `hiding_point`.
    /// * `dpf_root_0`: The first DPF key pseudorandom seed.
    /// * `dpf_root_1`: The second DPF key pseudorandom seed.
    ///
    /// # Warning
    ///
    /// It is vital that the `dpf_root_0` and `dpf_root_1` are kept secret and are chosen using a cryptographic source of randomness.
    ///
    /// # Output
    ///
    /// Two DPF keys, that when `eval`-ed at `hiding_point` yield additive shares of `point_val` and at any other point yield additive shared of zero.
    ///
    pub fn gen(
        hiding_point: usize,
        point_val: &[u8; DPF_KEY_SIZE],
        dpf_root_0: [u8; DPF_KEY_SIZE],
        dpf_root_1: [u8; DPF_KEY_SIZE],
    ) -> (DpfKey<DEPTH>, DpfKey<DEPTH>) {
        let mut correction = [CorrectionWord::default(); DEPTH];
        let mut toggle_0 = false;
        let mut cw_0 = dpf_root_0;
        let mut cw_1 = dpf_root_1;

        // Transform the hiding point into bits.
        let hiding_point = usize_to_bits::<DEPTH>(hiding_point);
        for i in 0..DEPTH {
            let (s_0_l, t_0_l, s_0_r, t_0_r) = expand_seed(&cw_0);
            let (s_1_l, t_1_l, s_1_r, t_1_r) = expand_seed(&cw_1);
            let (mut s_0_lose, s_1_lose, mut s_0_keep, mut s_1_keep, t_0_keep, t_1_keep) =
                if hiding_point[i] {
                    (s_0_l, s_1_l, s_0_r, s_1_r, t_0_r, t_1_r)
                } else {
                    (s_0_r, s_1_r, s_0_l, s_1_l, t_0_l, t_1_l)
                };
            xor_arrays(&mut s_0_lose, &s_1_lose);
            let s_cw = s_0_lose;
            let (t_cw_l, t_cw_r) = (
                !(t_0_l ^ t_1_l ^ hiding_point[i]),
                t_0_r ^ t_1_r ^ hiding_point[i],
            );
            let t_cw_keep = !(t_0_keep ^ t_1_keep);
            correction[i] = CorrectionWord {
                string: s_cw,
                bit_0: t_cw_l,
                bit_1: t_cw_r,
            };
            (cw_0, cw_1, toggle_0) = if toggle_0 {
                xor_arrays(&mut s_0_keep, &s_cw);
                (s_0_keep, s_1_keep, t_0_keep ^ t_cw_keep)
            } else {
                xor_arrays(&mut s_1_keep, &s_cw);
                (s_0_keep, s_1_keep, t_0_keep)
            }
        }
        let mut last_correction = *point_val;
        into_block(&mut cw_0);
        into_block(&mut cw_1);
        xor_arrays(&mut last_correction, &cw_0);
        xor_arrays(&mut last_correction, &cw_1);
        (
            DpfKey {
                first_word: dpf_root_0,
                first_toggle_bit: false,
                corrections: correction,
                last_correction,
            },
            DpfKey {
                first_word: dpf_root_1,
                first_toggle_bit: true,
                corrections: correction,
                last_correction,
            },
        )
    }

    /// Evaluates a DPF key at a given `point`.
    ///
    /// # Example
    ///
    /// ```
    /// use dpf::DPF_KEY_SIZE;
    /// use dpf::DpfKey;
    /// const DEPTH:usize = 10;
    /// let hiding_point = 5;
    /// let point_val = [1u8; DPF_KEY_SIZE];
    /// let dpf_root_0 = [0u8; DPF_KEY_SIZE];
    /// let dpf_root_1 = [0u8; DPF_KEY_SIZE];
    /// let (key_0,key_1) = DpfKey::<DEPTH>::gen(hiding_point, &point_val, dpf_root_0, dpf_root_1);
    /// let key_0_eval = key_0.eval(hiding_point);
    /// let key_1_eval = key_1.eval(hiding_point);
    /// for i in 0..DPF_KEY_SIZE {
    ///     assert_eq!(key_0_eval[i]^key_1_eval[i], point_val[i]);
    /// }
    /// assert_eq!(key_0.eval(hiding_point+1), key_1.eval(hiding_point+1));
    /// ```
    pub fn eval(&self, point: usize) -> [u8; DPF_KEY_SIZE] {
        let point = usize_to_bits::<DEPTH>(point);
        let mut s = self.first_word;
        let mut t = self.first_toggle_bit;
        for (point_bit, correction_item) in point.into_iter().zip(self.corrections.iter()) {
            let expanded = expand_seed(&s);
            let (s_next, mut t_next) = if point_bit {
                (expanded.2, expanded.3)
            } else {
                (expanded.0, expanded.1)
            };
            s = s_next;
            if t {
                xor_arrays(&mut s, &correction_item.string);
                t_next ^= if point_bit {
                    correction_item.bit_1
                } else {
                    correction_item.bit_0
                }
            };
            t = t_next;
        }
        into_block(&mut s);
        if t {
            xor_arrays(&mut s, &self.last_correction);
        }
        s
    }
    /// Evaluates a DPF key at all points from 0 to 2^`DEPTH`-1.
    ///
    /// # Output
    ///
    /// Returns a vector whose `i`-th entry contains the evaluation of the DPF key at point `i`.
    ///
    /// # Example
    ///
    /// ```
    /// use dpf::DPF_KEY_SIZE;
    /// use dpf::DpfKey;
    /// const DEPTH:usize = 10;
    /// let hiding_point = 5;
    /// let point_val = [1u8; DPF_KEY_SIZE];
    /// let dpf_root_0 = [0u8; DPF_KEY_SIZE];
    /// let dpf_root_1 = [0u8; DPF_KEY_SIZE];
    /// let (key_0,key_1) = DpfKey::<DEPTH>::gen(hiding_point, &point_val, dpf_root_0, dpf_root_1);
    /// let key_0_eval = key_0.eval_all();
    /// let key_1_eval = key_1.eval_all();
    /// for i in 0..1<<DEPTH {
    ///     assert_eq!(key_0_eval[i], key_0.eval(i));
    /// }
    /// ```
    pub fn eval_all(&self) -> Vec<[u8; DPF_KEY_SIZE]> {
        let mut output = vec![[0u8; DPF_KEY_SIZE]; 1 << DEPTH];
        let mut aux = vec![false; 1 << DEPTH];
        self.eval_all_into(&mut output, &mut aux);
        output
    }
    /// Evaluates a DPF key at all points from 0 to 2^`DEPTH`-1.
    /// Writes the evaluations into a mutable slice `output`.
    /// The current algorithm requires also an auxilliary mutable boolean slice `aux`.
    /// This is the allocation-free variant of [`eval_all`](Self::eval_all) and may be found useful in case multiple calls [`eval_all`](Self::eval_all) are needed to different DPF keys where memory can be reused.
    ///
    /// # Output
    ///
    /// Evaluation at point `i` will be written to `output[i]`.
    ///
    /// # Notice
    ///
    /// * `output` and `aux` must be of length **exactly** 2^`DEPTH`.
    /// * The contents of `aux` may be discarded after the function returns.
    ///
    /// # Example
    ///
    /// ```
    /// use dpf::DPF_KEY_SIZE;
    /// use dpf::DpfKey;
    /// const DEPTH:usize = 10;
    /// let hiding_point = 5;
    /// let point_val = [1u8; DPF_KEY_SIZE];
    /// let dpf_root_0 = [0u8; DPF_KEY_SIZE];
    /// let dpf_root_1 = [0u8; DPF_KEY_SIZE];
    /// let (key_0,key_1) = DpfKey::<DEPTH>::gen(hiding_point, &point_val, dpf_root_0, dpf_root_1);
    /// let mut key_0_eval = vec![[0u8;DPF_KEY_SIZE]; 1<<DEPTH];
    /// let mut key_1_eval = vec![[0u8;DPF_KEY_SIZE]; 1<<DEPTH];
    /// let mut aux = vec![false; 1<<DEPTH];
    /// key_0.eval_all_into(&mut key_0_eval, &mut aux);
    /// key_1.eval_all_into(&mut key_1_eval, &mut aux);
    /// for i in 0..1<<DEPTH {
    ///     assert_eq!(key_0_eval[i], key_0.eval(i));
    /// }
    /// ```
    pub fn eval_all_into(&self, output: &mut [[u8; DPF_KEY_SIZE]], aux: &mut [bool]) {
        assert_eq!(output.len(), 1 << DEPTH);
        assert_eq!(aux.len(), 1 << DEPTH);
        output[0] = self.first_word;
        aux[0] = self.first_toggle_bit;
        for i in 0..DEPTH {
            for j in (0..1 << i).rev() {
                let (mut s_l, mut t_l, mut s_r, mut t_r) = expand_seed(&output[j]);
                if aux[j] {
                    xor_arrays(&mut s_l, &self.corrections[i].string);
                    xor_arrays(&mut s_r, &self.corrections[i].string);
                    t_l ^= self.corrections[i].bit_0;
                    t_r ^= self.corrections[i].bit_1;
                };
                output[2 * j] = s_l;
                aux[2 * j] = t_l;
                output[2 * j + 1] = s_r;
                aux[2 * j + 1] = t_r;
            }
        }
        into_blocks(output);
        for i in 0..1 << DEPTH {
            if aux[i] {
                xor_arrays(&mut output[i], &self.last_correction);
            }
        }
    }

    /// Creates an Iterator
    pub fn iter<'a>(&'a self) -> DpfIterator<'a, DEPTH> {
        let mut output = self.first_word;
        let mut aux = self.first_toggle_bit;
        let iterator_state: [[([u8; DPF_KEY_SIZE], bool); 2]; DEPTH] = std::array::from_fn(|i| {
            let (mut s_l, mut t_l, mut s_r, mut t_r) = expand_seed(&output);
            if aux {
                xor_arrays(&mut s_l, &self.corrections[i].string);
                xor_arrays(&mut s_r, &self.corrections[i].string);
                t_l ^= self.corrections[i].bit_0;
                t_r ^= self.corrections[i].bit_1;
            }
            output = s_l;
            aux = t_l;
            [(s_l, t_l), (s_r, t_r)]
        });
        DpfIterator {
            dpf_key: &self,
            iterator_state,
            iteration_path: [false; DEPTH],
            done: false,
        }
    }

    pub fn bit_iter<'a>(&'a self) -> DpfBitIterator<'a, DEPTH> {
        let mut iter = self.iter();
        let current_item = iter.next();
        DpfBitIterator {
            dpf_iterator: iter,
            current_item,
            bit_idx: 0,
        }
    }
}
pub struct DpfIterator<'a, const DEPTH: usize> {
    dpf_key: &'a DpfKey<DEPTH>,
    iterator_state: [[([u8; DPF_KEY_SIZE], bool); 2]; DEPTH],
    iteration_path: [bool; DEPTH],
    done: bool,
}

impl<'a, const DEPTH: usize> Iterator for DpfIterator<'a, DEPTH> {
    type Item = [u8; DPF_KEY_SIZE];
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // Prepare response
        let (mut leaf, aux) = match self.iteration_path[DEPTH - 1] {
            false => self.iterator_state[DEPTH - 1][0],
            true => self.iterator_state[DEPTH - 1][1],
        };
        into_block(&mut leaf);
        if aux {
            xor_arrays(&mut leaf, &self.dpf_key.last_correction);
        }

        // Fix state
        let mut fixing_start = DEPTH - 1;
        while self.iteration_path[fixing_start] {
            if fixing_start == 0 {
                self.done = true;
                return Some(leaf);
            }
            fixing_start -= 1;
        }
        self.iteration_path[fixing_start] = true;
        let (mut output, mut aux) = self.iterator_state[fixing_start][1];
        for i in fixing_start + 1..DEPTH {
            self.iteration_path[i] = false;
            let (mut s_l, mut t_l, mut s_r, mut t_r) = expand_seed(&output);
            if aux {
                xor_arrays(&mut s_l, &self.dpf_key.corrections[i].string);
                xor_arrays(&mut s_r, &self.dpf_key.corrections[i].string);
                t_l ^= self.dpf_key.corrections[i].bit_0;
                t_r ^= self.dpf_key.corrections[i].bit_1;
            }
            output = s_l;
            aux = t_l;
            self.iterator_state[i] = [(s_l, t_l), (s_r, t_r)];
        }

        Some(leaf)
    }
}
pub struct DpfBitIterator<'a, const DEPTH: usize> {
    dpf_iterator: DpfIterator<'a, DEPTH>,
    current_item: Option<[u8; DPF_KEY_SIZE]>,
    bit_idx: usize,
}

impl<'a, const DEPTH: usize> Iterator for DpfBitIterator<'a, DEPTH> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_item == None {
            return None;
        }
        let item = ((self.current_item.unwrap()[self.bit_idx >> 3] >> (self.bit_idx & 7)) & 1) == 1;
        self.bit_idx += 1;
        self.bit_idx &= (DPF_KEY_SIZE << 3) - 1;
        if self.bit_idx == 0 {
            self.current_item = self.dpf_iterator.next();
        }
        Some(item)
    }
}

fn xor_arrays<const LENGTH: usize, T: Copy + BitXorAssign>(
    lhs: &mut [T; LENGTH],
    rhs: &[T; LENGTH],
) {
    for i in 0..LENGTH {
        lhs[i] ^= rhs[i];
    }
}
fn xor_slices<T: Copy + BitXorAssign>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut().zip(rhs.iter()).for_each(|(l, r)| {
        *l ^= *r;
    })
}

fn usize_to_bits<const SIZE: usize>(num: usize) -> [bool; SIZE] {
    core::array::from_fn(|i| ((num >> (SIZE - 1 - i)) & 1) == 1)
}

#[cfg(test)]
mod tests {
    use super::DpfKey;
    use super::DPF_KEY_SIZE;
    use crate::xor_arrays;
    #[test]
    fn test_dpf_single_point() {
        const DEPTH: usize = 10;
        let hiding_point = 0b100110;
        let mut point_val = [2u8; DPF_KEY_SIZE];
        let dpf_root_0 = [0u8; DPF_KEY_SIZE];
        let dpf_root_1 = [1u8; DPF_KEY_SIZE];
        point_val[0] = 1;
        let (k_0, k_1) = DpfKey::<DEPTH>::gen(hiding_point, &point_val, dpf_root_0, dpf_root_1);
        for i in 0..1 << DEPTH {
            let mut k_0_eval = k_0.eval(i);
            let k_1_eval = k_1.eval(i);
            xor_arrays(&mut k_0_eval, &k_1_eval);
            if i != hiding_point {
                assert_eq!(k_0_eval, [0u8; DPF_KEY_SIZE]);
            } else {
                assert_eq!(k_0_eval, point_val);
            }
        }
    }
    #[test]
    fn test_dpf_eval_all() {
        const DEPTH: usize = 20;
        const HIDING_POINT: usize = 0b100110;
        let mut point_val = [2u8; DPF_KEY_SIZE];
        let dpf_root_0 = [0u8; DPF_KEY_SIZE];
        let dpf_root_1 = [1u8; DPF_KEY_SIZE];
        point_val[0] = 1;
        let (k_0, k_1) = DpfKey::<DEPTH>::gen(HIDING_POINT, &point_val, dpf_root_0, dpf_root_1);
        let eval_all_0 = k_0.eval_all();
        let eval_all_1 = k_1.eval_all();
        for i in 0..1 << DEPTH {
            let mut k_0_eval = eval_all_0[i];
            let k_1_eval = eval_all_1[i];
            xor_arrays(&mut k_0_eval, &k_1_eval);
            if i != HIDING_POINT {
                assert_eq!(k_0_eval, [0u8; DPF_KEY_SIZE]);
            } else {
                assert_eq!(k_0_eval, point_val);
            }
        }
    }

    #[test]
    fn test_dpf_iterator() {
        const DEPTH: usize = 12;
        const HIDING_POINT: usize = 0b100110;
        let mut point_val = [2u8; DPF_KEY_SIZE];
        let dpf_root_0 = [0u8; DPF_KEY_SIZE];
        let dpf_root_1 = [1u8; DPF_KEY_SIZE];
        point_val[0] = 1;
        let (k_0, _) = DpfKey::<DEPTH>::gen(HIDING_POINT, &point_val, dpf_root_0, dpf_root_1);
        let eval_all_0 = k_0.eval_all();
        let eval_all_0_iter: Vec<_> = k_0.iter().collect();
        assert_eq!(eval_all_0.len(), eval_all_0_iter.len());
        assert_eq!(eval_all_0_iter, eval_all_0);
    }
}
