#![feature(portable_simd)]

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use dpf::pir::dpf_based::{self, ResponseScratchpad};
use dpf::pir::information_theoretic;
use dpf::{DpfKey, DPF_KEY_SIZE};
use std::mem::MaybeUninit;
use std::simd::u64x8;

fn dpf_gen<const DEPTH: usize>() -> (DpfKey<DEPTH>, DpfKey<DEPTH>) {
    let hiding_point = 0;
    let mut point_val = [0u8; DPF_KEY_SIZE];
    point_val[0] = 1;
    let dpf_root_0 = [1u8; DPF_KEY_SIZE];
    let dpf_root_1 = [2u8; DPF_KEY_SIZE];
    DpfKey::gen(hiding_point, &point_val, dpf_root_0, dpf_root_1)
}

fn dpf_keygen_bench<const DEPTH: usize>(c: &mut Criterion) {
    let i = DEPTH;
    c.bench_with_input(BenchmarkId::new("dpf_keygen", DEPTH), &i, |b, &_s| {
        let hiding_point = 0;
        let mut point_val = [0u8; DPF_KEY_SIZE];
        point_val[0] = 1;
        let dpf_root_0 = [1u8; DPF_KEY_SIZE];
        let dpf_root_1 = [2u8; DPF_KEY_SIZE];
        b.iter(|| {
            black_box(DpfKey::<DEPTH>::gen(
                hiding_point,
                &point_val,
                dpf_root_0,
                dpf_root_1,
            ))
        });
    });
}

fn dpf_evalall_bench<const DEPTH: usize>(c: &mut Criterion) {
    let i = DEPTH;
    c.bench_with_input(BenchmarkId::new("dpf_evalall", DEPTH), &i, |b, &_s| {
        let (k_0, _) = dpf_gen::<DEPTH>();
        let mut output = vec![[0u8; DPF_KEY_SIZE]; 1 << DEPTH];
        let mut aux = vec![false; 1 << DEPTH];
        b.iter(|| black_box(k_0.eval_all_into(&mut output, &mut aux)));
    });
}

pub fn bench_dpf_gen(g: &mut Criterion) {
    dpf_keygen_bench::<3>(g);
    dpf_keygen_bench::<4>(g);
    dpf_keygen_bench::<5>(g);
    dpf_keygen_bench::<6>(g);
    dpf_keygen_bench::<7>(g);
    dpf_keygen_bench::<8>(g);
    dpf_keygen_bench::<9>(g);
    dpf_keygen_bench::<10>(g);
    dpf_keygen_bench::<11>(g);
    dpf_keygen_bench::<12>(g);
    dpf_keygen_bench::<13>(g);
    dpf_keygen_bench::<14>(g);
    dpf_keygen_bench::<15>(g);
    dpf_keygen_bench::<16>(g);
    dpf_keygen_bench::<17>(g);
    dpf_keygen_bench::<18>(g);
    dpf_keygen_bench::<19>(g);
    dpf_keygen_bench::<20>(g);
}
pub fn bench_dpf_evalall(g: &mut Criterion) {
    dpf_evalall_bench::<3>(g);
    dpf_evalall_bench::<4>(g);
    dpf_evalall_bench::<5>(g);
    dpf_evalall_bench::<6>(g);
    dpf_evalall_bench::<7>(g);
    dpf_evalall_bench::<8>(g);
    dpf_evalall_bench::<9>(g);
    dpf_evalall_bench::<10>(g);
    dpf_evalall_bench::<11>(g);
    dpf_evalall_bench::<12>(g);
    dpf_evalall_bench::<13>(g);
    dpf_evalall_bench::<14>(g);
    dpf_evalall_bench::<15>(g);
    dpf_evalall_bench::<16>(g);
    dpf_evalall_bench::<17>(g);
    dpf_evalall_bench::<18>(g);
    dpf_evalall_bench::<19>(g);
    dpf_evalall_bench::<20>(g);
}
pub fn bench_pir_single_info_theoretic<const DPF_DEPTH: usize, const BATCH: usize>(
    c: &mut BenchmarkGroup<WallTime>,
    db: &[u64x8],
    query_index: usize,
) {
    let dpf_root_0 = [1u8; DPF_KEY_SIZE];
    let dpf_root_1 = [2u8; DPF_KEY_SIZE];
    let mut keys_0: [MaybeUninit<DpfKey<DPF_DEPTH>>; BATCH] =
        unsafe { [MaybeUninit::uninit().assume_init(); BATCH] };
    let mut keys_1: [MaybeUninit<DpfKey<DPF_DEPTH>>; BATCH] =
        unsafe { [MaybeUninit::uninit().assume_init(); BATCH] };
    for i in 0..BATCH {
        let (k_0, k_1) =
            dpf::pir::dpf_based::gen_query::<DPF_DEPTH>(query_index, dpf_root_0, dpf_root_1);
        keys_0[i].write(k_0);
        keys_1[i].write(k_1);
    }
    let keys_0 = unsafe { keys_0.as_ptr().cast::<[DpfKey<DPF_DEPTH>; BATCH]>().read() };
    let keys_1 = unsafe { keys_1.as_ptr().cast::<[DpfKey<DPF_DEPTH>; BATCH]>().read() };

    let mut output_0 = vec![
        [u64x8::default(); BATCH];
        (db.len() >> DPF_DEPTH) * std::mem::size_of::<u64x8>() / DPF_KEY_SIZE
    ];
    let mut output_1 = vec![
        [u64x8::default(); BATCH];
        (db.len() >> DPF_DEPTH) * std::mem::size_of::<u64x8>() / DPF_KEY_SIZE
    ];
    let mut scratchpad_0 = ResponseScratchpad::default();
    let mut scratchpad_1 = ResponseScratchpad::default();
    let batch = BATCH;
    c.throughput(Throughput::Bytes(
        u64::try_from(db.len() * std::mem::size_of::<u64x8>() * BATCH).unwrap(),
    ));
    c.bench_with_input(
        BenchmarkId::new("information_theoretic_pir_batch", BATCH),
        &batch,
        |b, _param| {
            b.iter(|| {
                information_theoretic::answer_query_batched_into(
                    db,
                    &scratchpad_0.batched_query,
                    &mut output_0[..],
                );
                information_theoretic::answer_query_batched_into(
                    db,
                    &scratchpad_1.batched_query,
                    &mut output_1[..],
                );
            });
        },
    );
    c.bench_with_input(
        BenchmarkId::new("dpf_pir_batch", BATCH),
        &batch,
        |b, _param| {
            b.iter(|| {
                dpf_based::answer_query_batched_into(db, &keys_0, &mut output_0, &mut scratchpad_0);
                dpf_based::answer_query_batched_into(db, &keys_1, &mut output_1, &mut scratchpad_1);
            });
        },
    );
}
pub fn bench_pir(c: &mut Criterion) {
    const LOG_DB_SZ: usize = 33;
    const DB_SZ: usize = 1 << LOG_DB_SZ;
    const DPF_DEPTH: usize = 11;
    const QUERY_INDEX: usize = 257;
    let mut g = c.benchmark_group("pir_batch");
    let db: Vec<_> = (0..(DB_SZ / (std::mem::size_of::<u64x8>() * 8)))
        .map(|i| u64x8::splat(i as u64))
        .collect();
    bench_pir_single_info_theoretic::<DPF_DEPTH, 1>(&mut g, &db, QUERY_INDEX);
    bench_pir_single_info_theoretic::<DPF_DEPTH, 2>(&mut g, &db, QUERY_INDEX);
    bench_pir_single_info_theoretic::<DPF_DEPTH, 4>(&mut g, &db, QUERY_INDEX);
    bench_pir_single_info_theoretic::<DPF_DEPTH, 8>(&mut g, &db, QUERY_INDEX);
    bench_pir_single_info_theoretic::<DPF_DEPTH, 16>(&mut g, &db, QUERY_INDEX);
    bench_pir_single_info_theoretic::<DPF_DEPTH, 32>(&mut g, &db, QUERY_INDEX);
    g.finish();
}

criterion_group!(benches, bench_pir, bench_dpf_gen, bench_dpf_evalall);
criterion_main!(benches);
